# Third-Party
import cv2 as cv
import numpy as np
from pyscrcpy import Client, const
import time
import os
import keyboard

# Type Hints
from typing import Callable, Tuple, Optional
from cv2.typing import MatLike, Point

# Custom
from ImageAnalyser import ImageAnalyser
from GameMap import GameMap
from solve_grid import breadth_first_search
from FrameMerchant import FrameMerchant
from utils.verify_phone import check_device_connection, AndroidConnectionError
from utils.kill_button import kill_button


class Bot:
    """Bot to automate Snake."""

    def __init__(
        self,
        map_config: str,
        max_size: int = 480,
        bitrate: int = 800000,
        max_fps: int = 20,
        visual_debug: bool = True,
    ):
        """Construct a new Bot to play Snake.

        Args:
            map_config (str): Custom YAML name of map type.
            max_size (int, optional): Maximum number of pixels for the phone's x/y. Defaults to 480.
            bitrate (int, optional): Rate of data transfer. Defaults to 800000.
            max_fps (int, optional): Maximum number of frames per second. Defaults to 20.
            greyscale (bool, optional): Apply greyscaling to images. Defaults to True.
            visual_debug (bool, optional): Apply visual debugging overlay. Defaults to True.
        """

        # CONSTANTS
        self.CONFIDENCE_THRESHOLD = 0.8

        # Server config
        self.max_size = max_size
        self.bitrate = bitrate
        self.max_fps = max_fps

        # Initialise image calibration
        self.image_analyser = ImageAnalyser()

        # Run hotkey in background
        keyboard.add_hotkey("esc", kill_button)

        # Shared state for keyboard input (to be accessed by global hotkeys)
        self.last_key_pressed = None

        # Register global hotkeys
        keyboard.on_press_key("w", lambda _: setattr(self, "last_key_pressed", "w"))
        keyboard.on_press_key("a", lambda _: setattr(self, "last_key_pressed", "a"))
        keyboard.on_press_key("s", lambda _: setattr(self, "last_key_pressed", "s"))
        keyboard.on_press_key("d", lambda _: setattr(self, "last_key_pressed", "d"))
        keyboard.on_press_key("e", lambda _: setattr(self, "last_key_pressed", "e"))

        # Calculate width of each cell
        self.frame_merchant = FrameMerchant(
            config_name=map_config,
            debug_overlay=visual_debug,
        )

        # Initialise game components
        self.game_map = GameMap(
            grid_nc_nr=self.frame_merchant.get_grid_dims(),
            grid_hpx_wpx=self.frame_merchant.get_cell_dims(),
        )
        self.client = Client(max_size=max_size, bitrate=bitrate, max_fps=max_fps)

        # Define movement directions for bot
        # dn=north(up), ds=south(down), de=east(right), dw=west(left)
        self.dirs = {"dn": (0, -1), "ds": (0, 1), "de": (1, 0), "dw": (-1, 0)}  # x,y
        self.dirs_alt = {
            (0, -1): "dn",
            (0, 1): "ds",
            (1, 0): "de",
            (-1, 0): "dw",
        }  # x,y
        self.opposite_dir = {"dn": "ds", "ds": "dn", "dw": "de", "de": "dw"}
        self.cur_dir = "de"

        # Movement directions for client
        self.KEYS = {
            "dn": const.KEYCODE_DPAD_UP,
            "ds": const.KEYCODE_DPAD_DOWN,
            "dw": const.KEYCODE_DPAD_LEFT,
            "de": const.KEYCODE_DPAD_RIGHT,
        }

        # Fill debug & pathfinding with inital dummy values
        self.snake_snoot_coords = (None, None)
        self.cur_path = []
        self.grid = []
        self.snake_body = []

    def play_snake(self) -> None:
        """Entry point to play a snake repeatedly."""
        # Setup and start
        if not self.client.alive:
            on_frame = self._get_on_frame()
            self.client.on_frame(on_frame)
            self.client.start(threaded=True)
        else:
            raise RuntimeError("Client was not previously closed.")

        try:
            while True:
                time.sleep(1)
        except:
            self.client.stop()
            cv.destroyAllWindows()
            kill_button()
        finally:
            self.client.stop()

    def _detect_and_process_object(
        self,
        hsv_frame: np.ndarray,
        frame: np.ndarray,
        lower_bound: list[int],
        upper_bound: list[int],
        colour_key: str,
        is_snake_head: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Detect an object using color filtering and return its grid coordinates.

        Args:
            hsv_frame (np.ndarray): HSV color space frame for filtering
            frame (np.ndarray): Original frame for drawing debug overlay
            lower_bound (list[int]): Lower HSV bounds for color filtering
            upper_bound (list[int]): Upper HSV bounds for color filtering
            colour_key (str): Color key for debug visualization
            is_snake_head (bool): Whether this is detecting the snake head

        Returns:
            Optional[Tuple[int, int]]: Grid coordinates (x, y) or None if not found
        """
        obj_loc = self.image_analyser.colour_filtering(
            frame=hsv_frame,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_snake_head=is_snake_head,
        )

        if obj_loc is None:
            return None

        # Extract width and height
        obj_w = obj_loc[2]
        obj_h = obj_loc[3]

        # Convert center to top-left corner
        top_left_x = obj_loc[0] - (obj_w // 2)
        top_left_y = obj_loc[1] - (obj_h // 2)
        top_left = (top_left_x, top_left_y)

        # Draw debug overlay
        self.frame_merchant._write_box_on_frame(
            frame,
            top_left=top_left,
            width=obj_w,
            height=obj_h,
            colour_key=colour_key,
        )

        # Convert to grid coordinates
        grid_coords = self.frame_merchant._pixel_to_coords(
            x=top_left_x, y=top_left_y, width=obj_w, height=obj_h
        )

        return grid_coords

    def _get_on_frame(self) -> Callable[[Client, np.ndarray], None]:

        def on_frame(client: Client, frame: np.ndarray) -> None:

            # Exit if not configured correctly
            if frame is None:
                return

            # Crop frame
            lpx, rpx, tpx, bpx = self.frame_merchant.get_crop_px()
            frame = frame[
                tpx:bpx,
                lpx:rpx,
            ]
            # Convert frame to greyscale for matching (only for template matching)
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Detect snake head
            snake_pos = self._detect_and_process_object(
                hsv_frame=hsv_frame,
                frame=frame,
                lower_bound=[83, 0, 190],
                upper_bound=[128, 173, 255],
                colour_key="snakeHead",
                is_snake_head=True,
            )

            # Detect apple
            apl_pos = self._detect_and_process_object(
                hsv_frame=hsv_frame,
                frame=frame,
                lower_bound=[4, 155, 191],
                upper_bound=[10, 241, 252],
                colour_key="apple",
                is_snake_head=False,
            )

            if apl_pos is None or snake_pos is None:
                cv.imshow("Bot Vision", frame)
                return

            if snake_pos and apl_pos:
                # Calculate the exact tip of the snake's nose based on movement direction
                snake_snooter_tip = self._get_snake_snooter_tip(snake_pos, self.cur_dir)

                # Don't pathfind if still in the same square
                if snake_snooter_tip != self.snake_snoot_coords:
                    self._on_new_cell(
                        frame=frame,
                        snake_snooter_tip=snake_snooter_tip,
                        apl_pos=apl_pos,
                    )

            # Render Everything
            self._render_soul(frame=frame)

            # Show the video feed with the overlay
            cv.imshow("Bot Vision", frame)
            cv.waitKey(1)

            # Check for keyboard input from global state
            if self.last_key_pressed:
                key = self.last_key_pressed
                self.last_key_pressed = None

                # Screenshot
                if key == "s":
                    cv.imwrite("debug_screenshot.png", frame)

                # Handle movement and restart commands
                self._interact_with_client(client=client, key=key)

        return on_frame

    def _on_new_cell(
        self,
        frame: MatLike,
        snake_snooter_tip: Tuple[int, int],
        apl_pos: Tuple[int, int],
    ) -> None:
        # Pathfinding
        self.grid, self.snake_body = self.game_map.build_grid(
            frame=frame,
        )

        self.snake_snoot_coords = snake_snooter_tip

        path = breadth_first_search(
            grid=self.grid,
            start_x=self.snake_snoot_coords[0] + 1,  # Add 1 for padding
            start_y=self.snake_snoot_coords[1] + 1,  # Add 1 for padding
            goal_x=apl_pos[0] + 1,  # Add 1 for padding
            goal_y=apl_pos[1] + 1,  # Add 1 for padding
            current_direction=self.dirs[self.cur_dir],
        )
        if path:
            unpadded_path = [(x - 1, y - 1) for x, y in path]
            self.cur_path = unpadded_path

            # Move the snake
            print(self.cur_path[0], self.snake_snoot_coords)
            self._move_snake()

    def _move_snake(self) -> None:
        if self.snake_snoot_coords[0] is None:
            return
        dx = self.cur_path[0][0] - self.snake_snoot_coords[0]
        dy = self.cur_path[0][1] - self.snake_snoot_coords[1]

        # Which direction to move in
        new_dir = self.dirs_alt[(dx, dy)]
        opposite_from_cur = self.opposite_dir.get(self.cur_dir)

        if new_dir == opposite_from_cur:
            return

        code = self.KEYS.get(new_dir)

        if code and new_dir:
            self.cur_dir = new_dir
            self.client.control.keycode(code, const.ACTION_DOWN)
            self.client.control.keycode(code, const.ACTION_UP)

    def _get_snake_snooter_tip(
        self, snake_head_pos: Tuple[int, int], direction: str
    ) -> Tuple[int, int]:
        """Calculate the exact tip of the snake's nose based on movement direction.

        Args:
            snake_head_pos (Tuple[int, int]): Center grid coordinates of snake head (x, y)
            direction (str): Current movement direction ("dn", "ds", "de", "dw")

        Returns:
            Tuple[int, int]: Grid coordinates of the snake's nose tip (x, y)
        """
        x, y = snake_head_pos

        # Adjust position based on which direction the snake is moving
        # This gives us the leading edge of the snake head
        match direction:
            case "dn":  # Moving north (up) - tip is at top
                return (x, y - 1) if y > 0 else (x, y)
            case "ds":  # Moving south (down) - tip is at bottom
                return (x, y + 1)
            case "de":  # Moving east (right) - tip is at right
                return (x + 1, y)
            case "dw":  # Moving west (left) - tip is at left
                return (x - 1, y) if x > 0 else (x, y)
            case _:  # Default: return center position
                return (x, y)

    def _interact_with_client(self, client: Client, key: str) -> None:
        """
        Instantly turns the snake.
        direction: "n", "s", "w", "e"
        """
        possible_keys = {"w", "a", "s", "d", "e"}
        # Map WASD to NESW
        code, new_dir = None, None

        # Protect against impossible movements
        opposite_from_cur = self.opposite_dir.get(self.cur_dir)
        if key not in possible_keys or key == opposite_from_cur:
            return

        match key:
            case "w":
                code = self.KEYS.get("dn")
                new_dir = "dn"
            case "a":
                code = self.KEYS.get("dw")
                new_dir = "dw"
            case "s":
                code = self.KEYS.get("ds")
                new_dir = "ds"
            case "d":
                code = self.KEYS.get("de")
                new_dir = "de"
            case "e":
                self.cur_dir = "de"
                self._restart(client)

        if new_dir == opposite_from_cur:
            return
        if code and new_dir:
            self.cur_dir = new_dir
            client.control.keycode(code, const.ACTION_DOWN)
            client.control.keycode(code, const.ACTION_UP)

    def _restart(self, client: Client) -> None:
        h, w = self.frame_merchant.get_original_device_px()
        x_coord = int(w * 0.463)
        y_coord = int(h * 0.701)
        client.control.touch(x_coord, y_coord)

    def _render_soul(self, frame: MatLike) -> None:
        self.frame_merchant.render_multi_coordinates(
            frame=frame,
            coordinates=self.snake_body,
            colour_key="snake",
            is_grid_coords=False,
        )

        if self.cur_path:
            self.frame_merchant.render_multi_coordinates(
                frame=frame,
                coordinates=self.cur_path,
                colour_key="path",
                is_grid_coords=True,
            )

        # Debug: Show current snoot position
        if self.snake_snoot_coords[0] is not None:
            self.frame_merchant.render_multi_coordinates(
                frame=frame,
                coordinates=[self.snake_snoot_coords],
                colour_key="snoot",
                is_grid_coords=True,
            )


if __name__ == "__main__":
    try:
        # Check device is connected before playing
        check_device_connection()

        bot = Bot(
            map_config="small",
            max_size=480,
            bitrate=800000,
            max_fps=20,
            visual_debug=True,
        )

        # Start playing
        bot.play_snake()
    except AndroidConnectionError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        kill_button()
