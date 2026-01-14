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

        # Initalise image calibration
        self.image_analyser = ImageAnalyser()

        # Run hotkey in background
        keyboard.add_hotkey("esc", kill_button)

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
        self.dirs = {"dn": (-1, 0), "ds": (1, 0), "de": (0, 1), "dw": (-1, 0)}
        self.opposite_dir = {"dn": "ds", "ds": "dn", "dw": "de", "de": "dw"}
        self.cur_dir = "de"

        # Movement directions for client
        self.KEYS = {
            "dn": const.KEYCODE_DPAD_UP,
            "ds": const.KEYCODE_DPAD_DOWN,
            "dw": const.KEYCODE_DPAD_LEFT,
            "de": const.KEYCODE_DPAD_RIGHT,
        }

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

            # Colour filtering for snake head
            head_loc = self.image_analyser.colour_filtering(
                frame=hsv_frame,
                lower_bound=[83, 0, 190],
                upper_bound=[128, 173, 255],
                is_snake_head=True,
            )

            if head_loc is not None:
                hw = head_loc[2]
                hh = head_loc[3]

                # Convert center to top-left corner
                top_left_x = head_loc[0] - (hw // 2)
                top_left_y = head_loc[1] - (hh // 2)
                top_left = (top_left_x, top_left_y)

                self.frame_merchant._write_box_on_frame(
                    frame,
                    top_left=top_left,
                    width=hw,
                    height=hh,
                    colour_key="snakeHead",
                )

                # For grid mapping, pass the top-left corner
                snake_pos = self.frame_merchant._pixel_to_coords(
                    x=top_left_x, y=top_left_y, width=hw, height=hh
                )
            else:
                snake_pos = None

            apl_loc = self.image_analyser.colour_filtering(
                frame=hsv_frame,
                lower_bound=[4, 155, 191],
                upper_bound=[10, 241, 252],
            )

            if apl_loc is not None:
                aw = apl_loc[2]
                ah = apl_loc[3]

                # Convert center to top-left corner
                top_left_x = apl_loc[0] - (aw // 2)
                top_left_y = apl_loc[1] - (ah // 2)
                top_left = (top_left_x, top_left_y)

                self.frame_merchant._write_box_on_frame(
                    frame,
                    top_left=top_left,
                    width=aw,
                    height=ah,
                    colour_key="apple",
                )

                # Use top-left for grid mapping (same as snake head)
                apl_pos = self.frame_merchant._pixel_to_coords(
                    top_left_x, top_left_y, width=aw, height=ah
                )
            else:
                apl_pos = None

            # Pathfinding
            grid, snake_body = self.game_map.build_grid(frame)

            # Render snake's body for debugger
            self.frame_merchant.render_multi_coordinates(
                frame=frame,
                coordinates=snake_body,
                colour_key="snake",
                is_grid_coords=False,
            )

            if snake_pos and apl_pos:
                path = breadth_first_search(
                    grid=grid,
                    start_x=snake_pos[0] + 1,  # Add 1 for padding
                    start_y=snake_pos[1] + 1,  # Add 1 for padding
                    goal_x=apl_pos[0] + 1,  # Add 1 for padding
                    goal_y=apl_pos[1] + 1,  # Add 1 for padding
                )
                if path:
                    # Convert back to unpadded coordinates for rendering
                    unpadded_path = [(x - 1, y - 1) for x, y in path]

                    self.frame_merchant.render_multi_coordinates(
                        frame=frame,
                        coordinates=unpadded_path,
                        colour_key="path",
                        is_grid_coords=True,
                    )

            # Show the video feed with the overlay
            cv.imshow("Bot Vision", frame)

            # cv.waitKey(1)
            key = cv.waitKey(1) & 0xFF
            self._interact_with_client(client=client, key=chr(key))
            if key == ord("s"):
                cv.imwrite("debug_screenshot.png", frame)

        return on_frame

    def _interact_with_client(self, client: Client, key: str) -> None:
        """
        Instantly turns the snake.
        direction: "n", "s", "w", "e"
        """
        # Ignore default key
        if key == "ÿ":
            return

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
                self._restart(client)

        if new_dir == opposite_from_cur:
            return
        print(new_dir, code)
        if code and new_dir:
            self.cur_dir = new_dir
            client.control.keycode(code, const.ACTION_DOWN)
            client.control.keycode(code, const.ACTION_UP)

    def _restart(self, client: Client) -> None:
        h, w = self.frame_merchant.get_original_device_px()
        x_coord = int(w * 0.463)
        y_coord = int(h * 0.701)
        client.control.touch(x_coord, y_coord)


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
