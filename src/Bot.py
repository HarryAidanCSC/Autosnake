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
from TemplateGetter import TemplateGetter
from GameMap import GameMap
from solve_grid import breadth_first_search
from FrameRenderer import FrameRender
from utils.verify_phone import check_device_connection, AndroidConnectionError
from utils.kill_button import kill_button


class Bot:
    """Bot to automate Snake."""

    def __init__(
        self,
        snake_file_path: str,
        apple_file_path: str,
        cog_file_path: str,
        max_size: int = 480,
        bitrate: int = 800000,
        max_fps: int = 20,
        greyscale: bool = False,
        visual_debug: bool = True,
    ):
        """Construct a new Bot to play Snake.

        Args:
            snake_file_path (str): File path of the snake head template.
            apple_file_path (str): File path of the apple template.
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

        # Movement directions
        self.KEYS = {
            "UP": const.KEYCODE_DPAD_UP,
            "DOWN": const.KEYCODE_DPAD_DOWN,
            "LEFT": const.KEYCODE_DPAD_LEFT,
            "RIGHT": const.KEYCODE_DPAD_RIGHT,
        }

        # Initalise image calibration
        self.template_getter = TemplateGetter(greyscale=greyscale)
        self.greyscale = greyscale

        # Run hotkey in background
        keyboard.add_hotkey("esc", kill_button)

        # Load templates
        self.snake_head_templates = self.template_getter.get_image_and_rotations(
            file_path=snake_file_path
        )
        self.apple_template = self.template_getter.get_image_and_scaled(
            file_path=apple_file_path, scales=[0.9, 1.0, 1.1]
        )

        # Load and start client
        self.client = Client(
            max_size=self.max_size, bitrate=self.bitrate, max_fps=self.max_fps
        )
        self.game_map = GameMap()

        # Original phone dimensions - adjust as needed
        self.PHONE_RESOLUTION = (1080, 2408)

        # Define trim values for cropping the frame
        self.left_crop, self.right_crop, self.top_crop, self.bottom_crop = (
            self._calibrate_capture_region(cog_file_path=cog_file_path)
        )

        # New capture width for the cropped region
        self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT = (
            self.right_crop - self.left_crop,
            self.bottom_crop - self.top_crop,
        )
        # Calculate width of each cell
        self.frame_renderer = FrameRender(
            capture_width=self.CAPTURE_WIDTH,
            capture_height=self.CAPTURE_HEIGHT,
            n_cells_w=self.game_map.GRID_W,
            n_cells_h=self.game_map.GRID_H,
            debug_overlay=visual_debug,
        )

    def _calibrate_capture_region(
        self, cog_file_path: str
    ) -> Tuple[int, int, int, int]:
        """Calibrate existing capture region to JUST the relevant part of the screen

        Arguements:
            cog_file_path (str): File path of cog icon for template matching.
        Raises:
            RuntimeError: Screen grab could not be taken.
            RuntimeError: Could not find a valid capture region.

        Returns:
            Tuple[int, int, int, int]: left, right, top and bottom crop coordinates.
        """
        self.latest_frame = None  # Ensure we don't have old data
        self.last_action_time = 0

        # Read in cog image template
        self.cog_image = self.template_getter._read_image(file_path=cog_file_path)

        # Temporary function that saves the frame
        def get_one_frame(client: Client, frame: MatLike) -> None:
            """Function to take one single frame during setup"

            Args:
                client (Client): Connection to client.
                frame (MatLike): Current viewing frame.
            """
            if frame is None:
                return

            # Exit if last action was recent
            if time.time() - self.last_action_time < 1.5:
                return

            # Check if the frame contains the setting cog icon
            value, _ = self.template_getter.match_with_template(
                img=self.cog_image, comparison_frame=frame
            )

            # If the cog is in frame, press play
            print(value)
            if value > self.CONFIDENCE_THRESHOLD:
                self._restart(client=client)
                self.last_action_time = time.time()
                return

            if frame is not None:
                self.latest_frame = frame
                self.client.stop()

        self.client.on_frame(get_one_frame)
        self.client.start(threaded=True)

        #  Wait until we have the first frame
        start_time = time.time()
        while self.latest_frame is None:
            # Emergency exit if it takes too long
            if time.time() - start_time > 5:
                raise RuntimeError("Could not capture an inital frame during setup.")
            time.sleep(0.01)

        # Find the capture region
        hsv_frame = cv.cvtColor(self.latest_frame, cv.COLOR_BGR2HSV)

        # Kill the client
        self.client.stop()

        # Colour filtering for playable region
        capture_region_loc = self.template_getter.get_corners(
            frame=hsv_frame,
            lower_bound=[32, 148, 173],
            upper_bound=[46, 175, 223],
        )

        if capture_region_loc is None:
            raise RuntimeError("Could not find the initial start frame.")

        top_left, bottom_right = capture_region_loc

        # Calibrate new capture region
        return top_left[0], bottom_right[0], top_left[1], bottom_right[1]

    def play_snake(self) -> None:
        """Entry point to play a snake repeatedly."""
        # Setup and start
        if not self.client.alive:
            self.client = Client(
                max_size=self.max_size, bitrate=self.bitrate, max_fps=self.max_fps
            )
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
            frame = frame[
                self.top_crop : self.bottom_crop, self.left_crop : self.right_crop
            ]
            # Convert frame to greyscale for matching (only for template matching)
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Colour filtering for snake head
            head_loc = self.template_getter.colour_filtering(
                frame=hsv_frame,
                lower_bound=[83, 0, 190],
                upper_bound=[128, 173, 255],
                is_snake_head=True,
            )

            if head_loc is not None:
                template_w = self.snake_head_templates[0][1]
                template_h = self.snake_head_templates[0][2]

                # Convert center to top-left corner
                top_left_x = head_loc[0] - (template_w // 2)
                top_left_y = head_loc[1] - (template_h // 2)
                top_left = (top_left_x, top_left_y)

                self.frame_renderer._write_box_on_frame(
                    frame,
                    top_left=top_left,
                    width=template_w,
                    height=template_h,
                    colour_key="snakeHead",
                )

                # For grid mapping, pass the top-left corner
                snake_pos = self.frame_renderer._pixel_to_coords(
                    top_left_x,
                    top_left_y,
                    self.snake_head_templates[0],  # type:ignore
                )
            else:
                snake_pos = None

            apl_loc = self.template_getter.colour_filtering(
                frame=hsv_frame,
                lower_bound=[4, 155, 191],
                upper_bound=[10, 241, 252],
            )

            if apl_loc is not None:
                template_w = self.apple_template[0][1]
                template_h = self.apple_template[0][2]

                # Convert center to top-left corner
                top_left_x = apl_loc[0] - (template_w // 2)
                top_left_y = apl_loc[1] - (template_h // 2)
                top_left = (top_left_x, top_left_y)

                self.frame_renderer._write_box_on_frame(
                    frame,
                    top_left=top_left,
                    width=template_w,
                    height=template_h,
                    colour_key="apple",
                )

                # Use top-left for grid mapping (same as snake head)
                apl_pos = self.frame_renderer._pixel_to_coords(
                    top_left_x,
                    top_left_y,
                    self.apple_template[0],  # type:ignore
                )
            else:
                apl_pos = None

            # Pathfinding
            grid, snake_body = self.game_map.build_grid(frame)

            # Render snake's body for debugger
            self.frame_renderer.render_multi_coordinates(
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
                    unpadded_path = [(y - 1, x - 1) for x, y in path]

                    self.frame_renderer.render_multi_coordinates(
                        frame=frame,
                        coordinates=unpadded_path,
                        colour_key="path",
                        is_grid_coords=True,
                    )

            # Show the video feed with the overlay
            cv.imshow("Bot Vision", frame)

            # cv.waitKey(1)
            key = cv.waitKey(1) & 0xFF
            if key == ord("w"):
                self._move_snake(client, "UP")
            elif key == ord("a"):
                self._move_snake(client, "LEFT")
            elif key == ord("s"):
                self._move_snake(client, "DOWN")
            elif key == ord("d"):
                self._move_snake(client, "RIGHT")
            elif key == ord("e"):
                self._restart(client)
            if key == ord("s"):
                cv.imwrite("debug_screenshot.png", frame)

        return on_frame

    def _match_best_to_template(
        self,
        primary_collection: list[Tuple[MatLike, int, int]],
        comparison_frame: np.ndarray,
        colour_key: str,
        frame: MatLike,
    ) -> Optional[Point]:
        """Match an image with another image. If it matches then return the threshold and location.

        Args:
            primary_collection ([(MatLike, int, int)]: Image(s) to locate the comparison upon.
            comparison_frame (MatLike): Image to compare.
            colour_key (str): Name of colour key to display using.
            frame: (MatLike): Image to display.
            "
        """
        best_val, best_loc, best_w, best_h = 0.0, None, 0, 0

        for collection in primary_collection:
            img, w, h = collection

            # Closest match
            max_val, max_loc = self.template_getter.match_with_template(
                img=img, comparison_frame=comparison_frame
            )

            # Keep best match
            if max_val > best_val:
                best_val, best_loc, best_w, best_h = (
                    max_val,
                    max_loc,
                    w,
                    h,
                )

            # Break condition

            if best_val >= self.CONFIDENCE_THRESHOLD and best_loc is not None:
                self.frame_renderer._write_box_on_frame(
                    frame,
                    top_left=best_loc,
                    width=best_w,
                    height=best_h,
                    colour_key=colour_key,
                    text=f"Match: {best_val:.2f}",
                )
                return max_loc

            return None

    def _move_snake(self, client: Client, direction: str):
        """
        Instantly turns the snake.
        direction: "UP", "DOWN", "LEFT", "RIGHT"
        """
        code = self.KEYS.get(direction)
        if code:
            client.control.keycode(code, const.ACTION_DOWN)
            client.control.keycode(code, const.ACTION_UP)
        print("Action Done")

    def _restart(self, client) -> None:
        x_coord = int(self.PHONE_RESOLUTION[0] * 0.463)
        y_coord = int(self.PHONE_RESOLUTION[1] * 0.701)
        client.control.touch(x_coord, y_coord)


if __name__ == "__main__":
    try:
        # Check device is connected before playing
        check_device_connection()

        bot = Bot(
            snake_file_path="assets/snake_snooter.png",
            apple_file_path="assets/apple.png",
            cog_file_path="assets/gear.png",
            max_size=480,
            bitrate=800000,
            max_fps=20,
            visual_debug=True,
        )
        bot.play_snake()
    except AndroidConnectionError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        kill_button()
