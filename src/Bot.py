import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from pyscrcpy import Client, const
from TemplateGetter import TemplateGetter
import time
import os
import keyboard
from typing import Callable, Tuple, Optional
from cv2.typing import Point
from GameMap import GameMap
from solve_grid import breadth_first_search


class Bot:
    def __init__(
        self,
        snake_file_path: str,
        apple_file_path: str,
        max_size: int = 480,
        bitrate: int = 800000,
        max_fps: int = 20,
        greyscale: bool = True,
    ):

        self.CONFIDENCE_THRESHOLD = 0.45
        self.FONT = cv.FONT_HERSHEY_SIMPLEX
        # Movement directions
        self.KEYS = {
            "UP": const.KEYCODE_DPAD_UP,
            "DOWN": const.KEYCODE_DPAD_DOWN,
            "LEFT": const.KEYCODE_DPAD_LEFT,
            "RIGHT": const.KEYCODE_DPAD_RIGHT,
        }
        # Original phone dimensions
        self.PHONE_RESOLUTION = (1080, 2408)

        self.template_getter = TemplateGetter(greyscale=greyscale)
        self.greyscale = greyscale

        # Run hotkey in background
        keyboard.add_hotkey("esc", self.kill_button)

        # Load templates
        self.snake_head_templates = self.template_getter.get_image_and_rotations(
            file_path=snake_file_path
        )
        self.apple_template = self.template_getter.get_image_and_scaled(
            file_path=apple_file_path, scales=[0.9, 1.0, 1.1]
        )

        # Load and start client
        self.client = Client(max_size=max_size, bitrate=bitrate, max_fps=max_fps)

        # Prepare map
        self.game_map = GameMap()

        # Define trim values for cropping the frame
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 215, 480
        self.left_crop = int(self.FRAME_WIDTH * 0.1)
        self.right_crop = self.FRAME_WIDTH - int(self.FRAME_WIDTH * 0.1)
        self.top_crop = int(self.FRAME_HEIGHT * 0.148)
        self.bottom_crop = self.FRAME_HEIGHT - int(self.FRAME_HEIGHT * 0.023)
        self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT = (
            self.right_crop - self.left_crop,
            self.bottom_crop - self.top_crop,
        )
        self.CELL_W = self.CAPTURE_WIDTH / self.game_map.GRID_W
        self.CELL_H = self.CAPTURE_HEIGHT / self.game_map.GRID_H

    def kill_button(
        self,
    ) -> None:
        """Keyboard escape if mouse cannot be force quitted."""

        print("\nEscape Button Pressed. Force quitting...")
        os._exit(0)

    def play_snake(self) -> None:

        # Setup and start
        on_frame = self._get_on_frame()
        self.client.on_frame(on_frame)
        self.client.start(threaded=True)

        # Stop client on main thread
        try:
            while True:
                time.sleep(1)
        except:
            self.client.stop()
            cv.destroyAllWindows()
            self.kill_button()

    def _get_on_frame(self) -> Callable[[Client, np.ndarray], None]:

        def on_frame(client: Client, frame: np.ndarray) -> None:
            # Exit if not configured correctly
            if frame is None:
                return

            # Crop frame
            frame = frame[
                self.top_crop : self.bottom_crop, self.left_crop : self.right_crop
            ]

            # Convert frame to greyscale for matching
            comparison_frame = (
                cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if self.greyscale else frame
            )

            # Perform Template Matching
            # Snake
            head_loc = self._match_best_to_template(
                primary_collection=self.snake_head_templates,
                comparison_frame=comparison_frame,
                text_colour=(0, 0, 255),
                frame=frame,
            )
            if head_loc is not None:
                snake_pos = self._pixel_to_coords(
                    head_loc[0],
                    head_loc[1],
                    self.snake_head_templates[0],  # type:ignore
                )
            else:
                snake_pos = None

            # Apple
            apl_loc = self._match_best_to_template(
                primary_collection=self.apple_template,
                comparison_frame=comparison_frame,
                text_colour=(0, 255, 0),
                frame=frame,
            )

            if apl_loc is not None:
                apl_pos = self._pixel_to_coords(
                    apl_loc[0],
                    apl_loc[1],
                    self.apple_template[0],  # type:ignore
                )
            else:
                apl_pos = None

            # Pathfinding
            grid = self.game_map.build_grid(frame)
            if snake_pos and apl_pos:
                path = breadth_first_search(
                    grid=grid,
                    start_x=snake_pos[0],
                    start_y=snake_pos[1],
                    goal_x=apl_pos[0],
                    goal_y=apl_pos[1],
                )
                if path:
                    for grid_x, grid_y in path:
                        print(grid_y, grid_x)
                        x, y = self._coords_to_pixels(grid_x=grid_x, grid_y=grid_y)
                        cv.circle(frame, (y, x), 3, (80, 98, 255), -1)

            # Show the video feed with the overlay
            cv.imshow("Bot Vision", frame)

            # cv.waitKey(1)
            key = cv.waitKey(1)
            if key == ord("w"):
                self._move_snake(client, "UP")
            if key == ord("a"):
                self._move_snake(client, "LEFT")
            if key == ord("s"):
                self._move_snake(client, "DOWN")
            if key == ord("d"):
                self._move_snake(client, "RIGHT")
            elif key == ord("e"):
                self._restart(client)

        return on_frame

    def _pixel_to_coords(self, x: int, y: int, template: MatLike) -> Tuple[int, int]:

        # Adjust x, y to be the centre of the object
        obj_w = template[1]
        obj_h = template[2]

        centre_x = x + (obj_w // 2)
        centre_y = y + (obj_h // 2)

        # Convert to Grid Index
        grid_x = int(centre_x // self.CELL_W)
        grid_y = int(centre_y // self.CELL_H)

        # Safety Clamp
        grid_x = max(0, min(grid_x, self.game_map.GRID_W - 1))
        grid_y = max(0, min(grid_y, self.game_map.GRID_H - 1))

        return (grid_x, grid_y)

    def _coords_to_pixels(self, grid_x: int, grid_y: int) -> Tuple[int, int]:

        # Convert to Grid Index
        x = int((grid_x * self.CELL_W) + (self.CELL_W / 2))
        y = int((grid_y * self.CELL_H) + (self.CELL_H / 2))

        return (x, y)

    def _match_best_to_template(
        self,
        primary_collection: list[Tuple[np.ndarray, int, int]],
        comparison_frame: np.ndarray,
        text_colour: Tuple[int, int, int],
        frame: np.ndarray,
    ) -> Optional[Point]:
        """Match an image with another image. If it matches then return the threshold and location.

        Args:
            primary_collection ([(np.ndarray, int, int)]: Image(s) to locate the comparison upon.
            comparison_frame (np.ndarray): Image to compare.
            text_colour ((int, int, int)): Colour on frame to display.
            frame: (np.ndarray): Image to display.
        """
        best_val, best_loc, best_w, best_h = 0.0, None, 0, 0

        for collection in primary_collection:
            img, w, h = collection
            result = cv.matchTemplate(img, comparison_frame, cv.TM_CCOEFF_NORMED)

            # Closest match
            _, max_val, _, max_loc = cv.minMaxLoc(result)

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
                self._write_text_on_frame(
                    frame,
                    top_left=best_loc,
                    width=best_w,
                    height=best_h,
                    colour=text_colour,
                    text=f"Match: {best_val:.2f}",
                )
                return max_loc

            return None

    def _write_text_on_frame(
        self,
        frame: np.ndarray,
        top_left: Point,
        width: int,
        height: int,
        colour: Tuple[int, int, int],
        text: str,
    ) -> None:
        # max_loc is the top-left corner of the match
        bottom_right = (top_left[0] + width, top_left[1] + height)

        # 5. Draw a Rectangle on the ORIGINAL frame
        # (Image, Start, End, Color(B,G,R), Thickness)
        cv.rectangle(frame, top_left, bottom_right, colour, 2)

        # Put text showing the confidence score
        cv.putText(
            frame,
            text,
            (top_left[0], top_left[1] - 10),
            self.FONT,
            0.5,
            colour,
            1,
        )

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
        bot = Bot(
            snake_file_path="assets/snake_snooter.png",
            apple_file_path="assets/apple.png",
            max_size=480,
            bitrate=800000,
            max_fps=20,
            greyscale=False,
        )
        bot.play_snake()
    except Exception:
        bot.kill_button()  # type:ignore
