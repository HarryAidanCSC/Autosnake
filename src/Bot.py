import cv2 as cv
import numpy as np
from pyscrcpy import Client, const
from TemplateGetter import TemplateGetter
import time
import os
import keyboard
from typing import Callable, Tuple, Optional
from cv2.typing import Point

# import utils.kill_button


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

        self.CONFIDENCE_THRESHOLD = 0.65
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
            file_path=apple_file_path, scales=[0.8, 1.0, 1.2]
        )

        # Load and start client
        self.client = Client(max_size=max_size, bitrate=bitrate, max_fps=max_fps)

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
            frame = frame[71:469, 22:193]

            # Convert frame to greyscale for matching
            comparison_frame = (
                cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if self.greyscale else frame
            )

            # Perform Template Matching
            # Snake
            self._match_best_to_template(
                primary_collection=self.snake_head_templates,
                comparison_frame=comparison_frame,
                text_colour=(0, 0, 255),
                frame=frame,
            )

            # Apple
            self._match_best_to_template(
                primary_collection=self.apple_template,
                comparison_frame=comparison_frame,
                text_colour=(0, 255, 0),
                frame=frame,
            )

            # Show the video feed with the overlay
            cv.circle(frame, (120, 350), 10, (255, 255, 0), -1)
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

    def _match_best_to_template(
        self,
        primary_collection: list[Tuple[np.ndarray, int, int]],
        comparison_frame: np.ndarray,
        text_colour: Tuple[int, int, int],
        frame: np.ndarray,
    ) -> None:
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
                return

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
            snake_file_path="assets/snake_eyes.png",
            apple_file_path="assets/apple.png",
            max_size=480,
            bitrate=800000,
            max_fps=20,
            greyscale=True,
        )
        bot.play_snake()
    except Exception:
        bot.kill_button()  # type:ignore
