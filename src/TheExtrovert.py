from utils.fix_dpi_scaling import fix_dpi_scaling

fix_dpi_scaling()

import mss
import cv2 as cv
from cv2.typing import MatLike
from typing import Any
import numpy as np
import pyautogui
from typing import Tuple, Optional
import keyboard
import yaml


class TheExtrovert:
    """Class to interact with the UI"""

    def __init__(self, config_name: str = "small") -> None:
        """Constructs a new Extrovert.

        Args:
            config_name (str, optional): Name of config file to attatch to. Defaults to "small".
        """
        self.mon = mss.mss().monitors[0]
        self.MAX_DISPLAY_RESOLUTION = 2500

        # Setup mss connection
        self.sct = mss.mss()

        # Parse config file
        self.display_config = self._parse_yaml(config_name=config_name)

        # Update monitor
        # Assume monitor is on left of main display
        self.mon["left"] = (
            +self.display_config["absolute_values"]["lpx"]
            - self.display_config["frame_size"]["width"]
        )
        self.mon["top"] += self.display_config["absolute_values"]["tpx"]
        self.mon["width"] = self.display_config["crop_size"]["width"]
        self.mon["height"] = self.display_config["crop_size"]["height"]

        # Cache frequently used coords
        self.coords_cache = {"restart": (420, 480)}

    def _parse_yaml(self, config_name: str) -> dict[str, Any]:
        """Parse the YAML config to calibrate display.

        Args:
            config_name (str): Name of config file to attatch to.

        Raises:
            RuntimeError: If the config is not loaded correctly due to a non-existent config name being used.

        Returns:
            dict[str, Any]: A display cofiguration.
        """
        with open("crop_config.yaml", "r") as file:
            config = yaml.safe_load(file)
            difficulty_config = config.get(config_name)

        if not difficulty_config:
            raise RuntimeError(
                f"Cannot load config correctly. Config: {config_name} does not exist. Select from:\n{', '.join(list(config.keys()))}"
            )

        return difficulty_config

    def get_raw_frame(self) -> MatLike:
        """Pull a single raw frame.

        Returns:
            MatLike: Display frame.
        """
        frame = np.array(self.sct.grab(self.mon))
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        return frame

    def display_live_feed(self, frames: dict[str, MatLike]) -> bool:
        """
        Display multiple video feeds at once. Allow a screenshot for the first feed.

        Args:
            frames: Dictionary where Key=WindowName and Value=Image.

        Returns:
            bool: Keep running?

        """
        if not frames:
            return True

        for window_name, frame in frames.items():
            display_img = self._resize_frame(frame)
            cv.imshow(window_name, display_img)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            cv.destroyAllWindows()
            return False
        return True

    def display_frame(self, frames: dict[str, MatLike]) -> None:
        """Display a single peristent raw frame(s) from multiple sources.

        Args:
            frames (dict[str, MatLike]): Name of window and frame.
        """
        if not frames:
            return

        for window_name, frame in frames.items():
            frame = self._resize_frame(frame=frame)
            cv.imshow(winname=window_name, mat=frame)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def _resize_frame(self, frame: MatLike) -> MatLike:
        """Helper function to ensure frame fits on screen nicely.

        Args:
            frame (MatLike): Display frame.

        Returns:
            MatLike: Resized frame.
        """
        height, width = frame.shape[:2]

        if height > self.MAX_DISPLAY_RESOLUTION or width > self.MAX_DISPLAY_RESOLUTION:
            scale = self.MAX_DISPLAY_RESOLUTION / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv.resize(frame, (new_width, new_height))

        return frame

    def click_screen(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        apply_offset: bool = True,
        coords_cache_key: str = "NOKEY",
    ) -> None:
        """Click screen coords optionally with a screen offset adjustment.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            apply_offset (bool, optional): Whether to apply offset. Defaults to True.
        """
        xy = self.coords_cache.get(coords_cache_key)
        if xy:
            x, y = xy

        if x is None or y is None:
            raise ValueError(
                f"X coordinate of {x} and Y coordinate of {y} are not valid."
            )

        if apply_offset:
            x, y = self._coord_offset(x=x, y=y)

        pyautogui.click(x=x, y=y)

    def _coord_offset(self, x: int, y: int) -> Tuple[int, int]:
        """Helper function to apply coordinate offset.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.

        Returns:
            Tuple[int, int]: (x,y)
        """
        x += self.mon["left"]
        y += self.mon["top"]
        return x, y

    def press_keyboard(self, keys: str) -> None:
        """Press a key or number of keys on the keyboard."

        Args:
            keys (str): Keys to press
        """
        keyboard.press_and_release(keys)

    def get_display_config(self) -> dict[str, Any]:
        return self.display_config

    def save_screenshot(self, frame: MatLike) -> None:
        cv.imwrite("debug_screenshot.png", frame)
