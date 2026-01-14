import numpy as np
from cv2.typing import Point, MatLike
from typing import Tuple, DefaultDict, Optional
import cv2 as cv
from collections import defaultdict
import yaml


class FrameMerchant:
    def __init__(
        self,
        config_name: str = "small",
        debug_overlay=True,
    ) -> None:

        self.FONT = cv.FONT_HERSHEY_SIMPLEX

        # Parse input file to define dimensions and sizes
        self.setup_config = self._parse_yaml(file_path="crop_config.yaml")
        if self.setup_config and config_name in self.setup_config.keys():
            # Original phone resolution
            self.org_phone_width = self.setup_config["original_resolution"]["width"]
            self.org_phone_height = self.setup_config["original_resolution"]["height"]

            self.cfg = self.setup_config[config_name]

            # Absolute crop values in px
            abs_vals = self.cfg.get("absolute_values", {})
            self.lpx = abs_vals.get("lpx", 0)
            self.rpx = abs_vals.get("rpx", 0)
            self.tpx = abs_vals.get("tpx", 0)
            self.bpx = abs_vals.get("bpx", 0)

            # Size of cropped frame in px
            crop_size = self.cfg.get("crop_size", {})
            self.cap_w = crop_size.get("width", 0)
            self.cap_h = crop_size.get("height", 0)

            # Number of columns and cells
            self.cell_cols = self.cfg.get("cell_cols", 0)
            self.cell_rows = self.cfg.get("cell_rows", 0)

            # Calculate cell dimensions
            self.cell_height_px = self.cap_h // self.cell_rows
            self.cell_width_px = self.cap_w // self.cell_cols

        # Option to not use debug overlays
        self.debug_overlay = debug_overlay

        # Colours to render
        self.colours: DefaultDict[str, Tuple[int, int, int]] = defaultdict(
            lambda: (0, 0, 0)
        )  # Default to black
        self.colours["path"] = (255, 0, 255)
        self.colours["snake"] = (0, 0, 255)
        self.colours["apple"] = (0, 255, 0)
        self.colours["snakehead"] = (255, 0, 0)

    def _parse_yaml(self, file_path: str) -> Optional[dict]:
        """Parse config yaml to setup dimensions.
        Args:
            file_path (str): Relative path of yaml file.

        Returns:
            Optional[dict]: Config dictionary.
        """
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            print(f"Error: Config file '{file_path}' not found.")
            return None
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            return None

    def _write_box_on_frame(
        self,
        frame: MatLike,
        top_left: Point,
        width: int,
        height: int,
        colour_key: str,
        text: str = "",
    ) -> None:
        """Render a box + text layer on top of the frame for the userr

        Args:
            frame (MatLike): Original frame.
            top_left (Point): Coordinates of the top left of the box.
            width (int): Width of the box (px).
            height (int): Height of the box (px).
            colour_key (str): Name cooresponding to RGB value of box + text.
            text (str): Any text to write. Defaults to empty string.
        """
        if not self.debug_overlay:
            return
        # Find opposite corrner
        bottom_right = (top_left[0] + width, top_left[1] + height)

        # Draw a rectangle on the frame
        colour = self.colours[colour_key.lower()]
        cv.rectangle(
            img=frame, pt1=top_left, pt2=bottom_right, color=colour, thickness=2
        )

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

    def _pixel_to_coords(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Tuple[int, int]:
        """Convert screen pixel to grid coordinates.

        Args:
            x (int): X pixel value.
            y (int): Y pixel value.
            width (int): Width of object
            height (int): Height of object.

        Returns:
            Tuple[int, int]: x, y coordinates.
        """
        cell_w = int(round(self.cell_width_px))
        cell_h = int(round(self.cell_height_px))

        # Adjust x, y to be the centre of the object
        centre_x = x + (width // 2)
        centre_y = y + (height // 2)

        # Convert to Grid Index
        grid_x = centre_x // cell_w
        grid_y = centre_y // cell_h

        # Safety Clamp - use GRID dimensions (number of cells), not cell pixel sizes!
        grid_x = max(0, min(grid_x, self.cell_cols - 1))
        grid_y = max(0, min(grid_y, self.cell_rows - 1))

        # Ensure always an integer
        return (int(grid_x), int(grid_y))

    def _coords_to_pixels(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert grid coordinates into pixels.

        Args:
            grid_x (int): X coordinate.
            grid_y (int): Y coordinate.

        Returns:
            Tuple[int, int]: (x,y) coordinate (px).
        """
        cell_w = int(round(self.cell_width_px))
        cell_h = int(round(self.cell_height_px))
        # Convert to Grid Index
        x = int(round((grid_x * cell_w) + (cell_w / 2)))
        y = int(round((grid_y * cell_h) + (cell_h / 2)))

        return (x, y)

    def render_multi_coordinates(
        self,
        frame: MatLike,
        coordinates: list[Tuple[int, int]],
        colour_key: str,
        is_grid_coords: bool,
    ) -> None:
        """Render the bot's current path using magenta circles

        Args:
            frame (MatLike): Current frame.
            coordinates (list[Tuple[int, int]]): Current path.
            colour_key (str): Name of colour key to display using.
            is_grid_coords (bool): Are the coordinates game grid coordinates.
        """
        if not self.debug_overlay:
            return

        colour = self.colours[colour_key.lower()]

        for x, y in coordinates:

            if is_grid_coords:
                x, y = self._coords_to_pixels(grid_x=x, grid_y=y)

            cv.circle(frame, (y, x), 3, colour, -1)

    def get_grid_dims(self) -> Tuple[int, int]:
        """Return number of columns and rows."""
        return self.cell_cols, self.cell_rows

    def get_cell_dims(self) -> Tuple[float, float]:
        """Return cell height and width."""
        return self.cell_height_px, self.cell_width_px

    def get_cropped_frame_px(self) -> Tuple[int, int]:
        """Return crop height and width."""
        return self.cap_h, self.cap_w

    def get_crop_px(self) -> Tuple[int, int, int, int]:
        """Get left, right, top and bottom cropping coordinates."""
        return self.lpx, self.rpx, self.tpx, self.bpx

    def get_original_device_px(self) -> Tuple[int, int]:
        """Return height and width of original device screen."""
        return self.org_phone_height, self.org_phone_width
