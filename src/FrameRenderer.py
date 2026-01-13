import numpy as np
from cv2.typing import Point, MatLike
from typing import Tuple, DefaultDict
import cv2 as cv
from collections import defaultdict


class FrameRender:
    def __init__(
        self,
        capture_width: int,
        capture_height: int,
        n_cells_w: int,
        n_cells_h: int,
        debug_overlay=True,
    ) -> None:
        self.FONT = cv.FONT_HERSHEY_SIMPLEX
        self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT = capture_width, capture_height

        # Calculate n pixels per cell
        self.CELL_WIDTH = capture_width / n_cells_w
        self.CELL_HEIGHT = capture_height / n_cells_h

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

    def _pixel_to_coords(self, x: int, y: int, template: MatLike) -> Tuple[int, int]:
        """Convert screen pixel to grid coordinates.

        Args:
            x (int): X pixel value.
            y (int): Y pixel value.
            template (MatLike): Template image.

        Returns:
            Tuple[int, int]: x, y coordinates.
        """
        # Adjust x, y to be the centre of the object
        centre_x = x + (template[1] // 2)
        centre_y = y + (template[2] // 2)

        # Convert to Grid Index
        grid_x = centre_x // self.CELL_WIDTH
        grid_y = centre_y // self.CELL_HEIGHT

        # Safety Clamp
        grid_x = max(0, min(grid_x, self.CELL_WIDTH - 1))
        grid_y = max(0, min(grid_y, self.CELL_HEIGHT - 1))

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

        # Convert to Grid Index
        x = int((grid_x * self.CELL_WIDTH) + (self.CELL_WIDTH / 2))
        y = int((grid_y * self.CELL_HEIGHT) + (self.CELL_HEIGHT / 2))

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
