import numpy as np
from typing import Tuple
from cv2.typing import MatLike
from SnakeState import SnakeState


class GameMap:
    def __init__(
        self,
        grid_nc_nr: Tuple[int, int],
        grid_hpx_wpx: Tuple[float, float],
        state: SnakeState,
    ):
        self.NCOLS, self.NROWS = grid_nc_nr
        self.CELL_HPX, self.CELL_WPX = grid_hpx_wpx
        self.state = state

        self.centre_coords = []
        for r in range(self.NROWS):
            row = []
            for c in range(self.NCOLS):

                # Calculate centre of this cell
                centre_x = int(c * self.CELL_WPX + self.CELL_WPX / 2)
                centre_y = int(r * self.CELL_HPX + self.CELL_HPX / 2)

                row.append((centre_x, centre_y))
            self.centre_coords.append(row)

    def build_grid(
        self,
        frame: MatLike,
    ) -> None:
        # Get frame dimensions
        img_h, img_w = frame.shape[:2]

        # Find snake body
        snake_body = []

        for r in range(self.NROWS):
            for c in range(self.NCOLS):
                value = 0

                centre_x, centre_y = self.centre_coords[r][c]

                # Safety check: stay inside image
                if centre_x >= img_w or centre_y >= img_h:
                    continue

                # Get Color at this spot (BGR)
                pixel = frame[centre_y, centre_x]
                try:
                    blue, _, _ = pixel
                except Exception as e:
                    print("YOU TWAT", pixel)
                    import sys

                    sys.exit(1)

                # Detect Snake Body
                is_snake_body = blue > 130

                if is_snake_body:
                    value = 1

                    snake_body.append((centre_x, centre_y))
                self.state.update_grid_coords(row=r, col=c, value=value)

        # Update state
        self.state.snake_body = snake_body
