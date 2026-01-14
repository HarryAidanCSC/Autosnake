import numpy as np
from typing import Tuple


class GameMap:
    def __init__(self, grid_nc_nr: Tuple[int, int], grid_hpx_wpx: Tuple[float, float]):
        self.NCOLS, self.NROWS = grid_nc_nr
        self.CELL_HPX, self.CELL_WPX = grid_hpx_wpx

        self.centre_coords = []
        for r in range(self.NROWS):
            row = []
            for c in range(self.NCOLS):

                # Calculate center of this cell
                centre_x = int(c * self.CELL_WPX + self.CELL_WPX / 2)
                centre_y = int(r * self.CELL_HPX + self.CELL_HPX / 2)

                row.append((centre_x, centre_y))
            self.centre_coords.append(row)

    def build_grid(self, frame: np.ndarray) -> Tuple[np.ndarray, list[Tuple[int, int]]]:
        # Get frame dimensions
        img_h, img_w = frame.shape[:2]

        # Create logic grid
        logic_grid = np.zeros((self.NROWS, self.NCOLS), dtype=int)

        # Find snake body
        snake_body = []

        for r in range(self.NROWS):
            for c in range(self.NCOLS):

                centre_x, centre_y = self.centre_coords[r][c]

                # Safety check: stay inside image
                if centre_x >= img_w or centre_y >= img_h:
                    continue

                # Get Color at this spot (BGR)
                pixel = frame[centre_y, centre_x]
                blue, green, red = pixel

                # Detect Snake Body
                is_snake_body = blue > 130

                if is_snake_body:
                    logic_grid[r, c] = 1  # Mark as Obstacle

                    snake_body.append((centre_y, centre_x))
        # Pad the grid
        logic_grid = np.pad(logic_grid, pad_width=1, mode="constant", constant_values=1)
        return logic_grid, snake_body

    def snooter_elimination_unit(self):
        pass
