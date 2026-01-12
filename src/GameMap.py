import numpy as np
import cv2 as cv


class GameMap:
    def __init__(self, grid_w: int = 6, grid_h=14):
        self.GRID_W = grid_w
        self.GRID_H = grid_h

    def build_grid(self, frame: np.ndarray):
        img_h = img_h, img_w = frame.shape[:2]

        cell_w = img_w // self.GRID_W
        cell_h = img_h // self.GRID_H

        logic_grid = np.zeros((self.GRID_H, self.GRID_W), dtype=int)

        for r in range(self.GRID_H):
            for c in range(self.GRID_W):

                # Calculate center of this cell
                center_x = int(c * cell_w + cell_w / 2)
                center_y = int(r * cell_h + cell_h / 2)

                # Safety check: stay inside image
                if center_x >= img_w or center_y >= img_h:
                    continue

                # Get Color at this spot (BGR)
                pixel = frame[center_y, center_x]
                blue, green, red = pixel

                # Detect Snake Body
                is_snake_body = blue > 130

                if is_snake_body:
                    logic_grid[r, c] = 1  # Mark as Obstacle

                    # VISUAL DEBUG
                    cv.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
        # Pad the grid
        logic_grid = np.pad(logic_grid, pad_width=1, mode="constant", constant_values=1)
        return logic_grid
