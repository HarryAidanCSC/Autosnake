import time
from typing import Tuple, Any
import numpy as np


class SnakeState:
    def __init__(self, ncols: int, nrows: int):
        self.NROWS = nrows
        self.NCOLS = ncols
        # Define movement directions for bot
        # dn=north(up), ds=south(down), de=east(right), dw=west(left)
        self.dirs = {"dn": (0, -1), "ds": (0, 1), "de": (1, 0), "dw": (-1, 0)}  # x,y
        self.dirs_alt = {
            (0, -1): ("dn", "up"),
            (0, 1): ("ds", "down"),
            (1, 0): ("de", "right"),
            (-1, 0): ("dw", "left"),
        }  # x,y
        self.opposite_dir = {"dn": "ds", "ds": "dn", "dw": "de", "de": "dw"}
        self.cur_dir = "de"

        # Fill debug & pathfinding with inital dummy values
        self.snake_snoot_coords: Tuple[int, int] = (-1, -1)
        self.apple_pos: Tuple[int, int] = (-1, -1)

        self.cur_path = []
        self.snake_body = []
        self.time_of_last_move = time.time()
        self.grid = np.zeros((self.NROWS, self.NCOLS))

        # Positions

    def get_int_dir(self) -> Tuple[int, int]:
        dir = self.dirs[self.cur_dir]
        return dir

    def update_grid_coords(self, row: int, col: int, value: int) -> None:
        self.grid[row, col] = value


    def reset_snake_state(self) -> None:
        self.snake_snoot_coords = (-1, -1)
        self.apple_pos = (-1, -1)
        self.cur_path = []
        self.snake_body = []
        self.time_of_last_move = time.time()
        self.grid = np.zeros((self.NROWS, self.NCOLS))