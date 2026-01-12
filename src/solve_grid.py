from collections import deque
import numpy as np
from typing import Optional, Tuple


def breadth_first_search(
    grid: np.ndarray, start_x: int, start_y: int, goal_x: int, goal_y: int
) -> Optional[list[Tuple[int, int]]]:
    if any(coord is None or coord < 0 for coord in [start_x, start_y, goal_x, goal_y]):
        return
    queue = deque([(start_y, start_x)])
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    parents: list[list[Optional[Tuple[int, int]]]] = [
        [None for _ in range(grid.shape[1])] for _ in range(grid.shape[0])
    ]
    parents[start_y][start_x] = (-1, -1)

    # Explore neighbours
    while queue:
        y, x = queue.popleft()

        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if parents[ny][nx] is None and grid[ny, nx] != 1:
                parents[ny][nx] = (y, x)
                queue.append((ny, nx))

    # No solution
    if parents[goal_y][goal_x] is None:
        return []  # No path found

    path = []
    cury = (goal_y, goal_x)

    # Trace back from target to start
    while cury and cury != (-1, -1):
        path.append(cury)
        cury = parents[cury[0]][cury[1]]

        # Stop if we reach start (start's parent is -1,-1)
        if cury == (-1, -1):
            break

    return path[::-1]
