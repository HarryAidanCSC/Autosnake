from collections import deque
import numpy as np
from typing import Optional, Tuple
from SnakeState import SnakeState


def breadth_first_search(state: SnakeState) -> Optional[list[Tuple[int, int]]]:
    """Find shortest path from start to goal using BFS.

    Args:
        TBC

    Returns:
        List of (x, y) tuples from start to goal (excluding start),
        or [] if no path, or None if invalid input

    Note:
        Internally uses (y, x) for numpy array indexing, but returns (x, y) format.
    """
    # Validate inputs
    start_x, start_y = state.snake_snoot_coords
    goal_x, goal_y = state.apple_pos

    # Get grid dimensions for bounds checking
    nrows, ncols = state.NROWS, state.NCOLS

    # Validate coordinates are within bounds
    if not (0 <= start_x < ncols and 0 <= start_y < nrows):
        return None
    if not (0 <= goal_x < ncols and 0 <= goal_y < nrows):
        return None

    # Initialise BFS
    dx, dy = state.get_int_dir()
    start_arrived_from = (dy, dx)
    queue = deque([(start_y, start_x, start_arrived_from)])
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    parents: list[list[Optional[Tuple[int, int]]]] = [
        [None for _ in range(ncols)] for _ in range(nrows)
    ]
    parents[start_y][start_x] = (-1, -1)

    # Explore neighbours
    while queue:
        y, x, arrived_from = queue.popleft()

        # Filter out the opposite direction of how we arrived
        if arrived_from is not None:
            # If we arrived from direction (dy, dx), we can't go back (-dy, -dx)
            opposite = (-arrived_from[0], -arrived_from[1])
            explore_dirs = [d for d in dirs if d != opposite]
        else:
            # At start position, can explore all directions
            explore_dirs = dirs

        for dy, dx in explore_dirs:
            ny, nx = y + dy, x + dx

            if not (0 <= ny < nrows and 0 <= nx < ncols):
                continue

            if parents[ny][nx] is None and state.grid[ny, nx] != 1:
                parents[ny][nx] = (y, x)
                # Store the direction we used to arrive at (ny, nx)
                queue.append((ny, nx, (dy, dx)))

    # No solution
    if parents[goal_y][goal_x] is None:
        return []

    path = []
    cury = (goal_y, goal_x)

    # Trace back from target to start
    while cury and cury != (-1, -1):
        # BFS uses (y, x) internally, but we'll convert to (x, y) for output
        path.append((cury[1], cury[0]))  # Swap to (x, y) format
        cury = parents[cury[0]][cury[1]]

        # Stop if we reach start
        if cury == (-1, -1):
            break

    # Reverse to get path from start to goal
    path = path[::-1]

    # Remove start position from path
    if path and path[0] == (start_x, start_y):  # Now comparing (x, y) format
        path = path[1:]

    return path
