from collections import deque
import numpy as np
from typing import Optional, Tuple


def breadth_first_search(
    grid: np.ndarray, start_x: int, start_y: int, goal_x: int, goal_y: int
) -> Optional[list[Tuple[int, int]]]:
    """Find shortest path from start to goal using BFS.
    
    Args:
        grid: 2D array where 0=walkable, 1=obstacle
        start_x: Starting column (X coordinate)
        start_y: Starting row (Y coordinate)
        goal_x: Goal column (X coordinate)  
        goal_y: Goal row (Y coordinate)
        
    Returns:
        List of (x, y) tuples from start to goal (excluding start),
        or [] if no path, or None if invalid input
    """
    # Validate inputs
    if any(coord is None or coord < 0 for coord in [start_x, start_y, goal_x, goal_y]):
        return None
    
    # Get grid dimensions for bounds checking
    height, width = grid.shape
    
    # Validate coordinates are within bounds
    if not (0 <= start_x < width and 0 <= start_y < height):
        return None
    if not (0 <= goal_x < width and 0 <= goal_y < height):
        return None
    
    # Initialise BFS
    queue = deque([(start_y, start_x)])
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    parents: list[list[Optional[Tuple[int, int]]]] = [
        [None for _ in range(width)] for _ in range(height)
    ]
    parents[start_y][start_x] = (-1, -1)

    # Explore neighbours
    while queue:
        y, x = queue.popleft()

        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            
            if parents[ny][nx] is None and grid[ny, nx] != 1:
                parents[ny][nx] = (y, x)
                queue.append((ny, nx))

    # No solution
    if parents[goal_y][goal_x] is None:
        return []  

    path = []
    cury = (goal_y, goal_x)

    # Trace back from target to start
    while cury and cury != (-1, -1):
        path.append((cury[1], cury[0]))
        cury = parents[cury[0]][cury[1]]

        # Stop if we reach start
        if cury == (-1, -1):
            break

    # Reverse to get path from start to goal
    path = path[::-1]
    
    # Remove start position from path 
    if path and path[0] == (start_x, start_y):
        path = path[1:]
    
    return path
