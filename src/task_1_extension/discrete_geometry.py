import numpy as np


def shift_and_trim_grid(grid, row_shift, col_shift):
    n, m = grid.shape

    # Handle row shift
    if row_shift > 0:
        # Shift down
        shifted_grid = np.concatenate((np.zeros((row_shift, m)), grid[:-row_shift]), axis=0)
    else:
        # Shift up
        shifted_grid = np.concatenate((grid[-row_shift:], np.zeros((-row_shift, m))), axis=0)

    # Handle column shift
    if col_shift > 0:
        # Shift right
        shifted_grid = np.concatenate((np.zeros((n, col_shift)), shifted_grid[:, :-col_shift]), axis=1)
    else:
        # Shift left
        shifted_grid = np.concatenate((shifted_grid[:, -col_shift:], np.zeros((n, -col_shift))), axis=1)

    return shifted_grid


def merge_grids(grid1, grid2, offset=(0, 0)):

    """Superimpose grid2 on grid1."""

    # Determine the size of the grids
    n, m = grid1.shape

    # Initialize the result array with the first grid
    result = np.array(grid1)

    row_offset, col_offset = offset

    # Shift the second grid
    shifted_grid2 = shift_and_trim_grid(grid2, row_offset, col_offset)

    # Add the second grid to the result
    result += shifted_grid2

    return result
