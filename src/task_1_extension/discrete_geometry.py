import numpy as np


def shift_and_trim_grid(grid, row_shift, col_shift):

    """
    Shifts the grid by the specified number of rows and columns and trims the excess.
    :param grid: 2D numpy array
    :param row_shift: int
    :param col_shift: int
    :return: 2D numpy array
    """

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

