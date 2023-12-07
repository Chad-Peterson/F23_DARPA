import numpy as np

def merge_grids(grid1, grid2):

    """Superimpose grid2 on grid1."""

    # Find the cells in grid2 that are not empty
    grid2_nonzero = grid2 > 0

    # Pick a random RGB color for the nonzero cells of grid2
    color = np.random.rand(3)

    merged_grid = grid1 + grid2_nonzero

    return merged_grid