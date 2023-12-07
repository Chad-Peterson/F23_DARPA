import numpy as np

def merge_grids(grid1, grid1_rgb, grid2):

    """Superimpose grid2 on grid1
    Returns two grids.
    First, one that adds the nonzero cells of grid2 to grid1, but assigns a unique color to the grid2 entries.
    Second, one that simply adds the nonzero cells of grid2 to grid1."""


    # Find the cells in grid2 that are not empty
    grid2_nonzero = grid2 > 0

    # Pick a random RGB color for the nonzero cells of grid2
    color = np.random.rand(3)

    merged_grid = grid1 + grid2_nonzero

    merged_grid_rgb = grid1_rgb + np.stack([grid2_nonzero * color[0], grid2_nonzero * color[1], grid2_nonzero * color[2]], axis=2)

    return merged_grid, merged_grid_rgb