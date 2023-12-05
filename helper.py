import numpy as np
import matplotlib.pyplot as plt

def plot_grid(grid, title):

    fig, ax = plt.subplots()

    ax.set_title(title)

    # Plot the workspace
    ax.imshow(grid, cmap='gray_r', interpolation='none', extent=[0, grid.shape[1], 0, grid.shape[0]])
    # ax.title('Workspace')

    # Draw the grid lines
    ax.grid(True)

    # Set the tick locations and labels for the x-axis and y-axis
    ax.xticks(np.arange(0, grid.shape[1]+1, 1))
    ax.yticks(np.arange(0, grid.shape[0]+1, 1))

    ax.colorbar()

    ax.gca().set_aspect('equal')

    return fig, ax
