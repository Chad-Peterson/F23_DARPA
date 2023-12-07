import numpy as np
import matplotlib.pyplot as plt


def plot_grid_rgb(grid_rgb, title):

    fig, ax = plt.subplots()

    ax.set_title(title)

    # Plot the workspace
    # ax.imshow(grid_rgb, interpolation='none', extent=[0, grid_rgb.shape[1], 0, grid_rgb.shape[0]])


    # Draw the grid lines
    ax.grid(True)

    # Set the tick locations and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(0, grid_rgb.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, grid_rgb.shape[0]+1, 1))

    ax.set_aspect('equal')

    return fig, ax
