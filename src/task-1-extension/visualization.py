import numpy as np
import matplotlib.pyplot as plt


def plot_grid(grid, title):

    fig, ax = plt.subplots()

    ax.set_title(title)

    # Plot the workspace
    ax.imshow(grid, cmap='gray_r', interpolation='none', extent=[0, grid.shape[1], 0, grid.shape[0]])

    # Draw the grid lines
    ax.grid(True)

    # Set the tick locations and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(0, grid.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, grid.shape[0]+1, 1))

    ax.set_aspect('equal')

    # Add color bar
    cbar = fig.colorbar(ax.imshow(grid, cmap='gray_r', interpolation='none', extent=[0, grid.shape[1], 0, grid.shape[0]]), ax=ax)

    return fig, ax


def plot_fft(fft_result, title, display_boundary=False):

    fig, ax = plt.subplots()

    # Shift the zero-frequency component to the center for visualization
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate the magnitude (absolute value)
    magnitude = np.abs(fft_shifted)

    # Scale the magnitude to make it easier to see
    magnitude_scaled = np.log1p(magnitude)

    cax = ax.imshow(magnitude_scaled, cmap='gray_r', interpolation='none', extent=[0, magnitude_scaled.shape[1], 0, magnitude_scaled.shape[0]])

    ax.set_title(title)

    cbar = fig.colorbar(cax, ax=ax)

    # Draw a boundary around the grid cells near zero
    if display_boundary:

        threshold = 0.1

        # Create a binary mask for values below the threshold
        mask = np.where(magnitude_scaled < threshold, 1, 0)

        # Plot the original array
        ax.imshow(magnitude_scaled, cmap='grey')
        # plt.colorbar()  # To show the value scale

        # Plot the contour around near-zero values
        ax.imshow(mask, cmap='Reds', interpolation='none', alpha=0.5)

    return fig, ax


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
