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

def plot_fft(fft_result, title):

    fig, ax = plt.subplots()

    # Shift the zero-frequency component to the center for visualization
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate the magnitude (absolute value)
    magnitude = np.abs(fft_shifted)

    # Scale the magnitude to make it easier to see
    magnitude_scaled = np.log1p(magnitude)

    cax = ax.imshow(magnitude_scaled, cmap='gray', interpolation='none', extent=[0, magnitude_scaled.shape[1], 0, magnitude_scaled.shape[0]])

    ax.set_title(title)

    cbar = fig.colorbar(cax, ax=ax)

    return fig, ax
