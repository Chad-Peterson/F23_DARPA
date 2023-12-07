import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_grid(grid, title):

    # Create a custom colormap (0: white, 1: black)
    colors = [(1, 1, 1), (0, 0, 0)]  # White to black
    cmap_name = 'custom'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    fig, ax = plt.subplots()

    ax.set_title(title)

    # Plot the workspace
    cax = ax.imshow(grid, cmap=custom_cmap, interpolation='none',
                    extent=[0, grid.shape[1], 0, grid.shape[0]],
                    vmin=0, vmax=1)

    # Draw the grid lines
    ax.grid(True)

    # Set the tick locations and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(0, grid.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, grid.shape[0]+1, 1))

    ax.set_aspect('equal')

    # Add color bar; 0 = white, 1 = black
    # cbar = fig.colorbar(ax.imshow(grid, cmap='gray_r', interpolation='none', extent=[0, grid.shape[1], 0, grid.shape[0]]), ax=ax)

    # cbar = fig.colorbar(ax.imshow(grid, cmap='gray_r', interpolation='none', extent=[0, grid.shape[1], 0, grid.shape[0]]), ax=ax)

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0', '1'])  # Optionally set labels


    return fig, ax


def plot_fft(fft_result, title):

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

    plt.show()


def convolve(a, w, plot=False):

    # Perform the FFTs
    w_fft = np.fft.fft2(w)

    a_flipped = np.flip(a, axis=1)
    a_flipped_fft = np.fft.fft2(a_flipped)

    # Perform the pointwise product
    pointwise_product = w_fft * a_flipped_fft

    # Perform the inverse FFT
    convolution = np.fft.ifft2(pointwise_product)

    # Take the real part of the result
    convolution = np.real(convolution)

    if plot:

        # Plot the FFT of the workspace
        plot_fft(w_fft, 'FFT of Workspace')
        plot_fft(a_flipped_fft, 'FFT of Flipped Object')
        plot_fft(pointwise_product, 'Pointwise Product')

    return convolution


def determine_optimal_offset(w, a):

    convolution_wa = convolve(w, a)

    # Trim the convolution to prevent object placement outside the workspace


    # Find the bottom-leftmost zero in the convolution
    bottom_leftmost_zero = np.argwhere(convolution_wa == 0)[0]

    # Determine the offset
    offset = np.array([a.shape[0] - bottom_leftmost_zero[0], bottom_leftmost_zero[1]])

    return offset