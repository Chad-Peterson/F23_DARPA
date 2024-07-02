import numpy as np
import matplotlib.pyplot as plt


# %% Define functions


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


# %% Define the workspace

w = np.zeros((16, 16))

w_rows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + \
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + \
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] + \
         [3, 3, 3, 3, 3, 3, 3, 3] + \
         [4, 4, 4, 4, 4, 4] + \
         [5, 5, 5, 5] + \
         [6, 6, 6, 6] + \
         [7, 7, 7, 7] + \
         [8, 8, 8, 8] + \
         [9, 9, 9, 9, 9, 9] + \
         [10, 10, 10, 10, 10, 10] + \
         [11, 11, 11, 11, 11, 11] + \
         [12, 12, 12, 12, 12, 12, 12, 12] + \
         [13, 13, 13, 13, 13, 13, 13, 13] + \
         [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14] + \
         [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

w_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + \
         [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15] + \
         [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15] + \
         [0, 1, 2, 3, 12, 13, 14, 15] + \
         [0, 1, 2, 13, 14, 15] + \
         [0, 1, 14, 15] + \
         [0, 1, 14, 15] + \
         [0, 1, 14, 15] + \
         [0, 1, 14, 15] + \
         [0, 1, 2, 13, 14, 15] + \
         [0, 1, 2, 13, 14, 15] + \
         [0, 1, 2, 13, 14, 15] + \
         [0, 1, 2, 3, 12, 13, 14, 15] + \
         [0, 1, 2, 3, 12, 13, 14, 15] + \
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + \
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

w[w_rows, w_cols] = 1

# Plot the workspace
w_fig, w_ax = plot_grid(w, 'Workspace')
w_fig.show()

w_rgb = np.stack([1-w, 1-w, 1-w], axis=2)
w_rgb_fig, w_rgb_ax = plot_grid_rgb(w_rgb, 'Workspace RBG')
w_rgb_fig.show()


# %% Define the Object

a = np.zeros((16, 16))

a_rows = [5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
a_cols = [6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 6, 7, 8]

a[a_rows, a_cols] = 1

# Plot the object
a_fig, a_ax = plot_grid(a, 'Object')
a_fig.show()

# %% Perform the FFTs

w_fft = np.fft.fft2(w)

# Plot the FFT of the workspace
w_fft_fig, w_fft_ax = plot_fft(w_fft, 'FFT of Workspace')
w_fft_fig.show()

# Flip a first
a_flipped = np.flip(a,axis=1)

a_flipped_fft = np.fft.fft2(a_flipped)



# Plot the FFT of the flipped object
a_flipped_fft_fig, a_flipped_fft_ax = plot_fft(a_flipped_fft, 'FFT of Flipped Object')
a_flipped_fft_fig.show()


pointwise_product =  a_flipped_fft * w_fft

# Plot the pointwise product
pointwise_product_fig, pointwise_product_ax = plot_fft(pointwise_product, 'Pointwise Product')
pointwise_product_fig.show()

# %% Perform the inverse FFT

convolution = np.fft.ifft2(pointwise_product)

# Plot the convolution
convolution_fig, convolution_ax = plot_fft(convolution, 'Convolution', display_boundary=True)

# Color the cells that have a value of close to 0
# convolution_ax.imshow(np.real(convolution) < 0.1, cmap='Blues', interpolation='none', alpha=0.5, extent=[0, convolution.shape[1], 0, convolution.shape[0]])


convolution_fig.show()

merged_grids, merged_grids_rgb = merge_grids(w, w_rgb, a)

# merged_grids_fig, merged_grids_ax = plot_grid_rgb(merged_grids_rgb, 'Merged Grids RGB')
# merged_grids_fig.show()
#
# merged_grids_fig, merged_grids_ax = plot_grid(merged_grids, 'Merged Grids')
# merged_grids_fig.show()




