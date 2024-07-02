import numpy as np
import matplotlib.pyplot as plt


# %% Define functions


def plot_bitmap(bitmap, title):

    fig, ax = plt.subplots()

    ax.set_title(title)

    # Plot the bitmap
    ax.imshow(bitmap, cmap='gray_r', interpolation='none', extent=[0, bitmap.shape[1], 0, bitmap.shape[0]])

    # Draw the grid lines
    ax.grid(True)

    # Set the tick locations and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(0, bitmap.shape[1] + 1, 1))
    ax.set_yticks(np.arange(0, bitmap.shape[0] + 1, 1))

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('equal')

    # Add color bar
    fig.colorbar(ax.imshow(bitmap, cmap='gray_r', interpolation='none', extent=[0, bitmap.shape[1], 0, bitmap.shape[0]]), ax=ax)

    return fig, ax


def plot_fft(fft_coeffs, title):

    fig, ax = plt.subplots()

    # Shift the zero-frequency component to the center for visualization
    fft_shifted = np.fft.fftshift(fft_coeffs)

    # Calculate the magnitude
    magnitude = np.abs(fft_shifted)

    # Scale the magnitude to make it easier to see
    magnitude_scaled = np.log1p(magnitude)

    cax = ax.imshow(magnitude_scaled, cmap='gray_r', interpolation='none', extent=[0, magnitude_scaled.shape[1], 0, magnitude_scaled.shape[0]])

    ax.set_title(title)

    fig.colorbar(cax, ax=ax)

    return fig, ax


def offset_and_trim_bitmap(bitmap, row_offset, col_offset):
    """
    Helper function to combine_bitmaps.

    If you want to combine two same-sized bitmaps, but the center position of one is offset
    from the center other, then you must trim the non-overlapping parts of one so that the
    combined bitmap retains the same size as the original bitmaps.

    :param bitmap: 2D numpy array
    :param row_offset: integer
    :param col_offset: integer
    :return: 2D numpy array
    """

    n, m = bitmap.shape

    # Handle row offset
    if row_offset > 0:
        # Offset down
        offset_bitmap = np.concatenate((np.zeros((row_offset, m)), bitmap[:-row_offset]), axis=0)
    else:
        # Offset up
        offset_bitmap = np.concatenate((bitmap[-row_offset:], np.zeros((-row_offset, m))), axis=0)

    # Handle column offset
    if col_offset > 0:
        # Offset right
        offset_bitmap = np.concatenate((np.zeros((n, col_offset)), offset_bitmap[:, :-col_offset]), axis=1)
    else:
        # Offset left
        offset_bitmap = np.concatenate((offset_bitmap[:, -col_offset:], np.zeros((n, -col_offset))), axis=1)

    return offset_bitmap


def combine_bitmaps(bitmap_1, bitmap_2, offset=(0, 0)):

    """Combine bitmap_1 and bitmap_2. Bitmap_2 may be offset from bitmap_1"""

    # Initialize the combined bitmap array as bitmap_1 (the reference bitmap)
    combined_bitmap = np.array(bitmap_1)

    # Unpack the offsets
    row_offset, col_offset = offset

    # Offset and trim the second bitmap (see helper function for more details)
    bitmap_2_offset = offset_and_trim_bitmap(bitmap_2, row_offset, col_offset)

    # Add the second grid to the result
    combined_bitmap += bitmap_2_offset

    return combined_bitmap


# %% Manually define the workspace bitmap (based on Cui et al. Figure 2)

ws = np.zeros((16, 16))

ws_rows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + \
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

ws_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + \
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

ws[ws_rows, ws_cols] = 1

# Plot the workspace
ws_fig, ws_ax = plot_bitmap(ws, 'Workspace Bitmap')
ws_fig.show()


# %% Manually define the object bitmap (based on Cui et al. Figure 2)

obj = np.zeros((16, 16))

obj_rows = [5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
obj_cols = [6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 6, 7, 8]

obj[obj_rows, obj_cols] = 1

# Plot the object
obj_fig, obj_ax = plot_bitmap(obj, 'Object Bitmap')
obj_fig.show()

# %% Perform the FFT Convolution of the object and workspace

# Perform the FFT of the workspace
ws_fft = np.fft.fft2(ws)

# Plot the FFT of the workspace
ws_fft_fig, ws_fft_ax = plot_fft(ws_fft, 'FFT of Workspace Bitmap')
ws_fft_fig.show()

# Flip the object first (a necessary convention for performing the convolution)
obj_flipped = np.flip(obj, axis=1)

# Perform the FFT of the flipped object
obj_flipped_fft = np.fft.fft2(obj_flipped)

# Plot the FFT of the flipped object
obj_flipped_fft_fig, obj_flipped_fft_ax = plot_fft(obj_flipped_fft, 'FFT of Flipped Object Bitmap')
obj_flipped_fft_fig.show()

# Perform the FFT convolution in the frequency domain
# This is done as the point-wise product of the FFT coefficients.
convolution_freq = obj_flipped_fft * ws_fft

# Plot the convolution in the frequency domain
convolution_freq_fig, convolution_freq_ax = plot_fft(convolution_freq, 'Convolution in Frequency Domain')
convolution_freq_fig.show()

# Convert the convolution in the frequency domain back to the spatial domain
# This is done through the inverse FFT
convolution_spatial = np.fft.ifft2(convolution_freq)

# Plot the convolution in the spatial domain
convolution_fig, convolution_ax = plot_fft(convolution_spatial, 'Convolution in the Spatial Domain')
convolution_fig.show()

# %% Now place the object in the workspace

# Place the object in the workspace
user_specified_offset = (0, 0)
merged_grids = combine_bitmaps(obj, ws, user_specified_offset)

# Plot the placed object in the workspace
merged_grids_fig, merged_grids_ax = plot_bitmap(merged_grids, 'Merged Bitmaps')
merged_grids_fig.show()




