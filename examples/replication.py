import numpy as np

from utils import plot_grid, plot_fft, plot_grid_rgb, merge_grids

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

merged_grids_fig, merged_grids_ax = plot_grid_rgb(merged_grids_rgb, 'Merged Grids RGB')
merged_grids_fig.show()

merged_grids_fig, merged_grids_ax = plot_grid(merged_grids, 'Merged Grids')
merged_grids_fig.show()

