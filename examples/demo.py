import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

from utils import plot_grid, plot_fft, plot_grid_rgb, merge_grids

# %% Define the workspace

w = np.zeros((16, 16))

w_fig, w_ax = plot_grid(w, 'Workspace')
w_fig.show()


# %% Define the Objects

a1 = np.zeros((16, 16))
a1_rows = [5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
a1_cols = [6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 6, 7, 8]
a1[a1_rows, a1_cols] = 1

# Plot the object
a1_fig, a1_ax = plot_grid(a1, 'Object A1')
a1_fig.show()

a2 = np.zeros((16, 16))
a2_rows = [5, 6, 6, 7, 7, 7, 8, 8]
a2_cols = [6, 6, 7, 6, 7, 8, 6, 7]
a2[a2_rows, a2_cols] = 1

# Plot the object
a2_fig, a2_ax = plot_grid(a2, 'Object A2')
a2_fig.show()

a3 = np.zeros((16, 16))
a3_rows = [6, 6, 7, 7, 8, 8]
a3_cols = [6, 7, 6, 7, 6, 7]
a3[a3_rows, a3_cols] = 1

# Plot the object
a3_fig, a3_ax = plot_grid(a3, 'Object A3')
a3_fig.show()




# %% Perform the FFTs

# w_fft = np.fft.fft2(w)
#
# # Plot the FFT of the workspace
# w_fft_fig, w_fft_ax = plot_fft(w_fft, 'FFT of Workspace')
# w_fft_fig.show()
#
# # Flip a first
# a_flipped = np.flip(a,axis=1)
#
# a_flipped_fft = np.fft.fft2(a_flipped)
#
#
#
# # Plot the FFT of the flipped object
# a_flipped_fft_fig, a_flipped_fft_ax = plot_fft(a_flipped_fft, 'FFT of Flipped Object')
# a_flipped_fft_fig.show()
#
#
# pointwise_product =  a_flipped_fft * w_fft
#
# # Plot the pointwise product
# pointwise_product_fig, pointwise_product_ax = plot_fft(pointwise_product, 'Pointwise Product')
# pointwise_product_fig.show()
#
# # %% Perform the inverse FFT
#
# convolution = np.fft.ifft2(pointwise_product)
#
# # Plot the convolution
# convolution_fig, convolution_ax = plot_fft(convolution, 'Convolution', display_boundary=True)
# convolution_fig.show()
#
# merged_grids, merged_grids_rgb = merge_grids(w, w_rgb, a)
#
# merged_grids_fig, merged_grids_ax = plot_grid(merged_grids, 'Merged Grids')
# merged_grids_fig.show()

