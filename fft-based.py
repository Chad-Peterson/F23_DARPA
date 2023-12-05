import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import convolve

from utils import plot_grid


# Define the workspace
w = np.zeros((10,10))

# Define the boundary
b_rows = [0,0,0,0,0,0,0,0,0,0,9,9,9,9,9,9,9,9,9,9]
b_cols = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
w[b_rows, b_cols] = 1

# Define the obstacle
o_rows = [2,2,2,2,2,2] + [8,8,8,8,8,8] + [3,4,5,6,7] + [3,7]
o_cols = [2,3,4,5,6,7] + [2,3,4,5,6,7] + [2,2,2,2,2] + [7,7]
w[o_rows, o_cols] = 1

# Make an entrance
w[[5, 6], [0, 0]] = 0

# # Plot the workspace
# plt.figure(figsize=(5,5))
# plt.imshow(w, cmap='gray_r', interpolation='none', extent=[0, w.shape[1], 0, w.shape[0]])
# plt.title('Workspace')

# # Draw the grid lines
# plt.grid(True)

# # Set the tick locations and labels for the x-axis and y-axis
# plt.xticks(np.arange(0, w.shape[1]+1, 1))
# plt.yticks(np.arange(0, w.shape[0]+1, 1))

# plt.colorbar()

# plt.gca().set_aspect('equal')

myfig, myax = plot_grid(w, 'Workspace')

# Show the plot
# myfig.show()