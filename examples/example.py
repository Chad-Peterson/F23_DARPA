import numpy as np
from task_1_extension.harmonic_analysis import convolve, plot_grid, determine_optimal_offset
from task_1_extension.discrete_geometry import merge_grids

# %% Define the workspace

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

# %% Convolve the Objects with the Workspace

# Convolve the objects with the workspace
a1_convolution = convolve(a1, w, plot=True)
a2_convolution = convolve(a2, w)
a3_convolution = convolve(a3, w)


# %% Place the objects

offset_a1 = determine_optimal_offset(a1, w)

# Place the object
w_updated = merge_grids(w, a1, offset_a1)

# Plot the result
w_updated_fig, w_updated_ax = plot_grid(w_updated, 'Workspace with Object A1')
w_updated_fig.show()