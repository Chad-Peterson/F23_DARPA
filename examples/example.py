import numpy as np
from task_1_extension.harmonic_analysis import convolve, determine_optimal_offset
# from task_1_extension.discrete_geometry import merge_grids
from task_1_extension.placement_3d import initialize_domain, initialize_object, find_valid_offset, place_object, generate_placement, plot_grid

# %% Define the workspace

# %% Define the workspace

# Dimensions of the workspace
n, m = 16, 16

w = initialize_domain(n, m)
plot_grid(w, 'Workspace')


# %% Define the Objects

# Object A1
a1_rows = [5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
a1_cols = [6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 6, 7, 8]
a1 = initialize_object(n, m, a1_rows, a1_cols)
plot_grid(a1, 'Object A1')

# Object A2
a2_rows = [5, 6, 6, 7, 7, 7, 8, 8]
a2_cols = [6, 6, 7, 6, 7, 8, 6, 7]
a2 = initialize_object(n, m, a2_rows, a2_cols)
# plot_grid(a2, 'Object A2')

# Object A3
a3_rows = [6, 6, 7, 7, 8, 8]
a3_cols = [6, 7, 6, 7, 6, 7]
a3 = initialize_object(n, m, a3_rows, a3_cols)
# plot_grid(a3, 'Object A3')

# convolution_wa = convolve(w, a1, plot=True)
# plot_grid(convolution_wa, 'Convolution of W and A1')
# # a1_offset = determine_optimal_offset(w, a1)
# a1_offset = (6, 6)

# a1_offset = find_valid_offset(w, a1)
# w_updated = place_object(w, a1, a1_offset)
# plot_grid(w_updated, 'Workspace with Object A1')
#
# a2_offset = find_valid_offset(w_updated, a2)
# w_updated = place_object(w_updated, a2, a2_offset)
# plot_grid(w_updated, 'Workspace with Object A2')
#
# a3_offset = find_valid_offset(w_updated, a3)
# w_updated = place_object(w_updated, a3, a3_offset)
# plot_grid(w_updated, 'Workspace with Object A3')

# Add the shifted object to the workspace
# w_updated = place_object(w, a3, (1, 3))
# plot_grid(w_updated, 'Workspace with Object A3')
#
# w_updated = place_object(w_updated, a2, (-1, -6))
# plot_grid(w_updated, 'Workspace with Object A2')

placement = generate_placement(w, [a1, a2, a3])
plot_grid(placement, 'Placement')

# %% Convolve the Objects with the Workspace

# # Convolve the objects with the workspace
# a1_convolution = convolve(a1, w, plot=True)
# a2_convolution = convolve(a2, w)
# a3_convolution = convolve(a3, w)
#
#
# # %% Place the objects
#
# offset_a1 = determine_optimal_offset(a1, w)
#
# # Place the object
# w_updated = merge_grids(w, a1, offset_a1)
#
# # Plot the result
# w_updated_fig, w_updated_ax = plot_grid(w_updated, 'Workspace with Object A1')
# w_updated_fig.show()

# %% Place the objects
