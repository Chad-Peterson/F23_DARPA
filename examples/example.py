import numpy as np
import networkx as nx
from task_1_extension.harmonic_analysis import convolve, determine_optimal_offset
# from task_1_extension.discrete_geometry import merge_grids
from task_1_extension.placement import initialize_domain, initialize_object, find_valid_offset, place_object, generate_placement, plot_grid
from task_1_extension.path_planning import make_3d, numpy_array_to_networkx_grid_3d, plot_grid_graph, plot_grid_3d, plot_grid_graph_3d, get_port_nodes,\
    find_shortest_path, find_shortest_paths

# Set the random seed for reproducibility
import random
random.seed(9)


# %% Define the workspace

# Dimensions of the workspace
n, m = 16, 16

w = initialize_domain(n, m, one_pad=False)
plot_grid(w, 'Workspace')


# %% Define the Objects

# Object A1
a1_rows = [5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
a1_cols = [6, 6, 7, 6, 7, 8, 6, 7, 8, 9, 6, 7, 8]
a1_port_rows = [5, 9]
a1_port_cols = [6, 8]
a1, a1_ports = initialize_object('a1', n, m, a1_rows, a1_cols, a1_port_rows, a1_port_cols)
# plot_grid(a1, 'Object A1')


# Object A2
a2_rows = [5, 6, 6, 7, 7, 7, 8, 8]
a2_cols = [6, 6, 7, 6, 7, 8, 6, 7]
a2_port_rows = [6, 8]
a2_port_cols = [6, 6]
a2, a2_ports = initialize_object('a2', n, m, a2_rows, a2_cols, a2_port_rows, a2_port_cols)
# plot_grid(a2, 'Object A2')

# Object A3
a3_rows = [6, 6, 7, 7, 8, 8]
a3_cols = [6, 7, 6, 7, 6, 7]
a3_port_rows = [6, 8]
a3_port_cols = [7, 7]
a3, a3_ports = initialize_object('a3', n, m, a3_rows, a3_cols, a3_port_rows, a3_port_cols)
# plot_grid(a3, 'Object A3')

# Object A4
a4_rows = [7, 7, 8, 8]
a4_cols = [6, 7, 6, 7]
a4_port_rows = [7, 7]
a4_port_cols = [6, 8]
a4, a4_ports = initialize_object('a4', n, m, a4_rows, a4_cols, a4_port_rows, a4_port_cols)
# plot_grid(a4, 'Object A4')

# %% Place the Objects

# Place the objects
placement, port_placement = generate_placement(w, [a1, a2, a3, a4], [a1_ports, a2_ports, a3_ports, a4_ports])
plot_grid(placement, 'Placement')
# plot_grid(port_placement, 'Port Placement')

# Define port connections
edges = [('a1_1', 'a2_0'), ('a2_1', 'a3_0'), ('a3_1', 'a4_0'), ('a4_1', 'a1_0')]


G = numpy_array_to_networkx_grid_3d(placement, port_placement)

nodes_dict = get_port_nodes(port_placement)

# Find the shortest path
path = find_shortest_path(G, nodes_dict, ('a1_1', 'a4_0'))

grid = make_3d(placement)

# Plot the grid graph
plot_grid_graph_3d(G, [path], grid)

# %% Create the Graphs



# Define start and end nodes (ensure these are not obstacle cells)
# start_node = (0, 0, 0)
# end_node = (5, 5, 2)
#
# # Check if path exists
# if nx.has_path(G, start_node, end_node):
#     path = nx.shortest_path(G, source=start_node, target=end_node)
#     print("Shortest path:", path)
#
#     grid = make_3d(placement)
#
#     # Plot the grid graph
#     plot_grid_graph_3d(G, path, grid)
# else:
#     print("No path exists between start and end nodes.")

# Plot the graph
# plot_grid_graph(G, [], placement)

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
