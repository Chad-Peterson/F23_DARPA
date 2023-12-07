import numpy as np
from task_1_extension.placement import initialize_domain, initialize_object, generate_placement
from task_1_extension.path_planning import numpy_array_to_networkx_grid_3d, get_port_nodes, find_shortest_paths
from task_1_extension.visualization import plot_grid, plot_grid_graph_3d
from yamada import SpatialGraph

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

# %% Define the System Connectivity

components = [a1, a2, a3, a4]
components_ports = [a1_ports, a2_ports, a3_ports, a4_ports]
nodes = ['a1_0', 'a1_1', 'a2_0', 'a2_1', 'a3_0', 'a3_1', 'a4_0', 'a4_1']
edges = [('a1_1', 'a2_0'), ('a2_1', 'a3_0'), ('a3_1', 'a4_0'), ('a4_1', 'a1_0')]

# %% Place the Objects

max_iter = 1

yamada_polynomials = []



for i in range(max_iter):
    try:

        # Find a valid placement for the objects
        placement, port_placement = generate_placement(w, components, components_ports)
        plot_grid(placement, 'Placement')

        # Convert the placements to a networkx grid graph
        G = numpy_array_to_networkx_grid_3d(placement, port_placement)

        # Get the nodes for the ports
        nodes_dict = get_port_nodes(port_placement)

        # Find the shortest path (for each interconnect)
        paths = find_shortest_paths(G, nodes_dict, edges)

        # Plot the grid graph
        plot_grid_graph_3d(G, paths, placement)

    except:
        print("No path exists between start and end nodes.")

    # %% Create the Spatial Graph Diagrams

    nodes = ['a1_0', 'a1_1', 'a2_0', 'a2_1', 'a3_0', 'a3_1', 'a4_0', 'a4_1']

    # add edges between the nodes of a component
    internal_edges = [('a1_0', 'a1_1'), ('a2_0', 'a2_1'), ('a3_0', 'a3_1'), ('a4_0', 'a4_1')]
    edges = edges + internal_edges

    # Convert the node positions from tuples to numpy arrays
    # node_positions = {node: np.array(nodes_dict[node]) for node in nodes_dict}

    # Convert nodes_dict into a 2D array where each row is a node and the columns are the x, y, and z coordinates
    node_positions = np.array([nodes_dict[node] for node in nodes_dict])

    # Create the spatial graph diagram
    sg1 = SpatialGraph(nodes=nodes, edges=edges, node_positions=node_positions)
    sg1.plot()
    sgd1 = sg1.create_spatial_graph_diagram()

    yp1 = sgd1.normalized_yamada_polynomial()
    print(yp1)
