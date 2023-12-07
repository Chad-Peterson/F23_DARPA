import networkx as nx
import numpy as np
from itertools import product


def make_3d(array_2d):
    """
    Converts a 2D array to a 3D array by stacking two 2D grids of zeros on either side of the 2D array
    :param array_2d: 2D array
    :return: 3D array
    """
    # Get the shape of the 2D array
    n, m = array_2d.shape

    # Create two 2D grids of zeros with the same shape
    zeros_grid = np.zeros((n, m))

    # Stack the 2D array between the two grids of zeros
    array_3d = np.dstack((zeros_grid, array_2d, zeros_grid))

    return array_3d

def make_3d_port_mask(array_2d):
    # Get the shape of the 2D array
    n, m = array_2d.shape

    # Create two 2D grids of zeros with the same shape
    zeros_grid = np.zeros((n, m))

    # Insert nonzero values as ones
    center_grid = np.zeros((n, m))
    center_grid[array_2d != '0.0'] = 1

    # Stack the 2D array between the two grids of zeros
    array_3d = np.dstack((zeros_grid, center_grid, zeros_grid))

    return array_3d


def numpy_array_to_networkx_grid_3d(placement, port_placement):

    # If the array is 2D, convert it to 3D
    if len(placement.shape) == 2:
        placement = make_3d(placement)

    if len(port_placement.shape) == 2:
        port_placement = make_3d_port_mask(port_placement)

    # Set ports to zero so that they are kept as nodes
    array = placement - port_placement

    n, m, p = array.shape
    G = nx.Graph()

    # Create nodes and edges for a 3D grid
    for i, j, k in product(range(n), range(m), range(p)):
        G.add_node((i, j, k))
        # Add edges to neighbors in the grid
        if i > 0:
            G.add_edge((i, j, k), (i-1, j, k))
        if j > 0:
            G.add_edge((i, j, k), (i, j-1, k))
        if k > 0:
            G.add_edge((i, j, k), (i, j, k-1))

    # Iterate over the array and remove nodes that correspond to obstacles
    for i, j, k in product(range(n), range(m), range(p)):
        if array[i, j, k] == 1:  # Assuming 1 represents an obstacle
            if (i, j, k) in G:
                G.remove_node((i, j, k))

    return G

def get_port_nodes(port_placement):

    n, m = port_placement.shape
    port_nodes = {}
    for i, j in product(range(n), range(m)):
        k = 0 # Assuming 2D port placement
        if port_placement[i, j] != '0.0':
            port_nodes[port_placement[i, j]] = (i, j, k)
    return port_nodes


def find_shortest_path(G, nodes_dict, edge):

    # Obtain the start and end nodes
    start_node_str, end_node_str = edge

    # Obtain the port nodes
    start_node = nodes_dict[start_node_str]
    end_node = nodes_dict[end_node_str]

    # Shift z coordinate by 1 since the port nodes are in the middle of the grid
    start_node = (start_node[0], start_node[1], start_node[2] + 1)
    end_node = (end_node[0], end_node[1], end_node[2] + 1)

    # Check if path exists
    if nx.has_path(G, start_node, end_node):
        path = nx.shortest_path(G, source=start_node, target=end_node)
        print("Shortest path:", path)

        # Remove the path from the graph
        G.remove_nodes_from(path)

        return G, path
    else:
        raise Exception("No path exists between start and end nodes.")


def find_shortest_paths(G, nodes_dict, edges):
    paths = []
    for edge in edges:
        G, path = find_shortest_path(G, nodes_dict, edge)
        paths.append(path)
    return paths




