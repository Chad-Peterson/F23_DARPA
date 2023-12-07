import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def make_3d(array_2d):
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
        return path
    else:
        print("No path exists between start and end nodes.")
        return None

def find_shortest_paths(G, nodes_dict, edges):
    paths = []
    for edge in edges:
        path = find_shortest_path(G, nodes_dict, edge)
        paths.append(path)
    return paths



def plot_grid_3d(grid, title="3D Grid Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # If the grid is 2D, convert it to 3D
    if len(grid.shape) == 2:
        grid = make_3d(grid)

    n, m, p = grid.shape
    x, y, z = np.indices((n + 1, m + 1, p + 1))

    # Create cubes of uniform size
    cube_size = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if grid[i, j, k] != 0:  # Plot only non-zero cells
                    # ax.bar3d(i, j, k, cube_size, cube_size, cube_size, color=plt.cm.gray_r(grid[i, j, k]), alpha=0.7,
                    #          shade=True)
                    ax.scatter(i, j, k, color=plt.cm.gray_r(grid[i, j, k]), alpha=0.7, s=100)

    # Set the view, labels, and title
    ax.view_init(elev=30, azim=30)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)

    # Setting the ticks for x, y, z axis
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.set_zticks(np.arange(p))

    plt.show()

def plot_grid_graph(G, path, grid):
    fig, ax = plt.subplots()
    # Plot the grid
    for (y, x) in G.nodes():
        ax.text(x, y, s=str(grid[y][x]), horizontalalignment='center', verticalalignment='center')
    for ((y1, x1), (y2, x2)) in G.edges():
        ax.plot([x1, x2], [y1, y2], color="black")

    # Highlight the shortest path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos={node: (node[1], node[0]) for node in G.nodes()}, edgelist=path_edges,
                           edge_color='orange', width=4, ax=ax)

    # Highlight the start and end nodes
    path_start_node = path[0]
    path_end_node = path[-1]

    nx.draw_networkx_nodes(G, pos={node: (node[1], node[0]) for node in G.nodes()}, nodelist=[path_start_node],
                           node_color='green', node_size=200, ax=ax)

    nx.draw_networkx_nodes(G, pos={node: (node[1], node[0]) for node in G.nodes()}, nodelist=[path_end_node],
                           node_color='red', node_size=200, ax=ax)

    # Set limits and labels
    ax.set_xlim(-1, grid.shape[1])
    ax.set_ylim(-1, grid.shape[0])
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()


def plot_grid_graph_3d(G, paths, grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming grid is 3D, plot the grid values at their corresponding nodes
    for (z, y, x) in G.nodes():
        ax.text(x, y, z, s=str(grid[z][y][x]), horizontalalignment='center', verticalalignment='center', color='gray', alpha=0.15)

    # Plot the edges
    for ((z1, y1, x1), (z2, y2, x2)) in G.edges():
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", linewidth=2, alpha=0.15)

    # Plot the obstacles
    for (z, y, x) in G.nodes():
        if grid[z][y][x] == 1:
            # ax.bar3d(x, y, z, 1, 1, 1, color='black', alpha=0.5)
            ax.scatter(x, y, z, color='black', alpha=0.75, s=500)

    for path in paths:
        # Highlight the shortest path
        path_edges = list(zip(path, path[1:]))
        for edge in path_edges:
            xline = [edge[0][2], edge[1][2]]
            yline = [edge[0][1], edge[1][1]]
            zline = [edge[0][0], edge[1][0]]
            ax.plot(xline, yline, zline, color='orange', linewidth=4)

        # Highlight the start and end nodes
        path_start_node = path[0]
        path_end_node = path[-1]

        ax.scatter(path_start_node[2], path_start_node[1], path_start_node[0], color='green', s=200)
        ax.scatter(path_end_node[2], path_end_node[1], path_end_node[0], color='red', s=200)

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title("3D Grid Graph")

    plt.show()


if __name__ == '__main__':

    # Example 2D grid (1s and 0s, where 0s represent obstacles)
    grid = np.ones((5, 5))
    grid[1][2] = 0  # Example obstacle

    # Create a graph from the grid
    G = numpy_array_to_networkx_grid_3d(grid)

    # Define start and end nodes (ensure these are not obstacle cells)
    start_node = (0, 0)
    end_node = (4, 4)

    # Check if path exists
    if nx.has_path(G, start_node, end_node):
        path = nx.shortest_path(G, source=start_node, target=end_node)
        print("Shortest path:", path)

        # Plot the grid graph
        plot_grid_graph(G, path, grid)
    else:
        print("No path exists between start and end nodes.")



