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


def numpy_array_to_networkx_grid_3d(array):

    array = make_3d(array)

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
                    ax.bar3d(i, j, k, cube_size, cube_size, cube_size, color=plt.cm.gray_r(grid[i, j, k]), alpha=0.7,
                             shade=True)

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



