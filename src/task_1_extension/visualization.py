import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from task_1_extension.path_planning import make_3d


def plot_grid(grid, title="Grid Plot"):
    """
    Plots a grid
    :param grid:
    :param title:
    :return:
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1)  # You can choose different colormaps like 'gray', 'viridis', etc.

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Value')

    # Add title and labels as needed
    ax.set_title(title)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')

    plt.show()


def plot_grid_3d(grid, title="3D Grid Plot"):
    """
    Plots a 3D grid
    :param grid: The 3D grid to plot
    :param title: The title of the plot
    :return: None
    """
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
    """
    Plots a grid graph
    :param G: The grid graph
    :param path: The shortest path
    :param grid: The grid
    :return:
    """
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
    """
    Plots a 3D grid graph
    :param G: The grid graph
    :param paths: The shortest paths
    :param grid: The grid
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # If the grid is 2D, convert it to 3D
    if len(grid.shape) == 2:
        grid = make_3d(grid)

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

def plot_grid_rgb(grid_rgb, title):

    fig, ax = plt.subplots()

    ax.set_title(title)

    # Plot the workspace
    # ax.imshow(grid_rgb, interpolation='none', extent=[0, grid_rgb.shape[1], 0, grid_rgb.shape[0]])


    # Draw the grid lines
    ax.grid(True)

    # Set the tick locations and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(0, grid_rgb.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, grid_rgb.shape[0]+1, 1))

    ax.set_aspect('equal')

    return fig, ax
