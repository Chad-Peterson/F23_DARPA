import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_grid_graph(grid):
    G = nx.Graph()
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            # Check if the current cell is an obstacle
            if grid[i, j] == 0:
                continue  # Skip obstacle cells

            # Add node for non-obstacle cells
            G.add_node((i, j))

            # Add edges to left and above neighbors if they are not obstacles
            if i > 0 and grid[i-1, j] != 0:
                G.add_edge((i, j), (i-1, j))
            if j > 0 and grid[i, j-1] != 0:
                G.add_edge((i, j), (i, j-1))
    return G


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
    # grid[2][2] = 0  # Example obstacle

    # Create a graph from the grid
    G = create_grid_graph(grid)

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



