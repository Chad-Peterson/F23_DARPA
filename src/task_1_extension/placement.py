import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


def initialize_domain(n, m, one_pad=True):
    """Initialize a domain of size n x m with zeros.
    One-pad the domain to avoid boundary effects.
    :param n: Number of rows
    :param m: Number of columns
    :param one_pad: Boolean to one-pad the domain
    :return: Initialized domain
    """

    domain = np.zeros((n, m))

    # One-pad the domain to avoid boundary effects
    if one_pad:
        domain = np.pad(domain, 1, 'constant', constant_values=1)
    else:
        domain = np.pad(domain, 1, 'constant', constant_values=0)

    return domain

def initialize_object(name, n, m, rows, cols, port_rows, port_cols):
    """Initialize an object of size n x m (plus padding) with ones at the specified rows and columns."""
    obj = np.zeros_like(initialize_domain(n, m))
    obj[rows, cols] = 1



    obj_ports = np.zeros_like(initialize_domain(n, m))
    obj_ports = obj_ports.astype(str)

    # Create a string for each port name
    port_names = [name+f'_{i}' for i in range(len(port_rows))]

    obj_ports[port_rows, port_cols] = port_names

    return obj, obj_ports


def find_valid_offset(workspace, object_shape):

    n, m = workspace.shape

    # Find the rows and columns in the object_shape that contain 1s
    rows_with_ones, cols_with_ones = np.where(object_shape == 1)

    # Determine the min and max rows and columns that contain 1s
    min_row, max_row = np.min(rows_with_ones), np.max(rows_with_ones)
    min_col, max_col = np.min(cols_with_ones), np.max(cols_with_ones)

    # Calculate the limits for the offsets
    row_offset_limit_low = -min_row
    row_offset_limit_high = n - 1 - max_row
    col_offset_limit_low = -min_col
    col_offset_limit_high = m - 1 - max_col

    valid_positions = []

    # Iterate over all possible positions (offsets) within the limits
    for row_offset in range(row_offset_limit_low, row_offset_limit_high + 1):
        for col_offset in range(col_offset_limit_low, col_offset_limit_high + 1):
            # Shift the object
            shifted_object = np.roll(np.roll(object_shape, row_offset, axis=0), col_offset, axis=1)

            # Check for collisions
            if np.all((workspace + shifted_object <= 1)):
                valid_positions.append((row_offset, col_offset))

    if not valid_positions:
        return None  # No valid position found

    # Randomly select one of the valid positions
    return random.choice(valid_positions)


def shift_and_trim_object(grid, row_shift, col_shift):

    n, m = grid.shape

    # Handle row shift
    if row_shift > 0:
        # Shift down
        shifted_grid = np.concatenate((np.zeros((row_shift, m)), grid[:-row_shift]), axis=0)
    else:
        # Shift up
        shifted_grid = np.concatenate((grid[-row_shift:], np.zeros((-row_shift, m))), axis=0)

    # Handle column shift
    if col_shift > 0:
        # Shift right
        shifted_grid = np.concatenate((np.zeros((n, col_shift)), shifted_grid[:, :-col_shift]), axis=1)
    else:
        # Shift left
        shifted_grid = np.concatenate((shifted_grid[:, -col_shift:], np.zeros((n, -col_shift))), axis=1)

    return shifted_grid


def place_object(grid, obj, offset=(0, 0)):

    # Shift and trim the object
    shifted_obj = shift_and_trim_object(obj, offset[0], offset[1])

    # Add the object to the grid
    grid += shifted_obj

    return grid

def place_object_ports(port_workspace, obj_ports, offset=(0, 0)):
    # Shift and trim the object
    shifted_obj_ports = shift_and_trim_object(obj_ports, offset[0], offset[1])

    # Add the nonzero elements of the object to the port_workspace
    port_workspace[shifted_obj_ports != '0.0'] = shifted_obj_ports[shifted_obj_ports != '0.0']

    return port_workspace


def generate_placement(workspace, objects, objects_ports):

    port_workspace = np.zeros_like(workspace).astype(str)
    for obj, obj_ports in zip(objects, objects_ports):
        offset = find_valid_offset(workspace, obj)
        workspace = place_object(workspace, obj, offset)
        port_workspace = place_object_ports(port_workspace, obj_ports, offset)

    return workspace, port_workspace


def plot_grid(grid, title="Grid Plot"):
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

