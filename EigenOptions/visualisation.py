import numpy as np
import math
import matplotlib.pyplot as plt


def plot_clusters(clusters, grid):
    """
    transform data into the shape of the gridworld (currently must be square)
    """
    clusters_with_walls = add_in_walls(clusters, grid)  # add in walls
    original_dim = grid.shape  # get dims to turn back into square grid
    full_grid = clusters_with_walls.reshape(original_dim)  # get dims to turn back into square grid
    plot_as_heat(full_grid)


def add_in_walls(not_walls, grid):
    """
    Add the walls back in to the grid so the clusters can be visualised in context
    """
    wall_states = np.isin(grid, 1)
    # flatten wall states matrix and matrix to add walls to
    flat_matrix = np.array(not_walls).flatten()
    wall_states_flat = wall_states.flatten()

    # add in walls to flattened matrix
    flat_with_walls = np.array(flat_matrix, dtype=float)
    for i, b in enumerate(wall_states_flat):
        if b:
            flat_with_walls = np.insert(flat_with_walls, i, np.nan)
    return flat_with_walls


def plot_as_heat(grid_clusters):
    """
    Plot the gridworld with clusters as a heatmap. Walls will be identified on the heatmap
    :param grid_clusters: gridworld with walls as nan, and clusters as real numbers
    """
    grid_clusters = np.nan_to_num(grid_clusters, nan=-1)
    plt.imshow(grid_clusters, cmap='hot', interpolation='nearest')
    plt.show()


def plot_eigenvector(vector, grid):
    full_grid = add_in_walls(vector, grid)
    original_dim = grid.shape  # get dims to turn back into square grid
    full_grid_sq = full_grid.reshape(original_dim)
    plot_as_heat(full_grid_sq)
