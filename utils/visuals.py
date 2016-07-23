"""
Visualization Module.
"""
import numpy as np


def uniform_query(x_lim, y_lim, n_x_query=250, n_y_query=250):
    """
    Generate uniform grid query points to visualize inference.

    This assumes that both the x and y are one dimensional.

    Parameters
    ----------
    x_lim : tuple
        The lower and upper bound in x
    y_lim : tuple
        The lower and upper bound in y
    n_x_query : int
        The number of query points in x
    n_y_query : int
        The number of query points in y

    Returns
    -------
    numpy.ndarray
        The query points for x (n_x_query, 1)
    numpy.ndarray
        The query points for y (n_y_query, 1)
    numpy.ndarray
        The mesh-grid for x (n_y_query, n_x_query)
    numpy.ndarray
        The mesh-grid for y (n_y_query, n_x_query)
    """
    xq_array = np.linspace(x_lim[0], x_lim[1], n_x_query)
    yq_array = np.linspace(y_lim[0], y_lim[1], n_y_query)
    xq_grid, yq_grid = np.meshgrid(xq_array, yq_array)
    xq = xq_array[:, np.newaxis]
    yq = yq_array[:, np.newaxis]
    return xq, yq, xq_grid, yq_grid


def find_bounded_extrema(c_grid, z_grid, z_lim):
    """
    Find the extrema of a mesh-grid on a limited area defined by another mesh.

    Parameters
    ----------
    c_grid : numpy.ndarray
        The mesh-grid to find the extrema for (n_y_query, n_x_query)
    z_grid : numpy.ndarray
        The mesh-grid to define the area of search (n_y_query, n_x_query)
    z_lim : tuple
        The limiting bounds on the mesh-grid defining the area of search

    Returns
    -------
    float
        The minimum value of the bounded mesh-grid
    float
        The maximum value of the bounded mesh-grid
    """
    z_min = z_lim[0]
    z_max = z_lim[1]

    c_grid_cut = c_grid[np.logical_and(z_grid < z_max, z_grid > z_min)]

    return np.min(c_grid_cut), np.max(c_grid_cut)



