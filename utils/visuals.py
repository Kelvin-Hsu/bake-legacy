import numpy as np

def centred_uniform_query(x, y, x_margin = 0, y_margin = 0, x_query = 250, y_query = 250):

    x_lim = np.max(np.abs(x)) + x_margin
    y_lim = np.max(np.abs(y)) + y_margin
    xq_array = np.linspace(-x_lim, x_lim, x_query)
    yq_array = np.linspace(-y_lim, y_lim, y_query)
    xq_grid, yq_grid = np.meshgrid(xq_array, yq_array)

    xq = xq_array[:, np.newaxis]
    yq = yq_array[:, np.newaxis]

    return xq, yq, xq_grid, yq_grid, x_lim, y_lim