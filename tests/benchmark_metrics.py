"""
Benchmark Metrics Module.

Standard metrics for benchmarking or testing algorithmic performance.
"""
import autograd.numpy as np
from scipy.spatial.distance import cdist


def loss_opt_loc(x=None, function=None, dist_ratio=0.01):
    """
    Compute loss metric for how close a point is to the test function optimum.

    Parameters
    ----------
    x : numpy.ndarray
        A set of points of size (n, n_dim) to consider, usually with n = 1
    function : callable
        The test function
    dist_ratio : float
        The ratio w.r.t. the test domain at which we consider a success

    Returns
    -------
    float
        The loss value
    """
    dists = cdist(x, function.x_opt, 'euclidean')
    success_radius = dist_ratio * np.min(function.x_max - function.x_min)
    return np.min(dists/success_radius)


def success_opt_loc(loss_opt_loc_value):
    """
    Determine if the loss value indicate a success for 'loss_opt_loc'.

    Parameters
    ----------
    loss_opt_loc_value : float
        The loss value

    Returns
    -------
    bool
        Success of failure
    """
    return loss_opt_loc_value < 1


def loss_func_approx(x_q, f_q, function=None):
    """
    Compute loss metric for function approximation (RMSE).

    Parameters
    ----------
    x_q : numpy.ndarray
        The query points for evaluation
    f_q : numpy.ndarray
        The predictions at those query points from some model
    function : callable
        The function in question

    Returns
    -------
    float
        The loss value (RMSE)
    """
    f_q_true = function(x_q)
    return np.sqrt(np.mean((f_q - f_q_true)**2))


def success_func_approx(loss_func_approx_value, rmse_bound=None):
    """
    Determine if the loss value indicate a success for 'loss_func_approx'.

    Parameters
    ----------
    loss_func_approx_value : float
        The loss value
    rmse_bound : float
        The bound for the loss value (RMSE) to determine success

    Returns
    -------
    bool
        Success or failure
    """
    return loss_func_approx_value < rmse_bound