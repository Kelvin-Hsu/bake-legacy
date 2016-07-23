"""
Data Module.

Utility tools that deal with the generation and handling of data or its
corresponding inferences.
"""
import numpy as np


def generate_uniform_data(n, d, l, u, seed=None):
    """
    Generate a uniformly distributed inputs.

    Parameters
    ----------
    n : int
        Number of points to generate
    l : np.ndarray or float
        The lower bound [(d,), (1,), 1]
    u : np.ndarray or float
        The upper bound [(d,), (1,), 1]
    seed : int, optional
        The randomization seed

    Returns
    -------
    numpy.ndarray
        The uniformly distributed data (n, d)
    """
    if seed:
        np.random.seed(seed)
    return (u - l) * np.random.rand(n, d) + l


def generate_waves(x, w, p, a, b, noise_level=0.0, seed=None):
    """
    Generate functional realizations from multiple waves randomly.

    This assumes that the input is 1 dimensional.

    Parameters
    ----------
    x : numpy.ndarray
        The input samples (n, 1)
    w : numpy.ndarray
        The frequencies of the waves [(n_waves,), (1,), 1]
    p : numpy.ndarray
        The phase shift of the waves [(n_waves,), (1,), 1]
    a : numpy.ndarray
        The magnitude of the waves [(n_waves,), (1,), 1]
    b : numpy.ndarray
        The bias of the waves [(n_waves,), (1,), 1]
    noise_level : float, optional
        The noise level to be applied
    seed : int, optional
        The randomization seed

    Returns
    -------
    numpy.ndarray
        The waves output (n, 1)
    """
    if seed:
        np.random.seed(seed)

    w, p, a, b = np.array(w), np.array(p), np.array(a), np.array(b)

    y = a * np.sin(w * x + p) + b

    n_waves, = w.shape if w.ndim == 1 else 1
    n_points, _ = x.shape
    pick = np.random.randint(0, n_waves, n_points)
    y_true = y[np.arange(n_points), pick]
    y_noise = (noise_level ** 2) * np.random.randn(n_points, 1)
    return y_true + y_noise


def joint_data(x, y):
    """
    Obtain joint data.

    Parameters
    ----------
    x : numpy.ndarray
        Any form of collection of points in the x domain
    y : numpy.ndarray
        Any form of collection of points in the y domain

    Returns
    -------
    numpy.ndarray
        Joint data
    """
    return np.vstack((x.ravel(), y.ravel())).T


def generate_multiple_gaussian(n_each, d, locs, scales, seed=None):
    """
    Generate mixture data from a multiple Gaussians.

    This is different to generating samples from a Gaussian mixture.

    Parameters
    ----------
    n_each : int
        Number of samples for each Gaussian
    d : int
        The dimensionality of the samples
    locs : numpy.ndarray
        The locations of the Gaussians (m, d); m is the number of Gaussians
    scales : numpy.ndarray
        The scales of the Gaussians (m, d); m is the number of Gaussians
    seed : int, optional
        The randomization seed

    Returns
    -------
    numpy.ndarray
        The resulting samples (n_each * m, d)
    """
    if seed:
        np.random.seed(seed)

    locs, scales = np.array(locs), np.array(scales)
    assert locs.shape == scales.shape
    assert d == locs.shape[1]
    m, _= locs.shape
    standard_sample = np.random.randn(n_each * m, d)
    locs_apply = np.repeat(locs, n_each * np.ones(m).astype(int), axis=0)
    scales_apply = np.repeat(scales, n_each * np.ones(m).astype(int), axis=0)
    return scales_apply * standard_sample + locs_apply



