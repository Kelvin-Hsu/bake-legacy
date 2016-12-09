"""
Acquisition Functions Module.

A collection of useful acquisition functions.
"""
import autograd.numpy as np
from scipy.stats import norm


def gaussian_expected_improvement(mu, std, best):
    """
    Expected Improvement Acquisition Function for a Gaussian process.

    Parameters
    ----------
    mu : numpy.ndarray
        The mean of the predictions (n_q,)
    std : numpy.ndarray
        The standard deviations of the predictions (n_q,)
    best : float
        The maximum observed value so far

    Returns
    -------
    numpy.ndarray
        The acquisition function evaluated at the corresponding points (n_q,)
    """
    diff = mu - best
    abs_diff = np.abs(diff)
    clip_diff = np.clip(diff, 0, np.inf)
    return clip_diff + std*norm.pdf(diff/std) - abs_diff*norm.cdf(-abs_diff/std)