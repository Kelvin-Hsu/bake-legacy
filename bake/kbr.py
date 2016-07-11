"""
Kernel Bayes Rule.
"""
import numpy as np
from .linalg import solve_posdef
from scipy.linalg import solve
from numpy.linalg import pinv


def posterior_weights_tikhonov(mu_y_prior, k_xx, k_yy, epsil, delta):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    mu_y_prior : numpy.ndarray
        The kernel embedding of the prior probability measure (n)
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n x n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n x n)
    epsil : numpy.float64
        The regularisation parameter for the prior effect
    delta : numpy.float64
        The regularisation parameter for the kernel Bayes' rule
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n) 
    """
    # [Data Size] n: scalar
    n = mu_y_prior.shape[0]

    # [Identity] I: (n x n)
    identity = np.eye(n)

    # [Prior Effect] prior_effect: (n x n)
    prior_effect = np.diag(solve_posdef(k_yy + n * epsil * identity, mu_y_prior)[0])

    # [Observation Prior] obs_prior: (n x n)
    obs_prior = np.dot(prior_effect, k_xx)

    # [Regularised Squared Observation Prior] obs_prior_sq: (n x n)
    reg_sq_obs_prior = np.linalg.matrix_power(obs_prior, 2) + delta * identity

    # [Posterior Weights] (n x n)
    return np.dot(obs_prior, solve(reg_sq_obs_prior, prior_effect))


def posterior_weights_linear(mu_y_prior, k_xx, k_yy, epsil, delta):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    mu_y_prior : numpy.ndarray
        The kernel embedding of the prior probability measure (n)
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n x n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n x n)
    epsil : numpy.float64
        The regularisation parameter for the prior effect
    delta : numpy.float64
        The regularisation parameter for the kernel Bayes' rule
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n) 
    """
    # [Data Size] n: scalar
    n, = mu_y_prior.shape

    # [Identity] I: (n x n)
    identity = np.eye(n)

    # [Prior Effect] d_y_prior: (n x n)
    d_y_prior = np.diag(solve_posdef(k_yy + epsil * identity, mu_y_prior)[0])

    # [Posterior Weights] v: (n x n)
    v = solve(np.dot(d_y_prior, k_xx) + (delta / n) * identity, d_y_prior)

    # Return the posterior weights
    return v


def posterior_weights_moore_penrose(mu_y_prior, k_xx, k_yy):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    mu_y_prior : numpy.ndarray
        The kernel embedding of the prior probability measure (n)
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n x n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n x n)
    epsil : numpy.float64
        The regularisation parameter for the prior effect
    delta : numpy.float64
        The regularisation parameter for the kernel Bayes' rule
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary
        input x values and queried at arbitrary output y values (n x n)
    """
    # [Prior Effect] d_y_prior: (n x n)
    d_y_prior = np.diag(np.dot(pinv(k_yy), mu_y_prior))

    # [Posterior Weights] v: (n x n)
    v = np.dot(pinv(np.dot(d_y_prior, k_xx)), d_y_prior)

    # Return the posterior weights
    return v
