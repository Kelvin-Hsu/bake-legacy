"""
Kernel Bayes Rule Module.
"""
import numpy as np
from .linalg import solve_posdef
from scipy.linalg import solve
from numpy.linalg import matrix_power, pinv


def posterior_field_tikhonov(mu_y_prior, k_xx, k_yy, epsil, delta):
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
    epsil : float
        The regularisation parameter for the prior
    delta : float
        The regularisation parameter for the likelihood
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n) 
    """
    # [Data Size] scalar
    n, = mu_y_prior.shape

    # [Identity] (n x n)
    identity = np.eye(n)

    # [Prior Effect] (n x n)
    d = np.diag(solve_posdef(k_yy + (epsil ** 2) * identity, mu_y_prior)[0])

    # [Prior Effect on Input Variables] (n x n)
    d_kxx = np.dot(d, k_xx)

    # [Regularised Squared Prior Effect on Input Variables] (n x n)
    d_kxx_sq_reg = matrix_power(d_kxx, 2) + ((delta ** 2) / n) * identity

    # [Posterior Weights] (n x n)
    return np.dot(d_kxx, solve(d_kxx_sq_reg, d))


def posterior_field_linear(mu_y_prior, k_xx, k_yy, epsil, delta):
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
    epsil : float
        The regularisation parameter for the prior
    delta : float
        The regularisation parameter for the likelihood
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n) 
    """
    # [Data Size] scalar
    n, = mu_y_prior.shape

    # [Identity] (n x n)
    identity = np.eye(n)

    # [Prior Effect] (n x n)
    d = np.diag(solve_posdef(k_yy + (epsil ** 2) * identity, mu_y_prior)[0])

    # [Posterior Weights] (n x n)
    return solve(np.dot(d, k_xx) + ((delta ** 2) / n) * identity, d)


def posterior_field_general(mu_y_prior, k_xx, k_yy, epsil, delta):
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
    epsil : float
        The regularisation parameter for the prior
    delta : float
        The regularisation parameter for the likelihood
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary
        input x values and queried at arbitrary output y values (n x n)
    """
    # [Data Size] scalar
    n, = mu_y_prior.shape

    # [Identity] (n x n)
    identity = np.eye(n)

    # [Prior Effect] (n x n)
    d = np.diag(np.dot(pinv(k_yy + (epsil ** 2) * identity), mu_y_prior))

    # [Posterior Weights] (n x n)
    return np.dot(pinv(np.dot(d, k_xx) + ((delta ** 2) / n) * identity), d)


posterior_fields = {'tikhonov': posterior_field_tikhonov,
                    'linear': posterior_field_linear,
                    'general': posterior_field_general}