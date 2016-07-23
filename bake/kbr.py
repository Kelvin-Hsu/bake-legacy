"""
Kernel Bayes Rule Module.
"""
import numpy as np
from .linalg import solve_posdef as _solve_posdef
from scipy.linalg import solve as _solve
from numpy.linalg import matrix_power as _matrix_power
from numpy.linalg import pinv as _pinv


def posterior_field_tikhonov(prior_embedding_y, k_xx, k_yy, epsil, delta):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    prior_embedding_y : numpy.ndarray
        The prior kernel embedding on y evaluated at the training outputs (n,)
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n, n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n, n)
    epsil : float
        The regularisation parameter for the prior
    delta : float
        The regularisation parameter for the likelihood
    Returns
    -------
    numpy.ndarray
        The posterior field ready to be conditioned on arbitrary
        input x values and queried at arbitrary output y values (n, n)
    """
    # [Data Size] scalar
    n, = prior_embedding_y.shape

    # [Identity] (n, n)
    identity = np.eye(n)

    # [Prior Effect] (n, n)
    k_yy_reg = k_yy + (epsil ** 2) * identity
    d = np.diag(_solve_posdef(k_yy_reg, prior_embedding_y)[0])

    # [Prior Effect on Input Variables] (n, n)
    d_k_xx = np.dot(d, k_xx)

    # [Regularised Squared Prior Effect on Input Variables] (n, n)
    d_kxx_sq_reg = _matrix_power(d_k_xx, 2) + ((delta ** 2) / n) * identity

    # [Posterior Weights] (n, n)
    return np.dot(d_k_xx, _solve(d_kxx_sq_reg, d))


def posterior_field_linear(prior_embedding_y, k_xx, k_yy, epsil, delta):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    prior_embedding_y : numpy.ndarray
        The prior kernel embedding on y evaluated at the training outputs (n,)
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n, n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n, n)
    epsil : float
        The regularisation parameter for the prior
    delta : float
        The regularisation parameter for the likelihood
    Returns
    -------
    numpy.ndarray
        The posterior field ready to be conditioned on arbitrary
        input x values and queried at arbitrary output y values (n, n)
    """
    # [Data Size] scalar
    n, = prior_embedding_y.shape

    # [Identity] (n, n)
    identity = np.eye(n)

    # [Prior Effect] (n, n)
    k_yy_reg = k_yy + (epsil ** 2) * identity
    d = np.diag(_solve_posdef(k_yy_reg, prior_embedding_y)[0])

    # [Posterior Weights] (n, n)
    return _solve(np.dot(d, k_xx) + ((delta ** 2) / n) * identity, d)


def posterior_field_general(prior_embedding_y, k_xx, k_yy, epsil, delta):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    prior_embedding_y : numpy.ndarray
        The prior kernel embedding on y evaluated at the training outputs (n,)
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n, n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n, n)
    epsil : float
        The regularisation parameter for the prior
    delta : float
        The regularisation parameter for the likelihood
    Returns
    -------
    numpy.ndarray
        The posterior field ready to be conditioned on arbitrary
        input x values and queried at arbitrary output y values (n, n)
    """
    # [Data Size] scalar
    n, = prior_embedding_y.shape

    # [Identity] (n, n)
    identity = np.eye(n)

    # [Prior Effect] (n, n)
    d = np.diag(np.dot(_pinv(k_yy + (epsil ** 2) * identity), prior_embedding_y))

    # [Posterior Weights] (n, n)
    return np.dot(_pinv(np.dot(d, k_xx) + ((delta ** 2) / n) * identity), d)


posterior_fields = {'tikhonov': posterior_field_tikhonov,
                    'linear'  : posterior_field_linear,
                    'general' : posterior_field_general}