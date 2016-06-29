"""
Bayesian Inference for Kernel Embeddings Module.
"""
import numpy as np
from .linalg import solve_posdef

def posterior_weights(prior_embedding, k_xx, k_yy, epsil, delta):
    """
    Obtain the posterior weights involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    prior_embedding : numpy.ndarray
        The kernel embedding of the prior probability measure
    k_xx : numpy.ndarray
        The gram matrix on the observed input variables (n x n)
    k_yy : numpy.ndarray
        The gram matrix on the observed output variables (n x n)
    epsil : numpy.float64
        The regularisation parameter for the prior effect
    delta : numpy.float64
        The reguarlisation parameter for the kernel Bayes' rule
    Returns
    -------
    numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n) 
    """
    # [Data Size] n: scalar
    n = prior_embedding.shape[0]

    # [Identity] I: (n x n)
    I = np.eye(n)

    # [Prior Effect] prior_effect: (n x n)
    prior_effect = np.diag(solve_posdef(k_yy + n * epsil * I, prior_embedding))

    # [Observation Prior] obs_prior: (n x n)
    obs_prior = np.dot(prior_effect, k_xx)

    # [Regularised Squared Observation Prior] obs_prior_sq: (n x n)
    reg_sq_obs_prior = np.linalg.matrix_power(obs_prior, 2) + delta * I

    # [Posterior Weights] (n x n)
    return np.dot(obs_prior, solve_posdef(reg_sq_obs_prior, prior_effect)[0])

def posterior_embedding(posterior_weights, k_xxq, k_yyq):
    """
    Obtain the posterior embedding involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    posterior_weights : numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n)
    k_xxq : numpy.ndarray
        The gram matrix between the observed and query input (n x n_qx)
    k_yyq : numpy.ndarray
        The gram matrix between the observed and query output (n x n_qy)
    Returns
    -------
    numpy.ndarray
        The posterior embeddings conditioned on each x and evaluated at each y
        (n_qy x n_qx)
    """
    # [Posterior Embedding] (n_qy x n_qx)
    return np.dot(k_yyq.T, np.dot(posterior_weights, k_xxq))






