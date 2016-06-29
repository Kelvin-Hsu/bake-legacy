"""
Bayesian Inference for Kernel Embeddings Module.
"""
import numpy as np
from .linalg import solve_posdef

def posterior_weights(prior_embedding, k_xx, k_yy, k_xxq, epsil, delta):
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
    k_xxq : numpy.ndarray
        The gram matrix between the observed and query input variables (n x n_q)
    epsil : numpy.float64
        The regularisation parameter for the prior effect
    delta : numpy.float64
        The reguarlisation parameter for the kernel Bayes' rule
    """
    # [Data Size] n: scalar
    n = prior_embedding.shape[0]

    # [Identity] I: (n x n)
    I = np.eye(n)

    # [Prior Effect] prior_effect: (n x n)
    prior_effect = np.diag(solve_posdef(k_yy + n * epsil * I, prior_embedding))

    # [Observation Prior] obs_prior: (n x n)
    obs_prior = np.dot(prior_effect, k_xx)

    # [Inference Prior] inf_prior: (n x n_q)
    inf_prior = np.dot(prior_effect, k_xxq)

    # [Regularised Squared Observation Prior] obs_prior_sq: (n x n)
    reg_sq_obs_prior = np.linalg.matrix_power(obs_prior, 2) + delta * I

    # [Posterior Weights] (n x n_q)
    return np.dot(obs_prior, solve_posdef(reg_sq_obs_prior, inf_prior)[0])






