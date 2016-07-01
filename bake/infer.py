"""
Bayesian Inference for Kernel Embeddings Module.
"""
import numpy as np
from .linalg import solve_posdef
from scipy.signal import argrelextrema

def embedding(w, x, k, theta):
    return lambda xq: np.dot(k(xq, x, theta), w)

def uniform_weights(x):
    return np.ones(x.shape[0]) / x.shape[0]

def posterior_weight_matrix(prior_embedding, k_xx, k_yy, epsil, delta):
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
    prior_effect = np.diag(solve_posdef(k_yy + n * epsil * I, prior_embedding)[0])

    # [Observation Prior] obs_prior: (n x n)
    obs_prior = np.dot(prior_effect, k_xx)

    # [Regularised Squared Observation Prior] obs_prior_sq: (n x n)
    reg_sq_obs_prior = np.linalg.matrix_power(obs_prior, 2) + delta * I

    # [Posterior Weights] (n x n)
    return np.dot(obs_prior, solve_posdef(reg_sq_obs_prior, prior_effect)[0])

def posterior_embedding_core(W, k_xxq, k_yyq):
    """
    Obtain the posterior embedding involved in Kernel Bayes' Rule.

    The posterior refers to the posterior distribution of y given x.

    Parameters
    ----------
    W : numpy.ndarray
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
    return np.dot(k_yyq.T, np.dot(W, k_xxq))

def posterior_embedding(mu_prior, x, y, k_x, k_y, theta_x, theta_y, epsil, delta):

    k_xx = k_x(x, x, theta_x)
    k_yy = k_y(y, y, theta_y)

    W = posterior_weight_matrix(mu_prior(y), k_xx, k_yy, epsil, delta)

    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), np.dot(W, k_x(x, xq, theta_x)))

def kernel_bayes_average(g, W, k_ygyg, k_yyg, k_xxq):
    """
    Obtain the expectance of a function g under the posterior.

    This function is vectorised over the input variable being conditioned on.

    Parameters
    ----------
    posterior_weights : numpy.ndarray
        The posterior weights ready to be conditioned on arbitrary 
        input x values and queried at arbitrary output y values (n x n)
    k_ygyg : numpy.ndarray
        The gram matrix on the output basis for function evaluation
    k_yyg : numpy.ndarray
        The gram matrix between the observed and basis output (n x m)
    k_xxq : numpy.ndarray
        The gram matrix between the observed and query input (n x n_qx)           
    """
    # [Weights of projection of g on the RKHS on y] alpha_g: (m, )
    alpha_g = solve_posdef(k_ygyg, g)

    # [Expectance of g(Y) under the posterior] (n_qx, )
    return np.dot(alpha_g, posterior_embedding_core(W, k_xxq, k_yyg))

def regressor_mode(mu_yqxq, xq_array, yq_array):

    # Assume xq_array and yq_array are just 1D arrays for now

    n_y, n_x = mu_yqxq.shape

    assert n_x == xq_array.shape[0]
    assert n_y == yq_array.shape[0]

    x_peaks = np.array([])
    y_peaks = np.array([])

    for i in range(n_x):
        ind = argrelextrema(mu_yqxq[:, i], np.greater)
        for j in ind[0]:
            x_peaks = np.append(x_peaks, xq_array[i])
            y_peaks = np.append(y_peaks, yq_array[j])

    return x_peaks, y_peaks


# def posterior_mode(w, ky, y, y0):

#     Requires flattening...
#     def objective(yq):
#         return -2 * np.dot(ky(yq, np.array([y])).flatten(), w) + ky(yq, yq)

# def kernel_herding():


# def embedding_to_density():







