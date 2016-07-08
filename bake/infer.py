"""
Bayesian Inference for Kernel Embeddings Module.
"""
import numpy as np
from .kernels import gaussian
from .linalg import solve_posdef
from .kbr import posterior_weights_quadratic
from .optimize import local_optimisation

def embedding(x, theta, w = None, k = gaussian):
    w = uniform_weights(x) if w is None else w
    return lambda xq: np.dot(k(xq, x, theta), w)

def uniform_weights(x):
    return np.ones((x.shape[0], 1)) / x.shape[0]

def conditional_embedding(x, y, theta_x, theta_y, zeta = 0, k_x = gaussian, k_y = gaussian, k_xx = None):

    k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
    k_xx_reg = k_xx + zeta ** 2 * np.eye(x.shape[0])

    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), solve_posdef(k_xx_reg, k_x(x, xq, theta_x))[0])

def posterior_embedding(mu_prior, x, y, theta_x, theta_y, epsil, delta, k_x = gaussian, k_y = gaussian, k_xx = None, k_yy = None):

    k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
    k_yy = k_y(y, y, theta_y) if not k_yy else k_yy

    W = posterior_weights_quadratic(mu_prior(y), k_xx, k_yy, epsil, delta)

    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), np.dot(W, k_x(x, xq, theta_x)))

def kernel_bayes_average(g, W, k_ygyg, k_yyg, k_xxq):

    # [Weights of projection of g on the RKHS on y] alpha_g: (m, )
    alpha_g = solve_posdef(k_ygyg, g)

    # [Expectance of g(Y) under the posterior] (n_qx, )
    return np.dot(alpha_g, posterior_embedding_core(W, k_xxq, k_yyg))

def mode(mu, xv_start, xv_min, xv_max):

    # Define the objective to be minimised
    # For stationary kernels, the objective to optimise can be reduced to simply
    # the embedding
    def objective(xvq):
        return -mu(np.array([xvq]))[0][0]

    # Find the mode
    x_mode, f_mode = local_optimisation(objective, xv_min, xv_max, xv_start)

    # Size: (n_dims)
    return x_mode

def multiple_modes(mu, xv_min, xv_max, n_modes = 10):

    # Make sure these are arrays
    xv_min = np.array(xv_min)
    xv_max = np.array(xv_max)

    # Generate a list of starting points
    n_dims, = xv_min.shape
    standard_range = np.random.rand(n_modes, n_dims)
    xv_start_list = (xv_max - xv_min) * standard_range + xv_min

    # Compute the modes
    # Size: (n_modes x n_dims)
    return np.array([mode(mu, xv_start, xv_min, xv_max) for xv_start in xv_start_list])

def conditional_modes(mu_yx, xq, yv_min, yv_max, n_modes = 10):

    # This finds the modes of a given embedding
    modes_of = lambda mu: multiple_modes(mu, yv_min, yv_max, n_modes = n_modes)

    # This finds the modes at the embedding conditioned at x
    modes_at = lambda x: modes_of(lambda yq: mu_yx(yq, x))

    # This computes the modes at all query points
    # Size: (n_query x n_modes x n_dims)
    y_modes = np.array([modes_at(x) for x in xq[:, np.newaxis]])

    # This returns the corresponding input coordinates for each mode
    # Size: (n_query x n_modes x n_dims)
    x_modes = np.repeat(xq[:, np.newaxis], n_modes, axis = 1)

    # Return the modes
    # Size: (n_query x n_modes x n_dims)
    # Size: (n_query x n_modes x n_dims)
    return x_modes, y_modes


def regressor_mode(mu_yqxq, xq_array, yq_array):

    from scipy.signal import argrelextrema

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


# def kernel_herding():


# def embedding_to_density():







