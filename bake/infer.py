"""
Bayesian Inference for Kernel Embeddings Module.
"""
import numpy as np
from .linalg import solve_posdef
from .kbr import posterior_weight_matrix
from .optimize import local_optimisation
from scipy.signal import argrelextrema

def embedding(w, x, k, theta):
    return lambda xq: np.dot(k(xq, x, theta), w)

def uniform_weights(x):
    return np.ones(x.shape[0]) / x.shape[0]

def conditional_embedding(x, y, k_x, k_y, theta_x, theta_y, zeta, k_xx = None):

    k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
    k_xx_reg = k_xx + zeta ** 2 * np.eye(x.shape[0])

    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), solve_posdef(k_xx_reg, k_x(x, xq, theta_x))[0])

def posterior_embedding(mu_prior, x, y, k_x, k_y, theta_x, theta_y, epsil, delta, k_xx = None, k_yy = None):

    k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
    k_yy = k_y(y, y, theta_y) if not k_yy else k_yy

    W = posterior_weight_matrix(mu_prior(y), k_xx, k_yy, epsil, delta)

    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), np.dot(W, k_x(x, xq, theta_x)))

def kernel_bayes_average(g, W, k_ygyg, k_yyg, k_xxq):

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

def mode(mu, xvs, xv_min, xv_max, k, theta):

    def objective(xvq):
        xq = np.array([xvq])
        uq = mu(xq)
        kqq = k(xq, xq, theta)
        return (-2 * uq + kqq)[0][0]

    x_mode, f_mode = local_optimisation(objective, xv_min, xv_max, xvs)
    print('The embedding has mode at %s with a objective value of %f' % (str(x_mode), f_mode))
    return x_mode

def multiple_modes(mu, xv_min, xv_max, k, theta, n_modes = 6):

    xv_min = np.array(xv_min)
    xv_max = np.array(xv_max)

    m = xv_min.shape[0]
    standard_range = np.random.rand(n_modes, m)
    xvs_list = (xv_max - xv_min) * standard_range + xv_min
    return np.array([mode(mu, xvs, xv_min, xv_max, k, theta) for xvs in xvs_list])


# def kernel_herding():


# def embedding_to_density():







