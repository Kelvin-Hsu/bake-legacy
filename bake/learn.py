"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from scipy.spatial.distance import cdist as _cdist
from .linalg import dist as _dist
from .linalg import solve_posdef as _solve_posdef
from .linalg import log_gaussian_density as _log_gaussian_density
from .optimize import hyper_opt as _hyper_opt


def nuclear_dominant_inferior_kernel_pair(x, theta, psi):
    """
    Compute the nuclear dominant and inferior gram matrices.

    This implementation assumes a Gaussian kernel and a
    Gaussian nuclear measure.

    Parameters
    ----------
    x : numpy.ndarray
        The collected data (n, d)
    theta : numpy.ndarray
        Hyperparameters of the dominant Gaussian kernel k [(d,), (1,), 1]
    psi : numpy.ndarray
        Hyperparameters of the nuclear Gaussian measure nu [(d,), (1,), 1]

    Returns
    -------
    numpy.ndarray
        The nuclear dominant gram matrix k (n, n)
    numpy.ndarray
        The nuclear inferior gram matrix r (n, n)
    """
    # Compute the normalised squared distance for the dominant kernel
    d_x = x/theta
    d_xx = _cdist(d_x, d_x, 'sqeuclidean')

    # Compute the length scale of the inferior kernel
    half_theta_sq = 0.5 * theta ** 2
    psi_sq = psi ** 2
    theta_inferior = np.sqrt(half_theta_sq + psi_sq)

    # Compute the normalised distance for the inferior kernel
    d_z = x / theta_inferior

    # Compute the normalised squared distance for the inferior kernel
    d_zz = 0.25 * d_xx + 0.125 * _cdist(d_z, -d_z, 'sqeuclidean')

    # Compute the sensitivity of the inferior kernel
    d = x.shape[1]
    det_theta_inf = np.prod(((1 / half_theta_sq + 1 / psi_sq) * np.ones(d)))
    s = np.sqrt(((2 * np.pi) ** d) * det_theta_inf)

    # Compute the dominant kernel gramix
    k_xx = np.exp(-0.5 * d_xx)

    # Compute the inferior kernel gramix
    r_xx = s * np.exp(-d_zz)

    # Return the nuclear dominant and inferior kernel pair
    return k_xx, r_xx


def joint_nlml(theta, psi, sigma, data):
    """
    Compute the negative log marginal likelihood for a joint embedding.

    This implementation assumes a Gaussian kernel and a
    Gaussian nuclear measure.

    Parameters
    ----------
    theta : numpy.ndarray
        Hyperparameters of the Gaussian kernel k [(d,), (1,), 1]
    psi : numpy.ndarray
        Hyperparameters of the nuclear Gaussian measure nu [(d,), (1,), 1]
    sigma : numpy.ndarray
        Noise level in the embedding likelihood model [(n,), (1,), 1]
        Use an array of n values to represent heteroscedasticity, or a single
        value to represent homoscedasticity
    data : tuple
        Cached data that does not vary with the hyperparameters
        Here the cache is (x,) where x is the collected data of size (n, d)

    Returns
    -------
    float
        The negative log marginal likelihood for a joint embedding.
    """
    # Obtain the data
    x, = data

    # Compute the dominant and inferior kernel gramices
    k_xx, r_xx = nuclear_dominant_inferior_kernel_pair(x, theta, psi)

    # Compute the empirical embedding evaluated at the training points
    mu = k_xx.mean(axis = 0)

    # Compute the regularised inferior kernel gramix
    n, _ = x.shape
    r_xx_reg = r_xx + np.diag(sigma ** 2) * np.eye(n)

    # Compute the final log correction factor for the log marginal likelihood
    log_gamma = 0.5 * np.sum([np.log(np.sum((np.sum(k_xx[:, [j]] * (x - x[j, :]),
        axis = 0) / (theta ** 2)) ** 2)) for j in np.arange(n)])

    # Compute the negative log marginal likelihood
    return - _log_gaussian_density(mu, 0, r_xx_reg) - log_gamma


def conditional_nlml(theta_x, theta_y, zeta, psi, sigma, data):
    """
    Compute the negative log marginal likelihood for a conditional embedding.

    Parameters
    ----------
    theta_x : numpy.ndarray
        Hyperparameters of the Gaussian kernel k on x [(d_x,), (1,), 1]
    theta_y : numpy.ndarray
        Hyperparameters of the Gaussian kernel k on y [(d_y,), (1,), 1]
    zeta : float
        The regularization parameter for the conditional embedding
    psi : numpy.ndarray
        Hyperparameters of the nuclear Gaussian measure nu
        [(d_y + d_x,), (1,), 1]
    sigma : numpy.ndarray
        Noise level in the embedding likelihood model [(n,), (1,), 1]
        Use an array of n values to represent heteroscedasticity, or a single
        value to represent homoscedasticity
    data : tuple
        Cached data that does not vary with the hyperparameters
        Here the cache is (x, y, yx, dist_y) where:
        x is the collected data of size (n, d_x)
        y is the collected data of size (n, d_y)
        yx is the joint collected data of size (n, d_y + d_x)
        dist_y is the collection of all paired differences in y
        dist_y = bake.kernels.dist(y, y)

    Returns
    -------
    float
        The negative log marginal likelihood for a conditional embedding.
    """
    # Obtain the data and cache
    x, y, yx, dist_y = data

    # Compute the identity
    n = yx.shape[0]
    identity = np.eye(n)

    # Compute the gramix in x
    d_x = x / theta_x
    d_xx = _cdist(d_x, d_x, 'sqeuclidean')
    k_xx = np.exp(-0.5 * d_xx)

    # Compute teh gramix in y
    d_y = y / theta_y
    d_yy = _cdist(d_y, d_y, 'sqeuclidean')
    k_yy = np.exp(-0.5 * d_yy)

    # Define the kernel gradient for the i-th data point (d, n)
    def kernel_grad(i):
        return (- dist_y[i] / (theta_y ** 2)).T * k_yy[i, :]

    # Compute the conditional embedding weights
    w, _ = _solve_posdef(k_xx + (zeta ** 2) * identity, k_xx)

    # Define the conditional embedding gradient for the i-th data point (d, 1)
    def embedding_grad(i):
        return np.dot(kernel_grad(i), w[:, [i]])

    # Compute the embedding gradients over all data points (d, n)
    embedding_grads = np.array([embedding_grad(i).T[0] for i in np.arange(n)]).T

    # Sum over the component dimension of the squared embedding gradients (n,)
    sum_sq_embedding_grads = np.sum(embedding_grads ** 2, axis = 0)

    # Compute the final log correction factor for the log marginal likelihood
    log_gamma =  0.5 * np.sum(np.log(sum_sq_embedding_grads))

    # Compute the inferior kernel of the joint embedding
    _, r_zz = nuclear_dominant_inferior_kernel_pair(yx,
                                                    np.append(theta_y, theta_x),
                                                    psi)

    # Regularize the inferior kernel
    r_zz_reg = r_zz + (sigma ** 2) * identity

    # Compute the joint embedding at the data points
    mu_yx = np.einsum('ij,ji->i', k_yy, w)

    # Compute the negative log marginal likelihood
    return - _log_gaussian_density(mu_yx, 0, r_zz_reg) - log_gamma


def optimal_joint_embedding(x, hyper_min, hyper_max, **kwargs):
    """
    Learn the optimal hyperparameters for a joint embedding.

    Parameters
    ----------
    x : numpy.ndarray
        The joint samples (n, d)
    hyper_min : tuple
        A tuple of arrays or lists representing the hyper lower bound
    hyper_max : tuple
        A tuple of arrays or lists representing the hyper upper bound
    hyper_init : tuple, optional
        A tuple of arrays or lists representing the initial hyperparameters
    n_samples : int, optional
        The number of sample points to use if sample optimization is involved
    n_repeat : int, optional
        The number of multiple explore optimisation to be performed if involved

    Returns
    -------
    tuple
        A tuple of arrays representing the optimal hyperparameters
    """
    return _hyper_opt(joint_nlml, (x,), hyper_min, hyper_max, **kwargs)


def optimal_conditional_embedding(x, y, hyper_min, hyper_max, **kwargs):
    """
    Learn the optimal hyperparameters for a conditional embedding.

    Parameters
    ----------
    x : numpy.ndarray
        The input samples (n, d_x)
    y : numpy.ndarray
        The output samples (n, d_y)
    hyper_min : tuple
        A tuple of arrays or lists representing the hyper lower bound
    hyper_max : tuple
        A tuple of arrays or lists representing the hyper upper bound
    hyper_init : tuple, optional
        A tuple of arrays or lists representing the initial hyperparameters
    n_samples : int, optional
        The number of sample points to use if sample optimization is involved
    n_repeat : int, optional
        The number of multiple explore optimisation to be performed if involved

    Returns
    -------
    tuple
        A tuple of arrays representing the optimal hyperparameters
    """
    yx = np.vstack((y.ravel(), x.ravel())).T
    dist_y = _dist(y, y)
    return _hyper_opt(conditional_nlml, (x, y, yx, dist_y),
                      hyper_min, hyper_max, **kwargs)