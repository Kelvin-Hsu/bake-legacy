"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .kernels import dist as _dist
from .linalg import solve_posdef as _solve_posdef
from .linalg import log_gaussian_density as _log_gaussian_density
from .optimize import hyper_opt as _hyper_opt


def nuclear_dominant_inferior_kernel_pair(x, theta, psi):

    # Compute the normalised squared distance for the dominant kernel
    dx = x/theta
    dxx = cdist(dx, dx, 'sqeuclidean')

    # Compute the length scale of the inferior kernel
    half_theta_sq = 0.5 * theta ** 2
    psi_sq = psi ** 2
    theta_inferior = np.sqrt(half_theta_sq + psi_sq)

    # Compute the normalised distance for the inferior kernel
    dz = x / theta_inferior

    # Compute the normalised squared distance for the inferior kernel
    dzz = 0.25 * dxx + 0.125 * cdist(dz, -dz, 'sqeuclidean')

    # Compute the sensitivity of the inferior kernel
    d = x.shape[1]
    det_theta_inf = np.prod(((1 / half_theta_sq + 1 / psi_sq) * np.ones(d)))
    s = np.sqrt(((2 * np.pi) ** d) * det_theta_inf)

    # Compute the dominant kernel gramix
    kxx = np.exp(-0.5 * dxx)

    # Compute the inferior kernel gramix
    rxx = s * np.exp(-dzz)

    # Return the nuclear dominant and inferior kernel pair
    return kxx, rxx


def joint_nlml(theta, psi, sigma, data):

    x, = data

    # Compute the dominant and inferior kernel gramices
    kxx, rxx = nuclear_dominant_inferior_kernel_pair(x, theta, psi)

    # Compute the empirical embedding evaluated at the training points
    mu = kxx.mean(axis = 0)

    # Compute the regularised inferior kernel gramix
    rxx_reg = rxx + (sigma ** 2) * np.eye(x.shape[0])

    # Compute the correction factor for the log marginal likelihood
    log_gamma = 0.5 * np.sum([np.log(np.sum((np.sum(kxx[:, [j]] * (x - x[j, :]), 
        axis = 0) / (theta ** 2)) ** 2)) for j in np.arange(x.shape[0])])

    # Compute the negative log marginal likelihood
    return - _log_gaussian_density(mu, 0, rxx_reg) - log_gamma


def conditional_nlml(theta_x, theta_y, zeta, psi, sigma, data):

    x, y, yx, dist_y = data

    n = yx.shape[0]
    identity = np.eye(n)

    dx = x / theta_x
    dxx = cdist(dx, dx, 'sqeuclidean')
    kxx = np.exp(-0.5 * dxx)

    dy = y / theta_y
    dyy = cdist(dy, dy, 'sqeuclidean')
    kyy = np.exp(-0.5 * dyy)

    def kernel_grad(i):
        # m x n
        return (- dist_y[i] / (theta_y ** 2)).T * kyy[i, :]

    w, _ = _solve_posdef(kxx + (zeta ** 2) * identity, kxx)

    def embedding_grad(i):
        # m x 1
        return np.dot(kernel_grad(i), w[:, [i]])

    # m x n
    embedding_grads = np.array([embedding_grad(i).T[0] for i in np.arange(n)]).T

    # n
    sum_sq_embedding_grads = np.sum(embedding_grads ** 2, axis = 0)

    # log gamma term!
    log_gamma =  0.5 * np.sum(np.log(sum_sq_embedding_grads))

    _, rzz = nuclear_dominant_inferior_kernel_pair(yx,
                                                   np.append(theta_y, theta_x),
                                                   psi)
    rzz_reg = rzz + np.diag(sigma ** 2) * identity

    mu_yx = np.dot(kyy, w).diagonal() # np.einsum('ij,ji->i', kyy, w)

    return - _log_gaussian_density(mu_yx, 0, rzz_reg) - log_gamma


def embedding(x, hyper_min, hyper_max, **kwargs):

    return _hyper_opt(joint_nlml, (x,), hyper_min, hyper_max, **kwargs)


def conditional_embedding(x, y, hyper_min, hyper_max, **kwargs):

    yx = np.vstack((x.ravel(), y.ravel())).T
    dist_y = _dist(y, y)
    return _hyper_opt(conditional_nlml, (x, y, yx, dist_y),
                      hyper_min, hyper_max, **kwargs)