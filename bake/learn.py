"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .linalg import solve_posdef
from .optimize import multi_pack, local_optimisation, explore_optimisation

def nuclear_dominant_inferior_kernel_pair(x, theta, psi):

    # Compute the normalised squared distance for the dominant kernel
    dx = x/theta
    dxx = cdist(dx, dx, 'sqeuclidean')

    # Compute the dominant kernel gramix
    kxx = np.exp(-0.5 * dxx)

    # Compute the length scale of the inferior kernel
    theta_inferior = np.sqrt(0.5 * theta ** 2 + psi ** 2)

    # Compute the normalised distance for the inferior kernel
    dz = x / theta_inferior

    # Compute the normalised squared distance for the inferior kernel
    dzz = 0.25 * dxx + 0.125 * cdist(dz, -dz, 'sqeuclidean')

    # Compute the sensitivity of the inferior kernel
    d = x.shape[1]
    s = (np.sqrt(2 * np.pi) ** d) * np.prod(theta_inferior * np.ones(d))

    # Compute the dominant kernel gramix
    kxx = np.exp(-0.5 * dxx)

    # Compute the inferior kernel gramix
    rxx = s * np.exp(-dzz)

    # Return the nuclear dominant and inferior kernel pair
    return kxx, rxx

def joint_nlml(x, theta, psi, sigma):

    # Compute the dominant and inferior kernel gramices
    kxx, rxx = nuclear_dominant_inferior_kernel_pair(x, theta, psi)

    # Compute the empirical embedding evaluated at the training points
    mu = kxx.mean(axis = 0)

    # Compute the regularised inferior kernel gramix
    A = rxx + (sigma ** 2) * np.eye(x.shape[0])

    # Solve the regularised inferior kernel gramix against the embedding
    b, logdetA = solve_posdef(A, mu)

    # Compute the correction factor for the log marginal likelihood
    log_gamma = 0.5 * np.sum([np.log(np.sum((np.sum(kxx[:, [j]] * (x - x[j, :]), 
        axis = 0) / (theta ** 2)) ** 2)) for j in np.arange(x.shape[0])])

    # Compute the negative log marginal likelihood
    return 0.5 * (np.dot(mu, b) + logdetA) - log_gamma

def learn_joint_embedding(x, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000):

    t_min_tuple = tuple([np.array(a) for a in t_min_tuple])
    t_max_tuple = tuple([np.array(a) for a in t_max_tuple])

    if t_init_tuple is None:
        t_min, t_max, t_indices = multi_pack(t_min_tuple, t_max_tuple)
    else:
        t_init_tuple = tuple([np.array(a) for a in t_init_tuple])
        t_min, t_max, t_init, t_indices = multi_pack(t_min_tuple, t_max_tuple, t_init_tuple)
        assert t_init.shape == t_min.shape

    assert t_min.shape == t_max.shape
    # assert t_min_tuple[0].shape[0] == x.shape[1]

    def objective(t):
        return joint_nlml(x, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = explore_optimisation(objective, t_min, t_max, n = n)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    return tuple([t_opt[i] for i in t_indices])








