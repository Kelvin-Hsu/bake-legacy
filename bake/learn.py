"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .linalg import solve_posdef
from .optimize import multi_pack, unpack, local_optimisation, multi_explore_optimisation, sample_optimisation
from .kbr import posterior_weights_quadratic

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

def posterior_nlml(mu_y, x, y, theta_x, theta_y, psi, sigma, epsil, delta):

    # Compute the dominant and inferior kernel gramices in y
    kyy, ryy = nuclear_dominant_inferior_kernel_pair(y, theta_y, psi)

    # Compute the kernel gramix in x
    dx = x/theta_x
    kxx = np.exp(-0.5 * cdist(dx, dx, 'sqeuclidean'))

    # Compute the posterior weight matrix
    v = posterior_weights_quadratic(mu_y, kxx, kyy, epsil, delta)

    # Condition the posterior weight matrix on the training data
    w = np.dot(v, kxx)

    # Compute the empirical conditional embedding at the training points
    mu_yx = np.einsum('ij,ji->i', kyy, w)

    # Compute the regularised inferior kernel gramix in y
    A = ryy + (sigma ** 2) * np.eye(y.shape[0])

    # Solve it against the conditional embedding
    b, logdetA = solve_posdef(A, mu_yx)

    # Compute the correction factor for the log marginal likelihood
    # 
    # (n x d) summed over axis 0 so now (d, )
    # sum_i = np.sum(w[:, [j]] * kyy[:, [j]] * (y - y[j, :]), axis = 0) 
    # 
    # (d, ) summed over axis 0 so now scalar
    # sum_di = np.sum((sum_i / (theta_y ** 2)) ** 2)
    # 
    # Sum up each component across j
    # log_gamma = 0.5 * np.sum([sum_di for j in np.arange(y.shape[0])])
    log_gamma = 0.5 * np.sum([np.sum((np.sum(
        w[:, [j]] * kyy[:, [j]] * (y - y[j, :]),
        axis = 0) / (theta_y ** 2)) ** 2)
        for j in np.arange(y.shape[0])])

    # Compute the negative log marginal likelihood
    return 0.5 * (np.dot(mu_yx, b) + logdetA) - log_gamma

def conditional_nlml(x, y, theta_x, theta_y, psi, sigma, zeta):

    # Compute the dominant and inferior kernel gramices in y
    kyy, ryy = nuclear_dominant_inferior_kernel_pair(y, theta_y, psi)

    # Compute the kernel gramix in x
    dx = x/theta_x
    kxx = np.exp(-0.5 * cdist(dx, dx, 'sqeuclidean'))

    # Condition the posterior weight matrix on the training data
    if zeta == 0:
        w = np.eye(x.shape[0])
    else:
        w = solve_posdef(kxx + (zeta ** 2) * np.eye(x.shape[0]), kxx)[0]

    # Compute the empirical conditional embedding at the training points
    mu_yx = np.einsum('ij,ji->i', kyy, w)

    # Compute the regularised inferior kernel gramix in y
    A = ryy + (sigma ** 2) * np.eye(y.shape[0])

    # Solve it against the conditional embedding
    b, logdetA = solve_posdef(A, mu_yx)

    # Compute the correction factor for the log marginal likelihood
    # 
    # (n x d) summed over axis 0 so now (d, )
    # sum_i = np.sum(w[:, [j]] * kyy[:, [j]] * (y - y[j, :]), axis = 0) 
    # 
    # (d, ) summed over axis 0 so now scalar
    # sum_di = np.sum((sum_i / (theta_y ** 2)) ** 2)
    # 
    # Sum up each component across j
    # log_gamma = 0.5 * np.sum([sum_di for j in np.arange(y.shape[0])])
    log_gamma = 0.5 * np.sum([np.sum((np.sum(
        w[:, [j]] * kyy[:, [j]] * (y - y[j, :]),
        axis = 0) / (theta_y ** 2)) ** 2)
        for j in np.arange(y.shape[0])])

    # Compute the negative log marginal likelihood
    return 0.5 * (np.dot(mu_yx, b) + logdetA) - log_gamma

def embedding(x, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):

    t_min_tuple = tuple([np.array(a) for a in t_min_tuple])
    t_max_tuple = tuple([np.array(a) for a in t_max_tuple])

    if t_init_tuple is None:
        t_min, t_max, t_indices = multi_pack(t_min_tuple, t_max_tuple)
    else:
        t_init_tuple = tuple([np.array(a) for a in t_init_tuple])
        t_min, t_max, t_init, t_indices = multi_pack(t_min_tuple, t_max_tuple, t_init_tuple)
        assert t_init.shape == t_min.shape

    assert t_min.shape == t_max.shape

    def objective(t):
        return joint_nlml(x, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    return unpack(t_opt, t_indices)

def posterior_embedding(mu, x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):

    t_min_tuple = tuple([np.array(a) for a in t_min_tuple])
    t_max_tuple = tuple([np.array(a) for a in t_max_tuple])

    if t_init_tuple is None:
        t_min, t_max, t_indices = multi_pack(t_min_tuple, t_max_tuple)
    else:
        t_init_tuple = tuple([np.array(a) for a in t_init_tuple])
        t_min, t_max, t_init, t_indices = multi_pack(t_min_tuple, t_max_tuple, t_init_tuple)
        assert t_init.shape == t_min.shape

    assert t_min.shape == t_max.shape

    mu_y = mu(y)
    def objective(t):
        return posterior_nlml(mu_y, x, y, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    return unpack(t_opt, t_indices)

def conditional_embedding(x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):

    t_min_tuple = tuple([np.array(a) for a in t_min_tuple])
    t_max_tuple = tuple([np.array(a) for a in t_max_tuple])

    if t_init_tuple is None:
        t_min, t_max, t_indices = multi_pack(t_min_tuple, t_max_tuple)
    else:
        t_init_tuple = tuple([np.array(a) for a in t_init_tuple])
        t_min, t_max, t_init, t_indices = multi_pack(t_min_tuple, t_max_tuple, t_init_tuple)
        assert t_init.shape == t_min.shape

    assert t_min.shape == t_max.shape

    def objective(t):
        return conditional_nlml(x, y, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = sample_optimisation(objective, t_min, t_max, n = n)
        # t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    return unpack(t_opt, t_indices)