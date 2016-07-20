"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .linalg import solve_posdef
from .optimize import multi_pack, unpack, local_optimisation, multi_explore_optimisation, sample_optimisation
# from .kbr import posterior_weights_tikhonov


def nuclear_dominant_inferior_kernel_pair(x, theta, psi):

    # Compute the normalised squared distance for the dominant kernel
    dx = x/theta
    dxx = cdist(dx, dx, 'sqeuclidean')

    # Compute the length scale of the inferior kernel
    theta_inferior = np.sqrt(0.5 * theta ** 2 + psi ** 2)

    # Compute the normalised distance for the inferior kernel
    dz = x / theta_inferior

    # Compute the normalised squared distance for the inferior kernel
    dzz = 0.25 * dxx + 0.125 * cdist(dz, -dz, 'sqeuclidean')

    # Compute the sensitivity of the inferior kernel
    d = x.shape[1]
    # s = (np.sqrt(2 * np.pi) ** d) * np.prod(theta_inferior * np.ones(d))
    s = (np.sqrt(2 * np.pi) ** d) / np.sqrt(np.prod(((2 / (theta ** 2) + 1 / (psi ** 2)) * np.ones(d))))

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


def log_normal_density(x, mu, sigma):

    b, log_det_sigma = solve_posdef(sigma, x - mu)
    const = x.shape[0] * np.log(2 * np.pi) # Can remove if needed
    return -0.5 * (np.dot(x - mu, b) + log_det_sigma + const)


def conditional_nlml(x, y, yx, dist_y, theta_x, theta_y, zeta, sigma):

    psi = np.append(theta_y, theta_x)

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

    w, _ = solve_posdef(kxx + (zeta ** 2) * identity, kxx)

    def embedding_grad(i):
        # m x 1
        return np.dot(kernel_grad(i), w[:, [i]])

    # print(dist_y.shape, kernel_grad(0).shape, embedding_grad(0).shape)

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

    mu_yx = np.einsum('ij,ji->i', kyy, w)

    return - log_normal_density(mu_yx, 0, rzz_reg) - log_gamma


# def posterior_nlml(mu_y, x, y, theta_x, theta_y, psi, sigma, epsil, delta):
#
#     # Compute the dominant and inferior kernel gramices in y
#     kyy, ryy = nuclear_dominant_inferior_kernel_pair(y, theta_y, psi)
#
#     # Compute the kernel gramix in x
#     dx = x/theta_x
#     kxx = np.exp(-0.5 * cdist(dx, dx, 'sqeuclidean'))
#
#     # Compute the posterior weight matrix
#     v = posterior_weights_tikhonov(mu_y, kxx, kyy, epsil, delta)
#
#     # Condition the posterior weight matrix on the training data
#     w = np.dot(v, kxx)
#
#     # Compute the empirical conditional embedding at the training points
#     mu_yx = np.einsum('ij,ji->i', kyy, w)
#
#     # Compute the regularised inferior kernel gramix in y
#     A = ryy + (sigma ** 2) * np.eye(y.shape[0])
#
#     # Solve it against the conditional embedding
#     b, logdetA = solve_posdef(A, mu_yx)
#
#     # Compute the correction factor for the log marginal likelihood
#     #
#     # (n x d) summed over axis 0 so now (d, )
#     # sum_i = np.sum(w[:, [j]] * kyy[:, [j]] * (y - y[j, :]), axis = 0)
#     #
#     # (d, ) summed over axis 0 so now scalar
#     # sum_di = np.sum((sum_i / (theta_y ** 2)) ** 2)
#     #
#     # Sum up each component across j
#     # log_gamma = 0.5 * np.sum([sum_di for j in np.arange(y.shape[0])])
#     log_gamma = 0.5 * np.sum([np.sum((np.sum(
#         w[:, [j]] * kyy[:, [j]] * (y - y[j, :]),
#         axis = 0) / (theta_y ** 2)) ** 2)
#         for j in np.arange(y.shape[0])])
#
#     # Compute the negative log marginal likelihood
#     return 0.5 * (np.dot(mu_yx, b) + logdetA) - log_gamma


def approx_conditional_nlml(x, y, theta_x, theta_y, zeta):
    # DOES NOT WORK
    dx = x/theta_x
    dxx = cdist(dx, dx, 'sqeuclidean')
    kxx = np.exp(-0.5 * dxx)

    identity = np.eye(x.shape[0])

    kxx_reg = kxx + (zeta ** 2) * identity

    w = solve_posdef(kxx_reg, kxx)[0]

    dy = y/theta_y
    dyy = cdist(dy, dy, 'sqeuclidean')
    kyy = np.exp(-0.5 * dyy)

    # Compute the empirical conditional embedding at the training points
    mu_yx = np.einsum('ij,ji->i', kyy, w)

    return -np.sum(np.log(mu_yx))


def latent_conditional_nlml(x, y, theta_x, theta_y, zeta_x, zeta_y, sigma):
    # DOES NOT WORK
    identity = np.eye(x.shape[0])

    dx = x/theta_x
    dxx = cdist(dx, dx, 'sqeuclidean')
    kxx = np.exp(-0.5 * dxx)

    kxx_reg = kxx + (zeta_x ** 2) * identity

    dy = y/theta_y
    dyy = cdist(dy, dy, 'sqeuclidean')
    kyy = np.exp(-0.5 * dyy)

    kyy_reg = kyy + (zeta_y ** 2) * identity

    lik = np.exp(-0.5 * cdist(y, y, 'sqeuclidean') / (sigma ** 2) / (sigma * np.sqrt(2 * np.pi)))

    w = solve_posdef(kyy_reg, np.dot(kyy, solve_posdef(kxx_reg, kxx)[0]))[0]

    # Compute the empirical conditional embedding at the training points
    evidence = np.einsum('ij,ji->i', lik, w)

    return -np.prod(evidence)


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


def conditional_embedding(x, y, yx, dist_y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):

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
        return conditional_nlml(x, y, yx, dist_y, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)
    print(t_opt, f_opt)
    return unpack(t_opt, t_indices)


# def posterior_embedding(mu, x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):
#
#     t_min_tuple = tuple([np.array(a) for a in t_min_tuple])
#     t_max_tuple = tuple([np.array(a) for a in t_max_tuple])
#
#     if t_init_tuple is None:
#         t_min, t_max, t_indices = multi_pack(t_min_tuple, t_max_tuple)
#     else:
#         t_init_tuple = tuple([np.array(a) for a in t_init_tuple])
#         t_min, t_max, t_init, t_indices = multi_pack(t_min_tuple, t_max_tuple, t_init_tuple)
#         assert t_init.shape == t_min.shape
#
#     assert t_min.shape == t_max.shape
#
#     mu_y = mu(y)
#     def objective(t):
#         return posterior_nlml(mu_y, x, y, *tuple([t[i] for i in t_indices]))
#
#     if t_init_tuple is None:
#         t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
#     else:
#         t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)
#
#     return unpack(t_opt, t_indices)
#
#
# def conditional_embedding(x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):
#
#     t_min_tuple = tuple([np.array(a) for a in t_min_tuple])
#     t_max_tuple = tuple([np.array(a) for a in t_max_tuple])
#
#     if t_init_tuple is None:
#         t_min, t_max, t_indices = multi_pack(t_min_tuple, t_max_tuple)
#     else:
#         t_init_tuple = tuple([np.array(a) for a in t_init_tuple])
#         t_min, t_max, t_init, t_indices = multi_pack(t_min_tuple, t_max_tuple, t_init_tuple)
#         assert t_init.shape == t_min.shape
#
#     assert t_min.shape == t_max.shape
#
#     def objective(t):
#         return conditional_nlml(x, y, *tuple([t[i] for i in t_indices]))
#
#     if t_init_tuple is None:
#         t_opt, f_opt = sample_optimisation(objective, t_min, t_max, n = n)
#         # t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
#     else:
#         t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)
#
#     return unpack(t_opt, t_indices)

def approx_conditional_embedding(x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):

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
        return approx_conditional_nlml(x, y, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = sample_optimisation(objective, t_min, t_max, n = n)
        # t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    return unpack(t_opt, t_indices)

def latent_conditional_embedding(x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 1):

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
        return latent_conditional_nlml(x, y, *tuple([t[i] for i in t_indices]))

    if t_init_tuple is None:
        t_opt, f_opt = sample_optimisation(objective, t_min, t_max, n = n)
        # t_opt, f_opt = multi_explore_optimisation(objective, t_min, t_max, n = n, repeat = repeat)
    else:
        t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    return unpack(t_opt, t_indices)