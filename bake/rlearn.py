"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from .linalg import solve_posdef
from .infer import posterior_weight_matrix
from .learn import gp_kernel
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def nlml_log(mu_y_prior_vector, x, y, log_theta_x, log_theta_y, log_epsil, log_delta, log_var, log_alpha):

    theta_x = np.exp(log_theta_x)
    theta_y = np.exp(log_theta_y)
    epsil = np.exp(log_epsil)
    delta = np.exp(log_delta)
    var = np.exp(log_var)
    alpha = np.exp(log_alpha)
    return nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var, alpha)


def nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var, alpha):

    # Compute the squared euclidean distance across the training data
    dx = x/theta_x
    dxx = cdist(dx, dx, 'sqeuclidean')

    # Compute the gram matrix with the rkhs kernel
    kxx = np.exp(-0.5 * dxx)

    # Compute the squared euclidean distance across the training data
    dy = y/theta_y
    dyy = cdist(dy, dy, 'sqeuclidean')

    # Compute the gram matrix with the rkhs kernel
    kyy = np.exp(-0.5 * dyy)

    # Joint posterior weight matrix
    v = posterior_weight_matrix(mu_y_prior_vector, kxx, kyy, epsil, delta)

    # Conditioned posterior weight matrix
    w = np.dot(v, kxx)

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

    # Compute the conditional embedding at the training points
    mu_yx = np.einsum('ij,ji->i', kyy, w)

    # Compute the gram matrix with the gp kernel
    rxx = gp_kernel(y, theta_y, alpha, dyy)

    # This is the regularised gram matrix with the gp kernel
    A = rxx + var * np.eye(y.shape[0])

    # Solve the regularised gp kernel matrix against the conditional embedding
    b, logdetA = solve_posdef(A, mu_yx)

    # Compute the negative log marginal likelihood
    return 0.5 * (np.dot(mu_yx, b) + logdetA) - log_gamma

def hyperparameters(mu_y_prior, x, y, theta_x_init, theta_y_init, epsil_init, delta_init, var_init = None, alpha_init = None, var_fix = 1.0, alpha_fix = 1.0):

    options = {'disp': True, 'maxiter': 150000, 'ftol': 1e-12}
    method = 'COBYLA'

    m_x = theta_x_init.shape[0]
    m_y = theta_y_init.shape[0]
    m_total = m_x + m_y + 2

    ind_total = np.arange(m_total)
    ind_x = np.arange(m_x)
    ind_y = np.arange(m_x, m_x + m_y)
    ind_epsil = m_x + m_y
    ind_delta = ind_epsil + 1

    mu_y_prior_vector = mu_y_prior(y)

    t_init = np.append(np.append(np.append(theta_x_init, theta_y_init), epsil_init), delta_init)

    if var_init is None and alpha_init is None:

        var_init = var_fix
        alpha_init = alpha_fix

        def objective(t):
            theta_x = t[ind_x]
            theta_y = t[ind_y]
            epsil = t[ind_epsil]
            delta = t[ind_delta]
            return nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var_init, alpha_init)

        constraints = tuple([{'type': 'ineq', 'fun': lambda t: t[i] - 1e-8} for i in range(t_init.shape[0])])
        optimal_result = minimize(objective, t_init, options = options, method = method, constraints = constraints)

        theta_x_opt = optimal_result.x[ind_x]
        theta_y_opt = optimal_result.x[ind_y]
        epsil_opt = optimal_result.x[ind_epsil]
        delta_opt = optimal_result.x[ind_delta]
        return theta_x_opt, theta_y_opt, epsil_opt, delta_opt, var_init, alpha_init

    elif alpha_init is None:

        alpha_init = alpha_fix
        t_init = np.append(t_init, var_init)

        def objective(t):
            theta_x = t[ind_x]
            theta_y = t[ind_y]
            epsil = t[ind_epsil]
            delta = t[ind_delta]
            var = t[-1]
            return nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var, alpha_init)

        constraints = tuple([{'type': 'ineq', 'fun': lambda t: t[i] - 1e-8} for i in range(t_init.shape[0])])
        optimal_result = minimize(objective, t_init, options = options, method = method, constraints = constraints)

        theta_x_opt = optimal_result.x[ind_x]
        theta_y_opt = optimal_result.x[ind_y]
        epsil_opt = optimal_result.x[ind_epsil]
        delta_opt = optimal_result.x[ind_delta]
        var_opt = optimal_result.x[-1]
        return theta_x_opt, theta_y_opt, epsil_opt, delta_opt, var_opt, alpha_init

    elif var_init is None:

        var_init = var_fix
        t_init = np.append(t_init, alpha_init)

        def objective(t):
            theta_x = t[ind_x]
            theta_y = t[ind_y]
            epsil = t[ind_epsil]
            delta = t[ind_delta]
            alpha = t[-1]
            return nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var_init, alpha)

        constraints = tuple([{'type': 'ineq', 'fun': lambda t: t[i] - 1e-8} for i in range(t_init.shape[0])])
        optimal_result = minimize(objective, t_init, options = options, method = method, constraints = constraints)

        theta_x_opt = optimal_result.x[ind_x]
        theta_y_opt = optimal_result.x[ind_y]
        epsil_opt = optimal_result.x[ind_epsil]
        delta_opt = optimal_result.x[ind_delta]
        alpha_opt = optimal_result.x[-1]
        return theta_x_opt, theta_y_opt, epsil_opt, delta_opt, var_init, alpha_opt

    else:

        t_init = np.append(np.append(t_init, var_init), alpha_init)

        def objective(t):
            theta_x = t[ind_x]
            theta_y = t[ind_y]
            epsil = t[ind_epsil]
            delta = t[ind_delta]
            var = t[-2]
            alpha = t[-1]
            return nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var, alpha)

        constraints = tuple([{'type': 'ineq', 'fun': lambda t: t[i] - 1e-8} for i in range(t_init.shape[0])])
        optimal_result = minimize(objective, t_init, options = options, method = method, constraints = constraints)

        theta_x_opt = optimal_result.x[ind_x]
        theta_y_opt = optimal_result.x[ind_y]
        epsil_opt = optimal_result.x[ind_epsil]
        delta_opt = optimal_result.x[ind_delta]
        var_opt = optimal_result.x[-2]
        alpha_opt = optimal_result.x[-1]

        return theta_x_opt, theta_y_opt, epsil_opt, delta_opt, var_opt, alpha_opt

def log_hyperparameters(mu_y_prior, x, y, log_theta_x_init, log_theta_y_init, log_epsil_init, log_delta_init, log_var_init = None, log_alpha_init = None, log_var_fix = 0.0, log_alpha_fix = 0.0):

    options = {'disp': True, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000}
    method = 'L-BFGS-B'

    m_x = log_theta_x_init.shape[0]
    m_y = log_theta_y_init.shape[0]
    m_total = m_x + m_y + 2

    ind_total = np.arange(m_total)
    ind_x = np.arange(m_x)
    ind_y = np.arange(m_x, m_x + m_y)
    ind_epsil = m_x + m_y
    ind_delta = ind_epsil + 1

    mu_y_prior_vector = mu_y_prior(y)

    t_init = np.append(np.append(np.append(log_theta_x_init, log_theta_y_init), log_epsil_init), log_delta_init)

    if log_var_init is None and log_alpha_init is None:

        log_var_init = log_var_fix
        log_alpha_init = log_alpha_fix

        def objective(t):
            log_theta_x = t[ind_x]
            log_theta_y = t[ind_y]
            log_epsil = t[ind_epsil]
            log_delta = t[ind_delta]
            return nlml_log(mu_y_prior_vector, x, y, log_theta_x, log_theta_y, log_epsil, log_delta, log_var_init, log_alpha_init)

        optimal_result = minimize(objective, t_init, options = options, method = method)

        theta_x_opt = np.exp(optimal_result.x[ind_x])
        theta_y_opt = np.exp(optimal_result.x[ind_y])
        epsil_opt = np.exp(optimal_result.x[ind_epsil])
        delta_opt = np.exp(optimal_result.x[ind_delta])

        return theta_x_opt, theta_y_opt, epsil_opt, delta_opt, np.exp(log_var_init), np.exp(log_alpha_init)

def optimal_hyperparameters(mu_y_prior, x, y, n = 20000):

    mu_y_prior_vector = mu_y_prior(y)

    m_x = x.shape[1]
    m_y = y.shape[1]
    m_total = m_x + m_y + 4

    ind_total = np.arange(m_total)
    ind_x = np.arange(m_x)
    ind_y = np.arange(m_x, m_x + m_y)
    ind_epsil = m_x + m_y
    ind_delta = ind_epsil + 1
    ind_var = ind_delta + 1
    ind_alpha = ind_var + 1

    def objective(t):
        theta_x = t[ind_x]
        theta_y = t[ind_y]
        epsil = t[ind_epsil]
        delta = t[ind_delta]
        var = t[ind_var]
        alpha = t[ind_alpha]
        return nlml(mu_y_prior_vector, x, y, theta_x, theta_y, epsil, delta, var, alpha)

    t_opt, f_opt = brute_force_minimise(objective, m_total, n = n)
    print(f_opt)

    theta_x_opt = t_opt[ind_x]
    theta_y_opt = t_opt[ind_y]
    epsil_opt = t_opt[ind_epsil]
    delta_opt = t_opt[ind_delta]
    var_opt = t_opt[ind_var]
    alpha_opt = t_opt[ind_alpha]       
    return theta_x_opt, theta_y_opt, epsil_opt, delta_opt, var_opt, alpha_opt

def brute_force_minimise(objective, m, n = 20000):

    t_values = np.abs(np.random.rand(n, m)) + 1e-3

    t_opt = 0.5*np.abs(np.random.rand(m)) + 1e-3
    f_opt = objective(t_opt)

    i = 0

    for t in t_values:

        f = objective(t)

        if f < f_opt:
            t_opt = t
            f_opt = f
            print(i, ' : ', t_opt, f_opt)
        i += 1

    return t_opt, f_opt








