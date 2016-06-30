"""
Bayesian Learning for Kernel Embeddings Module.
"""
import numpy as np
from .linalg import solve_posdef
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def nlml_log(x, log_theta, log_var, log_alpha):

    # Convert the hyperparameters back to its original scale
    theta = np.exp(log_theta)
    var = np.exp(log_var)
    alpha = np.exp(log_alpha)

    # Compute the negative log marginal likelihood
    return nlml(x, theta, var, alpha)

def nlml(x, theta, var, alpha):

    # Compute the squared euclidean distance across the training data
    dxx = cdist(x/theta, x/theta, 'sqeuclidean')

    # Compute the gram matrix with the rkhs kernel
    kxx = np.exp(-0.5 * dxx)

    # Compute the mean embedding at the training points
    mu = kxx.mean(axis = 0)

    # Compute the gram matrix with the gp kernel
    rxx = gp_kernel(x, theta, alpha, dxx)

    # This is the regularised gram matrix with the gp kernel
    A = rxx + var * np.eye(x.shape[0])

    # Solve the regularised gp kernel matrix against the mean embedding
    b, logdetA = solve_posdef(A, mu)

    # Compute the correction factor for the log marginal likelihood
    log_gamma = 0.5 * np.sum([np.log(np.sum((np.sum(kxx[:, [j]] * (x - x[j, :]), 
        axis = 0) / (theta ** 2)) ** 2)) for j in np.arange(x.shape[0])])

    # Compute the negative log marginal likelihood
    return 0.5 * (np.dot(mu, b) + logdetA) - log_gamma

def gp_kernel(x, theta, alpha, dxx):

    # This is the length scale hyperparameters of the gp
    theta_gp = np.sqrt(0.5 * theta ** 2 + alpha ** 2)

    # This is the normalised feature vector for the gp
    z_gp = x / theta_gp

    # This is the sensitivity of the gp kernel
    s = (np.sqrt(2 * np.pi) ** theta.shape[0]) * np.prod(theta_gp)

    # This is the normalised squared distances of the gp features
    dzz = 0.25 * dxx + 0.125 * cdist(z_gp, -z_gp, 'sqeuclidean')

    # This is the resulting gram matrix
    return s * np.exp(-dzz)

def hyperparameters(x, theta_init, var_init = None, alpha_init = None, var_fix = 1.0, alpha_fix = 1.0):

    options = {'disp': True, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000}
    method = 'L-BFGS-B'

    if var_init is None and alpha_init is None:

        var_init = var_fix
        alpha_init = alpha_fix

        def objective(t):
            return nlml(x, t, var_init, alpha_init)

        optimal_result = minimize(objective, theta_init, method = method, options = options)
        return optimal_result.x, var_init, alpha_init

    elif alpha_init is None:

        alpha_init = alpha_fix
        t_init = np.append(theta_init, var_init)

        def objective(t):
            theta = t[:-1]
            var = t[-1]
            return nlml(x, theta, var, alpha_init)

        optimal_result = minimize(objective, t_init, method = method, options = options)

        return optimal_result.x[:-1], optimal_result.x[-1], alpha_init

    elif var_init is None:

        var_init = var_fix
        t_init = np.append(theta_init, alpha_init)

        def objective(t):
            theta = t[:-1]
            alpha = t[-1]
            return nlml(x, theta, var_init, alpha)

        optimal_result = minimize(objective, t_init, method = method, options = options)

        return optimal_result.x[:-1], var_init, optimal_result.x[-1]

    else:

        t_init = np.append(np.append(theta_init, var_init), alpha_init)

        def objective(t):
            theta = t[:-2]
            var = t[-2]
            alpha = t[-1]
            return nlml(x, theta, var, alpha)

        optimal_result = minimize(objective, t_init, method = method, options = options)

        return optimal_result.x[:-2], optimal_result.x[-2], optimal_result.x[-1]

def log_hyperparameters(x, log_theta_init, log_var_init = None, log_alpha_init = None, log_var_fix = 0.0, log_alpha_fix = 0.0):

    options = {'disp': False, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000}
    method = 'L-BFGS-B'

    if log_var_init is None and log_alpha_init is None:

        log_var_init = log_var_fix
        log_alpha_init = log_alpha_fix

        def objective(t):
            return nlml_log(x, t, log_var_init, log_alpha_init)

        optimal_result = minimize(objective, log_theta_init, method = method, options = options)
        return optimal_result.x, log_var_init, log_alpha_init

    elif log_alpha_init is None:

        log_alpha_init = log_alpha_fix
        t_init = np.append(log_theta_init, log_var_init)

        def objective(t):
            log_theta = t[:-1]
            log_var = t[-1]
            return nlml_log(x, log_theta, log_var, log_alpha_init)

        optimal_result = minimize(objective, t_init, method = method, options = options)

        return optimal_result.x[:-1], optimal_result.x[-1], log_alpha_init

    elif log_var_init is None:

        log_var_init = log_var_fix
        t_init = np.append(log_theta_init, log_alpha_init)

        def objective(t):
            log_theta = t[:-1]
            log_alpha = t[-1]
            return nlml_log(x, log_theta, log_var_init, log_alpha)

        optimal_result = minimize(objective, t_init, method = method, options = options)

        return optimal_result.x[:-1], log_var_init, optimal_result.x[-1]

    else:

        t_init = np.append(np.append(log_theta_init, log_var_init), log_alpha_init)

        def objective(t):
            log_theta = t[:-2]
            log_var = t[-2]
            log_alpha = t[-1]
            return nlml_log(x, log_theta, log_var, log_alpha)

        optimal_result = minimize(objective, t_init, method = method, options = options)

        return optimal_result.x[:-2], optimal_result.x[-2], optimal_result.x[-1]            