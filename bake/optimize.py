"""
Optimization Module.
"""
import numpy as np
from scipy.optimize import minimize
import logging


def sample_optimization(objective, t_min, t_max, n_samples=1000):
    """
    Evaluate the objective at random sample points and pick its optimum.

    Parameters
    ----------
    objective : callable
        The objective function to be minimized
    t_min : numpy.ndarray
        The minimum bound on the input (d,)
    t_max : numpy.ndarray
        The maximum bound on the input (d,)
    n_samples : int, optional
        The number of sample points to use

    Returns
    -------
    numpy.ndarray
        The optimal input parameters (d,)
    float
        The optimal objective value
    """
    assert t_min.shape == t_max.shape
    t_range = t_max - t_min
    assert t_range.ndim == 1
    d, = t_range.shape

    t_samples_standardised = np.random.rand(n_samples, d)
    t_samples = t_range * t_samples_standardised + t_min

    t_opt = np.random.rand(d)
    f_opt = objective(t_opt)

    logging.debug('Sample Optimization Initialized')
    logging.debug('Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    for i, t in enumerate(t_samples):
        f = objective(t)
        if f < f_opt:
            # print(t_opt, f_opt)
            t_opt = t
            f_opt = f
            logging.debug('Iteration: %d | Hyperparameters: %s | Objective: %f'
                          % (i, str(t), f))

    logging.debug('Sample Optimization Completed')
    logging.debug('Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt


def local_optimization(objective, t_min, t_max, t_init, jac=None):
    """
    Perform local optimization.

    Parameters
    ----------
    objective : callable
        The objective function to be minimized
    t_min : numpy.ndarray
        The minimum bound on the input (d,)
    t_max : numpy.ndarray
        The maximum bound on the input (d,)
    t_init : numpy.ndarray
        The initial value of the input (d,)

    Returns
    -------
    numpy.ndarray
        The optimal input parameters (d,)
    float
        The optimal objective value
    """
    method = 'L-BFGS-B'
    options = {'disp': True, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05,
               'eps': 1e-08, 'maxiter': 15000, 'ftol': 1e-9,
               'maxcor': 20, 'maxfun': 15000}
    bounds = [(t_min[i], t_max[i]) for i in range(len(t_init))]

    logging.debug('Local Optimization Initialized')
    logging.debug('Hyperparameters: %s | Objective: %f'
                  % (str(t_init), objective(t_init)))

    optimal_result = minimize(objective, t_init,
                              method=method, bounds=bounds, options=options,
                              jac=jac)
    t_opt, f_opt = optimal_result.x, optimal_result.fun

    logging.debug('Local Optimization Completed')
    logging.debug('Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt


def explore_optimization(objective, t_min, t_max, n_samples=1000, jac=None):
    """
    Perform sample optimization to initialize a local optimization procedure.

    Parameters
    ----------
    objective : callable
        The objective function to be minimized
    t_min : numpy.ndarray
        The minimum bound on the input (d,)
    t_max : numpy.ndarray
        The maximum bound on the input (d,)
    n_samples : int, optional
        The number of sample points to use

    Returns
    -------
    numpy.ndarray
        The optimal input parameters (d,)
    float
        The optimal objective value
    """
    t_init, f_init = sample_optimization(objective, t_min, t_max,
                                         n_samples=n_samples)
    t_opt, f_opt = local_optimization(objective, t_min, t_max, t_init, jac=jac)

    logging.debug('Explore Optimization Completed')
    logging.debug('Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt


def multi_explore_optimization(objective, t_min, t_max,
                               n_samples=100, n_repeat=10, jac=None):
    """
    Repeat explore optimisation and pick the best optimum from the results.

    Parameters
    ----------
    objective : callable
        The objective function to be minimized
    t_min : numpy.ndarray
        The minimum bound on the input (d,)
    t_max : numpy.ndarray
        The maximum bound on the input (d,)
    n_samples : int, optional
        The number of sample points to use
    n_repeat : int, optional
        The number of multiple explore optimisation to be performed

    Returns
    -------
    numpy.ndarray
        The optimal input parameters (d,)
    float
        The optimal objective value
    """
    results = [explore_optimization(objective, t_min, t_max,
                                    n_samples=n_samples, jac=jac)
               for i in range(n_repeat)]
    results = list(map(list, zip(*results)))

    t_list = results[0]
    f_list = results[1]

    i_opt = np.argmin(f_list)
    t_opt, f_opt = t_list[i_opt], f_list[i_opt]

    logging.debug('Multi-Explore Optimization Completed')
    logging.debug('List of local optimums and their objective function value: ')
    [logging.debug(t_list[i], f_list[i]) for i in range(n_repeat)]
    logging.debug('Best Optimum:')
    logging.debug('Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt


def pack(t_tuple):
    """
    Pack a tuple of lists or arrays together as an array.

    Parameters
    ----------
    t_tuple : tuple
        A tuple of lists or arrays

    Returns
    -------
    numpy.array
        The packed array
    list
        The locations of where the array is to be split when unpacking
    """
    t = np.concatenate(t_tuple)
    t_sizes = [len(t_) for t_ in t_tuple]
    t_breaks = np.cumsum(t_sizes)
    t_indices = np.split(np.arange(t_breaks[-1]), t_breaks)[:-1] # list
    return t, t_indices


def multi_pack(*args):
    """
    Pack an arbitrary amount of tuple of lists or arrays that are in the same
    format together as arrays.

    Parameters
    ----------
    args : *tuple
        Tuples of lists or arrays that are all in the same format

    Returns
    -------
    *tuple
        A tuple of packed arrays
    list
        The locations of where the arrays are to be split when unpacking
    """
    def same_split(t_indices_1, t_indices_2):
        return np.prod([t_indices_1[i].shape[0] == t_indices_2[i].shape[0]
                        for i in range(len(t_indices_1))])

    packed_args = [pack(arg) for arg in args]

    if not np.prod([same_split(packed_args[0][1], packed_args[i][1])
                    for i in range(len(args))]):
        raise ValueError('Parameter tuples are not of the same format.')

    t_indices = packed_args[0][1]
    ts = [packed_args[i][0] for i in range(len(args))]
    ts.append(t_indices)

    return tuple(ts)


def unpack(t, t_indices):
    """
    Unpack arrays into tuples.

    Parameters
    ----------
    t : numpy.array
        A packed array
    t_indices : list
        The locations of where the array is to be split for unpacking

    Returns
    -------
    tuple
        The original unpacked tuple
    """
    return tuple([t[i] for i in t_indices])


def hyper_opt(f, data, hyper_min, hyper_max,
              hyper_init=None, n_samples=1000, n_repeat=1, hyper_warp=None):
    """
    Optimize Hyperparameters.

    Parameters
    ----------
    f : callable
        The learning objective to be minimized
    data : tuple
        The relevant data and cache, determined by the learning objective
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
    hyper_warp : callable, optional
        The function that wraps the parameter space

    Returns
    -------
    tuple
        A tuple of arrays representing the optimal hyperparameters
    """
    # Pack the hyper tuples into hyper arrays for optimization
    if hyper_init is None:
        t_min, t_max, t_indices = multi_pack(hyper_min, hyper_max)
    else:
        hyper_init = tuple([np.array(a) for a in hyper_init])
        t_min, t_max, t_init, t_indices = multi_pack(hyper_min, hyper_max,
                                                     hyper_init)
        assert t_init.shape == t_min.shape
    assert t_min.shape == t_max.shape

    # Define the objective
    if hyper_warp is None:
        def objective(t):
            return f(*unpack(t, t_indices), data)
    else:
        def objective(t):
            return f(*hyper_warp(*unpack(t, t_indices)), data)

    # Perform optimization
    if hyper_init is None:
        t_opt, f_opt = multi_explore_optimization(objective, t_min, t_max,
                                                  n_samples=n_samples,
                                                  n_repeat=n_repeat)
    else:
        t_opt, f_opt = local_optimization(objective, t_min, t_max, t_init)

    # Unpack the hyperparameters in the same format
    return unpack(t_opt, t_indices)


def solve_unit_constrained_quadratic_iterative(a, b, x_init):
    """
    Solve a quadratic problem whose inputs are unit constrained.

    Parameters
    ----------
    a : numpy.ndarray
        The symmetric, positive definite matrix in the objective (n, n)
    b : numpy.ndarray
        The linear coefficient vector in the objective (n,)
    x_init : numpy.ndarray
        The initial solution to start the optimization with (n,)

    Returns
    -------
    numpy.ndarray
        The final optimal solution (n,)
    """
    def objective(x):
        return 0.5 * np.dot(x, np.dot(a, x)) + np.dot(b, x)
    zeros = np.zeros(x_init.shape[0])
    ones = np.ones(x_init.shape[0])
    x_opt, f_opt = local_optimization(objective, zeros, ones, x_init)
    return x_opt


def solve_normalized_unit_constrained_quadratic_iterative(a, b, x_init):
    """
    Solve a quadratic problem whose inputs are unit constrained and normalized.

    This method sets up the optimization as is with the original constraints.

    .. note:: Experimentally, this is not working as well as the altered method.

    Parameters
    ----------
    a : numpy.ndarray
        The symmetric, positive definite matrix in the objective (n, n)
    b : numpy.ndarray
        The linear coefficient vector in the objective (n,)
    x_init : numpy.ndarray
        The initial solution to start the optimization with (n,)

    Returns
    -------
    numpy.ndarray
        The final optimal solution (n,)
    """
    def objective(x):
        return 0.5 * np.dot(x, np.dot(a, x)) + np.dot(b, x)

    def constraint(x):
        return np.sum(x) - 1

    constraints = {'type': 'eq', 'fun': constraint}
    options = {'disp': False}
    bounds = [(0.0, 1.0) for x in x_init]

    optimal_result = minimize(objective, x_init,
                              bounds=bounds,
                              constraints=constraints,
                              options=options)
    return optimal_result.x


def solve_normalized_unit_constrained_quadratic(a, b, x_init):
    """
    Solve a quadratic problem whose inputs are unit constrained and normalized.

    This method reduces the dimensionality of the problem by incorporating the
    equality constraint into the objective. However, as a result, it is not
    guaranteed that the last input is positive.

    Parameters
    ----------
    a : numpy.ndarray
        The symmetric, positive definite matrix in the objective (n, n)
    b : numpy.ndarray
        The linear coefficient vector in the objective (n,)
    x_init : numpy.ndarray
        The initial solution to start the optimization with (n,)

    Returns
    -------
    numpy.ndarray
        The final optimal solution (n,)
    """
    n, = x_init.shape
    a_matrix = a[:-1, :-1]
    a_vector = a[-1, :-1]
    a_slider = np.tile(a_vector, (n - 1, 1))
    a_number = a[-1, -1]
    b_vector = b[:-1]
    one_matrix = np.ones((n - 1, n - 1))
    one_vector = np.ones(n - 1)

    a_new = a_matrix - 2 * a_slider + a_number * one_matrix
    b_new = a_vector + b_vector - (a_number + 1) * one_vector

    z_init = x_init[:-1]

    z_opt = solve_unit_constrained_quadratic_iterative(a_new, b_new, z_init)
    x_n_opt = 1 - np.sum(z_opt)
    return np.append(z_opt, x_n_opt)