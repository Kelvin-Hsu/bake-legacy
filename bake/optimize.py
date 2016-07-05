"""
Optimisation Module.
"""
import numpy as np
from scipy.optimize import minimize

def sample_optimisation(objective, t_min, t_max, n = 1000):

    assert t_min.shape == t_max.shape
    t_range = t_max - t_min
    assert t_range.ndim == 1
    m = t_range.shape[0]

    t_samples_standardised = np.random.rand(n, m)

    t_samples = t_range * t_samples_standardised + t_min

    t_opt = t_samples[0]
    f_opt = objective(t_opt)

    i = 0

    for t in t_samples:

        f = objective(t)

        if f < f_opt:

            t_opt = t
            f_opt = f
            print('Iteration: %d | Hyperparameters: %s | Objective: %f' % (i, str(t), f))

        i += 1

    return t_opt, f_opt

def local_optimisation(objective, t_min, t_max, t_init):

    method = 'L-BFGS-B'
    options = {'disp': False, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 1e-9, 'maxcor': 20, 'maxfun': 15000}
    bounds = [(t_min[i], t_max[i]) for i in range(len(t_min))]

    optimal_result = minimize(objective, t_init, method = method, bounds = bounds, options = options)
    t_opt, f_opt = optimal_result.x, optimal_result.fun

    # print('Optimisation Completed || Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt

def explore_optimisation(objective, t_min, t_max, n = 1000):

    t_init, f_init = sample_optimisation(objective, t_min, t_max, n = n)

    print('Optimisation Initialised || Hyperparameters: %s | Objective: %f' % (str(t_init), f_init))

    t_opt, f_opt = local_optimisation(objective, t_min, t_max, t_init)

    print('Optimisation Completed || Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt

def multi_explore_optimisation(objective, t_min, t_max, n = 1000, repeat = 10):

    results = [explore_optimisation(objective, t_min, t_max, n = n) for i in range(repeat)]
    results = list(map(list, zip(*results)))

    t_list = results[0]
    f_list = results[1]

    print('List of local optimums and their objective function value: ')
    [print(t_list[i], f_list[i]) for i in range(repeat)]

    i_opt = np.argmin(f_list)
    t_opt, f_opt = t_list[i_opt], f_list[i_opt]

    print('Optimisation Finalised || Hyperparameters: %s | Objective: %f' % (str(t_opt), f_opt))

    return t_opt, f_opt

def multi_pack(*args):

    def same_split(t_indices_1, t_indices_2):
        return np.prod([t_indices_1[i].shape[0] == t_indices_2[i].shape[0] for i in range(len(t_indices_1))])

    packed_args = [pack(arg) for arg in args]

    if not np.prod([same_split(packed_args[0][1], packed_args[i][1]) for i in range(len(args))]):
        raise ValueError('Parameter tuples are not of the same format.')

    t_indices = packed_args[0][1]
    ts = [packed_args[i][0] for i in range(len(args))]
    ts.append(t_indices)
    return tuple(ts)
    
def pack(t_tuple):

    t = np.concatenate(t_tuple)
    t_sizes = [t_.shape[0] for t_ in t_tuple]
    t_breaks = np.cumsum(t_sizes)
    t_indices = np.split(np.arange(t_breaks[-1]), t_breaks)[:-1] # list
    return t, t_indices

def unpack(t, t_indices):

    return tuple([t[i] for i in t_indices])
