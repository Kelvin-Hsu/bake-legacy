"""
Testing framework for active samplers.
"""
import utils
import numpy as np

from bayesian_optimization.active_samplers import \
    GaussianProcessSampler, RandomSampler
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

# TODO: REMEMBER CORRECTNESS AND SIMPLICITY BEATS OPTIMIZED CODE FOR SPEED


def setup_test_funcs():

    return [utils.benchmark.branin_hoo,
            utils.benchmark.griewank,
            utils.benchmark.levy,
            utils.benchmark.schaffer]


def create_query_data(test_func, n_query=250):

    # Extract test function characteristics
    n_dim = test_func.n_dim
    x_min = test_func.x_min
    x_max = test_func.x_max

    # Ensure that the test function is 2D for this particular test setting
    assert n_dim == 2

    # Create a uniform grid of query points!
    x_1_lim = (x_min[0], x_max[0])
    x_2_lim = (x_min[1], x_max[1])
    _, _, x_q_1_mesh, x_q_2_mesh = \
        utils.visuals.uniform_query(x_1_lim, x_2_lim, n_query, n_query)
    x_query = np.array([x_q_1_mesh.ravel(), x_q_2_mesh.ravel()]).T
    return x_query


def create_starting_data(test_func, x_query, n_start=10, seed=20):
    """
    Create starting data.

    For testing purposes, we will select starting points from a given set of
    query points.

    Parameters
    ----------
    x_query : numpy.ndarray
        The query locations to select training locations from
    n_start, optional: int
        The number of starting (training) data points

    Returns
    -------
    np.ndarray
        The input data of size (n, d)
    np.ndarray
        The output data of size (n,)
    """
    # Ensure that the query points make sense in the context of the function
    n_dim = test_func.n_dim
    assert n_dim == x_query.shape[1]

    # Select points from the query points as the starting points
    n_query, n_dim = x_query.shape
    np.random.seed(seed)
    random_indices = np.random.permutation(np.arange(n_query))[:n_start]
    x_start = x_query[random_indices]
    y_start = test_func(x_start)
    return x_start, y_start


def setup_samplers():
    """
    Setup samplers to test.
    """
    gaussian_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    matern_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(10, (1e-2, 1e2))

    kernels = [gaussian_kernel, matern_kernel]
    acquisition_methods = ['EI', 'STD']
    n_stop_trains = [15, 30]

    samplers = [GaussianProcessSampler(kernel=kernel,
                                       acquisition_method=acquisition_method,
                                       n_stop_train=n_stop_train)
                   for kernel in kernels
                   for acquisition_method in acquisition_methods
                   for n_stop_train in n_stop_trains]

    kernel_names = ['Gaussian', 'Matern']
    acquisition_names = acquisition_methods
    n_stop_train_names = [str(n_stop_train) for n_stop_train in n_stop_trains]

    sampler_names = ['Gaussian Process Sampler with %s kernel, '
                      '%s acquisition, and %s training threshold'
                      % (kernel_name, acquisition_name, n_stop_train_name)
                      for kernel_name in kernel_names
                      for acquisition_name in acquisition_names
                      for n_stop_train_name in n_stop_train_names]

    samplers.append(RandomSampler())
    sampler_names.append('Random Sampler')
    for i, name in enumerate(sampler_names):
        samplers[i].name = sampler_names[i]
    return samplers


def simulate_performance_metric(test_func, sampler, x_start, y_start, x_query,
                                n_trials=30):
    assert x_start.shape[1] == x_query.shape[1]
    assert x_start.shape[0] == y_start.shape[0]
    x_stars = []
    y_stars = []
    sampler.fit(x_start, y_start)
    metric_success_trial = np.inf

    for i in range(n_trials):

        # Pick a location to observe
        x_star = sampler.pick(x_query)
        x_stars.append(x_star[0])

        # Observe that location
        y_star = test_func(x_star)
        y_stars.append(y_star[0])

        # Update the model about the new observation
        sampler.update(x_star, y_star)

        # Use a loss metric to measure the current performance
        loss_value = utils.benchmark.loss_opt_loc(x=x_star,
                                                  function=test_func,
                                                  dist_ratio=0.01)
        if utils.benchmark.success_opt_loc(loss_value) \
                and metric_success_trial == np.inf:
            metric_success_trial = i + 1
            print('\t\tSuccess at %d steps with loss %f'
                  % (metric_success_trial, loss_value))
        print('\t\tStep %d' % (i + 1))

    x_opt = x_query[[np.argmax(np.append(y_start, y_stars))]]
    final_loss_value = utils.benchmark.loss_opt_loc(x=x_opt,
                                                    function=test_func,
                                                    dist_ratio=0.01)

    if metric_success_trial < np.inf:
        print('\t\tNumber of steps taken to succeed (up to %d): %d' % (
            n_trials, metric_success_trial))
    else:
        print('\t\tDid not succeed after %d steps' % n_trials)
    print('\t\tFinal loss value after %d steps: %f'
          % (n_trials, final_loss_value))

    return final_loss_value


def average_performance_metric(test_func, sampler, n_query=250, n_start=10,
                               n_trials=50):

    x_query = create_query_data(test_func, n_query=n_query)

    seeds = 100*np.arange(10)
    metrics = np.zeros(seeds.shape)
    for i, seed in enumerate(seeds):
        print('\tSimulating performance with seed %d' % seed)
        x_start, y_start = create_starting_data(test_func, x_query,
                                                n_start=n_start, seed=seed)
        metric = simulate_performance_metric(test_func, sampler,
                                             x_start, y_start, x_query,
                                             n_trials=n_trials)
        metrics[i] = metric
    return np.mean(metrics)


def benchmark_active_samplers(n_query=250, n_start=10, n_trials=30):

    samplers = setup_samplers()
    test_funcs = setup_test_funcs()

    results = np.zeros((len(samplers), len(test_funcs)))
    for i, sampler in enumerate(samplers):
        for j, test_func in enumerate(test_funcs):
            print('Benchmarking: "%s" on "%s"' %
                  (samplers[i].names, test_func.name))
            results[i, j] = average_performance_metric(test_func, sampler,
                                                       n_query=n_query,
                                                       n_start=n_start,
                                                       n_trials=n_trials)
    np.savez('active_sampler_benchmark.npz', results=results,
                                             samplers=samplers,
                                             test_funcs=test_funcs,
                                             n_query = n_query,
                                             n_start = n_start,
                                             n_trials = n_trials)
    return results

if __name__ == "__main__":
    utils.misc.time_module(benchmark_active_samplers)