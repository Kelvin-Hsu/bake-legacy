"""
Testing framework for active samplers.
"""
import utils
import numpy as np

from bayesian_optimization.active_samplers import \
    GaussianProcessSampler, RandomSampler
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

# TODO: REMEMBER CORRECTNESS AND SIMPLICITY BEATS OPTIMIZED CODE FOR SPEED


def setup_test_func():

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
    assert n_dim == x_query.shape

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
    samplers.append(RandomSampler())
    return samplers


def test_opt_loc_metric(test_func, sampler, x_start, y_start, x_query,
                        n_trials=30):
    """
    Run Bayesian Optimisation.
    """
    assert x_start.shape[1] == x_query.shape[1]
    assert x_start.shape[0] == y_start.shape[0]
    n_start, = y_start.shape
    n_query, n_dim = x_query.shape

    # Ground Truth
    y_q_true = test_func(x_query)
    y_q_true_mesh = np.reshape(y_q_true, (n_query, n_query))
    vmin = np.min(y_q_true)
    vmax = np.max(y_q_true)

    # Best observation so far
    i_opt = np.argmax(y)
    x_opt_init = x[[i_opt]]
    y_opt_init = y[i_opt]
    x_opt = x[[i_opt]]

    # The proposed points
    x_stars = []
    y_stars = []

    # Start Bayesian Optimisation
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
            print('Success at %d steps with loss %f' % (metric_success_trial,
                                                        loss_value))
        print('Step %d' % (i + 1))

    final_loss_value = utils.benchmark.loss_opt_loc(x=x_opt,
                                                    function=test_func,
                                                    dist_ratio=0.01)

    if metric_success_trial < np.inf:
        print('Number of steps taken to succeed (up to %d): %d' % (
            n_trials, metric_success_trial))
    else:
        print('Did not succeed after %d steps' % n_trials)
    print('Final loss value after %d steps: %f' % (n_trials, final_loss_value))


if __name__ == "__main__":
    utils.misc.time_module(test_opt_loc_metric)