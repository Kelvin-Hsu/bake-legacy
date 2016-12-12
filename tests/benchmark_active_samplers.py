"""
Benchmark framework for active samplers.

This script benchmarks the performance of active samplers on standard 2D test
functions.
"""
import utils
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from bayesian_optimization.active_samplers import \
    GaussianProcessSampler, RandomSampler
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

# TODO: REMEMBER CORRECTNESS AND SIMPLICITY BEATS OPTIMIZED CODE FOR SPEED

N_QUERY = 250
N_TRIALS = 60

logging.basicConfig(level=logging.DEBUG)


def setup_test_funcs():
    """
    Setup test functions.

    The test functions must be 2D and adhere to the format
    given in 'benchmark.py'.

    Returns
    -------
    lists
        A list of test functions.
    """
    return [utils.benchmark.branin_hoo,
            utils.benchmark.griewank,
            utils.benchmark.levy,
            utils.benchmark.schaffer]


def create_query_data(test_func, n_query=N_QUERY):
    """
    Create a 2D grid of query points.

    The query points are restricted to 2D for testing purposes.
    This also makes it possible to visualize the results.

    Parameters
    ----------
    test_func : callable
        A test function
    n_query : int, optional
        The number of query points in each dimension

    Returns
    -------
    numpy.ndarray
        A set of query points of size (n_query * n_query, 2)
    """
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


def setup_sampler_initializers():
    """
    Setup sampler initializers.

    This is where each sampler is initialized with its initial settings.

    Returns
    -------
    list
        A list of sampler initializers
    """
    gaussian_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    matern_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(10, (1e-2, 1e2))

    kernels = [gaussian_kernel, matern_kernel]
    acquisition_methods = ['EI']
    n_stop_trains = [25]

    sampler_initializers = [lambda: GaussianProcessSampler(
        kernel=kernel,
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

    sampler_initializers.append(lambda: RandomSampler())
    sampler_names.append('Random Sampler')
    return sampler_initializers, sampler_names


def simulate_performance_metric(test_func, sampler_initializer,
                                n_query=N_QUERY, n_trials=N_TRIALS,
                                return_visual_data=False):
    """
    Run a Bayesian optimization procedure to test active samplers.

    For this benchmarking scenario the active sampler is to find the global
    optimum of the test function with the least number of function evaluations.

    The sampler is started off without any training data, so it literally
    has no prior knowledge of the function, not even from random samples.

    The performance metric is determined by each test function/scenario, where
    the loss is a distance measure from the current proposal point to the
    actual optimum of the test function. The active sampler succeeds in the test
    if and only if it is able to propose a point that is very close to the
    actual optimum, as determined by the test function/scenario.

    Parameters
    ----------
    test_func : callable
        The test function
    sampler_initializer : callable
        A sampler initializer which initializes the sampler a-fresh
    n_query : int, optional
        The number of query points in each dimension
    n_trials : int, optional
        The number of steps the Bayesian optimisation will run for
    return_visual_data : bool, optional
        Whether or not data relevant to visualizing the benchmark is returned

    Returns
    -------
    float
        The final loss value achieved by the sampler under this test
    int
        The number of steps taken by the sampler to succeed (infinite if failed)
    tuple, optional
        A collection of data relevant to visualizing the benchmark
    """
    x_query = create_query_data(test_func, n_query=n_query)
    x_stars = []
    y_stars = []
    success_trial = np.inf
    sampler = sampler_initializer()
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
                and success_trial == np.inf:
            success_trial = i + 1
            logging.info('\t\tSuccess at %d steps with loss %f'
                  % (success_trial, loss_value))
        logging.info('\t\tStep %d | Immediate loss value: %f'
                     % (i + 1, loss_value))

    x_stars = np.array(x_stars)
    y_stars = np.array(y_stars)
    x_opt = x_stars[[np.argmax(y_stars)]]
    final_loss_value = utils.benchmark.loss_opt_loc(x=x_opt,
                                                    function=test_func,
                                                    dist_ratio=0.01)

    if success_trial < np.inf:
        logging.info('\t\tNumber of steps taken to succeed (up to %d): %d' % (
            n_trials, success_trial))
    else:
        logging.info('\t\tDid not succeed after %d steps' % n_trials)

    logging.info('\t\tFinal loss value after %d steps: %f'
          % (n_trials, final_loss_value))

    if return_visual_data:
        visual_data = (sampler, test_func, x_query, x_stars, y_stars)
        return final_loss_value, success_trial, visual_data
    else:
        return final_loss_value, success_trial


def benchmark_active_samplers(setup_sampler_initializers=
                              setup_sampler_initializers,
                              result_filename='active_sampler_benchmark.npz',
                              n_query=N_QUERY, n_trials=N_TRIALS, seed=0):
    """
    Benchmark active samplers and store the results.

    Parameters
    ----------
    setup_sampler_initializers : callable, optional
        A function for setting up the sampler initializers
    result_filename : str, optional
        The file name to store the results in
    n_query : int, optional
        The number of query points in each dimension
    n_trials : int, optional
        The number of steps the Bayesian optimisation will run for
    seed : int, optional
        The randomization seed

    Returns
    -------
    None
    """
    np.random.seed(seed)
    sampler_initializers, sampler_names = setup_sampler_initializers()
    test_funcs = setup_test_funcs()

    final_loss_values = np.zeros((len(sampler_initializers), len(test_funcs)))
    success_trial = np.zeros((len(sampler_initializers), len(test_funcs)))
    visual_data_matrix = []
    for i, sampler_initializer in enumerate(sampler_initializers):
        visual_data_list = []
        for j, test_func in enumerate(test_funcs):
            test_case = '"%s" on "%s"' % (sampler_names[i], test_func.name)
            logging.info('Benchmarking: %s' % test_case)
            final_loss_values[i, j], success_trial[i, j], visual_data = \
                simulate_performance_metric(test_func, sampler_initializer,
                                            n_query=n_query,
                                            n_trials=n_trials,
                                            return_visual_data=True)
            visual_data_list.append(visual_data)
        visual_data_matrix.append(visual_data_list)

    np.savez(result_filename,
             final_loss_values=final_loss_values,
             success_trial=success_trial,
             visual_data_matrix=visual_data_matrix,
             sampler_names=sampler_names,
             test_funcs=test_funcs,
             n_query=n_query,
             n_trials=n_trials)


def extract_result(result_file):
    """
    Extract result from a npz file into a 'BenchmarkResult' instance.

    Parameters
    ----------
    result_file : numpy.lib.npyio.NpzFile
        The npz file storing the results from 'benchmark_active_samplers'

    Returns
    -------
    tests.benchmark_active_samplers.BenchmarkResult
        A 'BenchmarkResult' instance
    """
    return BenchmarkResult(
        final_loss_values=result_file['final_loss_values'],
        success_trial=result_file['success_trial'],
        visual_data_matrix=result_file['visual_data_matrix'],
        sampler_names=result_file['sampler_names'],
        test_funcs=result_file['test_funcs'],
        n_query=result_file['n_query'],
        n_trials=result_file['n_trials'])


def visualize_benchmark(sampler, test_func, x_query, x_stars, y_stars):
    """
    Visualize the benchmark result.

    Parameters
    ----------
    sampler
        A sampler class
    test_func : callable
        A test function
    x_query : numpy.ndarray
        A collection of query points used for the benchmark
    x_stars : numpy.ndarray
        The active sampled input locations.
    y_stars
        The active sampled outputs observations.

    Returns
    -------
    None
    """
    x_1 = x_query[:, 0]
    x_2 = x_query[:, 1]
    n_query_squared = x_query.shape[0]
    n_query = int(np.sqrt(n_query_squared))

    x_1_mesh = np.reshape(x_1, (n_query, n_query))
    x_2_mesh = np.reshape(x_2, (n_query, n_query))

    y_true = test_func(x_query)
    y_true_mesh = np.reshape(y_true, (n_query, n_query))
    y_pred = sampler.predict(x_query)
    y_pred_mesh = np.reshape(y_pred, (n_query, n_query))
    acq = sampler.acquisition(x_query)
    acq_mesh = np.reshape(acq, (n_query, n_query))

    vmin = np.min(y_true)
    vmax = np.max(y_true)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_true_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$f(x)$')
    ax.set_title('Ground Truth')
    ax.scatter(test_func.x_opt[:, 0], test_func.x_opt[:, 1],
               test_func.f_opt,
               c=test_func.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_pred_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$\mu(x)$')
    ax.set_title('Mean Prediction')
    ax.scatter(test_func.x_opt[:, 0], test_func.x_opt[:, 1],
               test_func.f_opt,
               c=test_func.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], y_stars,
               c=y_stars, vmin=vmin, vmax=vmax,
               label='Active Sampled Data')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, acq_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel(r'$\alpha(x)$')
    ax.set_title('Acquisition Function')
    ax.scatter(test_func.x_opt[:, 0], test_func.x_opt[:, 1],
               np.min(Z),
               c=test_func.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], np.min(Z),
               c=y_stars, vmin=vmin, vmax=vmax,
               label='Active Sampled Data')
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(np.arange(y_stars.shape[0]), y_stars, c='c',
            where='post', label='Sampled values')
    ax.step(np.arange(y_stars.shape[0]),
            np.maximum.accumulate(y_stars), c='g', where='post',
            label='Current Optimum')
    ax.axhline(y=test_func.f_opt[0], color = 'k',
               label='True Optimum')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('$y$')
    ax.set_title('Active Sampled Target Values')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_pred_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$f(x)$')
    ax.set_title('Trajectory of Active Samples')
    ax.plot(x_stars[:, 0], x_stars[:, 1], y_stars,
            c='k', label='Trajectory of Active Samples')
    ax.plot(x_stars[:, 0], x_stars[:, 1], np.min(Z),
            linestyle=':', c='k', label='Trajectory of Active Samples')
    stars = np.concatenate((x_stars, y_stars[:, np.newaxis]), axis=1)
    for i in range(len(stars)):
        ax.text(stars[i, 0], stars[i, 1],
                np.min(Z) + 0.05*(np.max(Z) - np.min(Z)),
                '%s' % (str(i)), size=10, zorder=1, color='k')


class BenchmarkResult():
    """
    Benchmark Results.

    Parameters
    ----------
    final_loss_values : numpy.ndarray
        A 2D array of size (n_samplers, n_test_funcs) summarizing loss values
    success_trial : numpy.ndarray
        A 2D array of size (n_samplers, n_test_funcs) summarizing success trials
    visual_data_matrix : list
        A 2D list of size (n_samplers, n_test_funcs) containing visual data
    sampler_names : list
        A list of length (n_samplers,) containing names for the samplers
    test_funcs : list
        A list of length (n_test,) containing the test functions
    n_query : int, optional
        The number of query points in each dimension
    n_trials : int, optional
        The number of steps the Bayesian optimisation ran for

    Attributes
    ----------
    final_loss_values : numpy.ndarray
        A 2D array of size (n_samplers, n_test_funcs) summarizing loss values
    success_trial : numpy.ndarray
        A 2D array of size (n_samplers, n_test_funcs) summarizing success trials
    visual_data_matrix : list
        A 2D list of size (n_samplers, n_test_funcs) containing visual data
    sampler_names : list
        A list of length (n_samplers,) containing names for the samplers
    test_funcs : list
        A list of length (n_test,) containing the test functions
    n_query : int, optional
        The number of query points in each dimension
    n_trials : int, optional
        The number of steps the Bayesian optimisation ran for
    """
    def __init__(self,
                 final_loss_values=None,
                 success_trial=None,
                 visual_data_matrix=None,
                 sampler_names=None,
                 test_funcs=None,
                 n_query=None,
                 n_trials=None):
        self.final_loss_values = final_loss_values
        self.success_trial = success_trial
        self.visual_data_matrix = visual_data_matrix
        self.sampler_names = sampler_names
        self.test_funcs = test_funcs
        self.n_query = n_query
        self.n_trials = n_trials

    def summary(self):
        """
        Summarize the benchmark results.

        Returns
        -------
        None
        """
        print('This benchmark test uses %d query points in each '
                     'dimension and runs each optimiser for %d iterations'
                     % (self.n_query, self.n_trials))
        print('The active samplers tested are:')
        [print('\t%s' % name) for name in self.sampler_names]
        print('Each sampler is tested on the following functions:')
        [print('\t%s' % func.name) for func in self.test_funcs]
        print('Loss metric table (rows: samplers, cols: test functions):')
        print('------------------------------------------------------------')
        print(self.final_loss_values)
        print('------------------------------------------------------------')
        print('Trials until success (rows: samplers, cols: test functions):')
        print('------------------------------------------------------------')
        print(self.success_trial)
        print('------------------------------------------------------------')

    def visualize(self, i, j):
        """
        Visualize the benchmark results for a particular test case.

        Parameters
        ----------
        i : int
            The index determining the sampler being benchmarked.
        j : int
            The index determining the test function used for the benchmark.

        Returns
        -------
        None
        """
        test_case = '"%s" on "%s"' % (self.sampler_names[i],
                                      self.test_funcs[j].name)
        print('Showing results for testing %s' % test_case)
        visual_data = self.visual_data_matrix[i, j]
        if self.success_trial[i, j] < np.inf:
            print('Number of steps taken to succeed (up to %d): %d' % (
                  self.n_trials, self.success_trial[i, j]))
        else:
            print('Did not succeed after %d steps' % self.n_trials)

        print('Final loss value after %d steps: %f'
              % (self.n_trials, self.final_loss_values[i, j]))
        return visualize_benchmark(*visual_data)


if __name__ == "__main__":
    utils.misc.time_module(benchmark_active_samplers)