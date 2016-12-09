"""
Demonstration of Bayesian optimisation with Gaussian processes
"""
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from manifold.regression import GPRegressor
from bayesian_optimization.active_samplers import \
    GaussianProcessSampler, RandomSampler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

seed = 20

true_phenomenon = utils.benchmark.branin_hoo
d = true_phenomenon.n_dim
x_min = true_phenomenon.x_min
x_max = true_phenomenon.x_max


def noisy_phenomenon(x, sigma=1.0):
    """
    Noisy Function.

    Parameters
    ----------
    x : numpy.ndarray
        The input to the function of size (n, 2)
    sigma : float, optional
        The noise level to be added

    Returns
    -------
    numpy.ndarray
        The noisy output from the function
    """
    f = true_phenomenon(x)
    e = np.random.normal(loc=0, scale=sigma, size=x.shape[0]) \
        if sigma > 0 else 0
    y = f + e
    return y


def create_training_data(n=100, sigma=1.0):
    """
    Create training data.

    Parameters
    ----------
    n, optional: int
        The number of training data points
    sigma : float, optional
        The noise level to be added

    Returns
    -------
    np.ndarray
        The input data of size (n, d)
    np.ndarray
        The output data of size (n,)
    """
    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = noisy_phenomenon(x, sigma=1.0)
    return x, y


def setup_model():
    """
    Setup the model.

    Must have fit, update
    Returns
    -------

    """
    # Initialise a Gaussian Process Regressor
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    return GaussianProcessSampler(kernel=kernel, n_stop_train=30)


def bayesian_optimisation():
    """
    Run Bayesian Optimisation.
    """
    # Create training data
    n_start = 5
    sigma = 0.0
    x, y = create_training_data(n=n_start, sigma=sigma)

    model = setup_model()

    # Generate the query points
    n_query = 250
    x_1_lim = (x_min[0], x_max[0])
    x_2_lim = (x_min[1], x_max[1])
    x_1_q, x_2_q, x_1_mesh, x_2_mesh = \
        utils.visuals.uniform_query(x_1_lim, x_2_lim, n_query, n_query)
    x_q = np.array([x_1_mesh.ravel(), x_2_mesh.ravel()]).T

    # Ground Truth
    y_q_true = true_phenomenon(x_q)
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
    model.fit(x, y)
    n_trials = 30

    metric_success_trial = np.inf
    for i in range(n_trials):

        # Pick a location to observe
        x_star = model.pick(x_q)
        x_stars.append(x_star[0])

        # Observe that location
        y_star = noisy_phenomenon(x_star, sigma=sigma)
        y_stars.append(y_star[0])

        # Update the model about the new observation
        model.update(x_star, y_star)

        # Use a loss metric to measure the current performance
        loss_value = utils.benchmark.loss_opt_loc(x=x_star,
                                                  function=true_phenomenon,
                                                  dist_ratio=0.01)
        if utils.benchmark.success_opt_loc(loss_value) \
                and metric_success_trial == np.inf:
            metric_success_trial = i + 1
            print('Success at %d steps with loss %f' % (metric_success_trial,
                                                        loss_value))

        print('Step %d' % (i + 1))

    final_loss_value = utils.benchmark.loss_opt_loc(x=x_opt,
                                              function=true_phenomenon,
                                              dist_ratio=0.01)

    if metric_success_trial < np.inf:
        print('Number of steps taken to succeed (up to %d): %d' % (
            n_trials, metric_success_trial))
    else:
        print('Did not succeed after %d steps' % n_trials)
    print('Final loss value after %d steps: %f' % (n_trials, final_loss_value))

    ### PLOTTING ###
    x_stars = np.array(x_stars)
    y_stars = np.array(y_stars)
    print(y.shape, y_stars.shape)
    x = np.concatenate((x, x_stars), axis=0)
    y = np.concatenate((y, y_stars), axis=0)
    y_q_exp = model.predict(x_q)
    y_q_exp_mesh = np.reshape(y_q_exp, (n_query, n_query))
    acq = model.acquisition(x_q)
    acq_mesh = np.reshape(acq, (n_query, n_query))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_q_true_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$f(x)$')
    ax.set_title('Ground Truth')
    ax.scatter(true_phenomenon.x_opt[:, 0], true_phenomenon.x_opt[:, 1],
               true_phenomenon.f_opt,
               c=true_phenomenon.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_q_exp_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$\mu(x)$')
    ax.set_title('Mean Prediction')
    ax.scatter(true_phenomenon.x_opt[:, 0], true_phenomenon.x_opt[:, 1],
               true_phenomenon.f_opt,
               c=true_phenomenon.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x[:, 0], x[:, 1], y, c=y, vmin=vmin, vmax=vmax,
               label='Training Data')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], y_stars,
               c=y_stars, marker='+', vmin=vmin, vmax=vmax,
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
    ax.set_zlabel(r'$\alpha_{EI}(x)$')
    ax.set_title('Expected Improvement')
    ax.scatter(true_phenomenon.x_opt[:, 0], true_phenomenon.x_opt[:, 1],
               np.min(Z),
               c=true_phenomenon.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x[:, 0], x[:, 1], np.min(Z), c=y, vmin=vmin, vmax=vmax,
               label='Training Data')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], np.min(Z),
               c=y_stars, marker='+', vmin=vmin, vmax=vmax,
               label='Active Sampled Data')
    ax.legend()

    x_star_history = np.concatenate((x_opt_init, x_stars), axis=0)
    y_star_history = np.append(y_opt_init, y_stars)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(np.arange(y_star_history.shape[0]), y_star_history, c='c',
            where='post', label='Sampled values')
    ax.step(np.arange(y_star_history.shape[0]),
            np.maximum.accumulate(y_star_history), c='g', where='post',
            label='Current Optimum')
    ax.axhline(y=true_phenomenon.f_opt[0], color = 'k',
               label='True Optimum')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('$y$')
    ax.set_title('Active sampled target values v.s. Iterations')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_q_true_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$f(x)$')
    ax.set_title('Trajectory of Active Samples')
    ax.plot(x_star_history[:, 0], x_star_history[:, 1], y_star_history,
            c='k', label='Trajectory of Active Samples')
    ax.plot(x_star_history[:, 0], x_star_history[:, 1], np.min(Z),
            linestyle=':', c='k', label='Trajectory of Active Samples')
    m = np.concatenate((x_star_history, y_star_history[:, np.newaxis]), axis=1)
    # ax.scatter(m[:, 0], m[:, 1], m[:, 2], c=m[:, 2])
    for i in range(len(m)):  # plot each point + it's index as text above
        ax.text(m[i, 0], m[i, 1], np.min(Z) + 0.05*(np.max(Z) - np.min(Z)),
                '%s' % (str(i)), size=10, zorder=1, color='k')


def gaussian_expected_improvement(mu, std, best):
    """
    Expected Improvement Acquisition Function for a Gaussian process.

    Parameters
    ----------
    mu : numpy.ndarray
        The mean of the predictions (n_q,)
    std : numpy.ndarray
        The standard deviations of the predictions (n_q,)
    best : float
        The maximum observed value so far

    Returns
    -------
    numpy.ndarray
        The acquisition function evaluated at the corresponding points (n_q,)
    """
    diff = mu - best
    abs_diff = np.abs(diff)
    clip_diff = np.clip(diff, 0, np.inf)
    return clip_diff + std*norm.pdf(diff/std) - abs_diff*norm.cdf(-abs_diff/std)
#

if __name__ == "__main__":
    utils.misc.time_module(bayesian_optimisation)
    plt.show()