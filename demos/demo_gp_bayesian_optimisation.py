"""
Demonstration of simple kernel embeddings.
"""
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from manifold.regression import GPRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

seed = 20

true_phenomenon = utils.benchmark.levy
d = true_phenomenon.n_dim
x_min = true_phenomenon.x_min
x_max = true_phenomenon.x_max


def noisy_phenomenon(x, sigma=1.0):
    """
    Noisy Branin-Hoo Function.

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
    e = np.random.normal(loc=0, scale=sigma, size=x.shape[0])
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


def bayesian_optimisation(retrain=True):
    """
    Run Bayesian Optimisation.
    """
    # Create training data
    sigma = 0.1
    x, y = create_training_data(n=5, sigma=sigma)

    # Initialise a Gaussian Process Regressor
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GPRegressor(kernel=kernel)

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
    y_opt = y[i_opt]

    # The proposed points
    x_stars = []
    y_stars = []

    # Start Bayesian Optimisation
    gpr.fit(x, y)
    for i in range(32):

        if i > 0:

            if retrain:
                x = np.concatenate((x, x_star), axis=0)
                y = np.append(y, y_star)
                gpr.fit(x, y)
            else:
                gpr.update(x_star, y_star)

            if y_star > y_opt:
                x_opt = x_star
                y_opt = y_star

        # Prediction
        y_q_exp, y_q_std = gpr.predict(x_q, return_std=True)
        y_q_exp_mesh = np.reshape(y_q_exp, (n_query, n_query))

        # Expected Improvement
        ei = gaussian_expected_improvement(y_q_exp, y_q_std, y_opt)
        ei_mesh = np.reshape(ei, (n_query, n_query))

        # Proposed location of observation
        x_star = x_q[[np.argmax(ei)]]
        x_stars.append(x_star[0])

        # Observe!
        y_star = noisy_phenomenon(x_star, sigma=sigma)
        y_stars.append(y_star)

    x_stars = np.array(x_stars)
    y_stars = np.array(y_stars)

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, y_q_exp_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$E[f]$')
    ax.set_title('Predictions')
    ax.scatter(true_phenomenon.x_opt[:, 0], true_phenomenon.x_opt[:, 1],
               true_phenomenon.f_opt,
               c=true_phenomenon.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x[:, 0], x[:, 1], y, c=y, vmin=vmin, vmax=vmax,
               label='Training Data')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], y_stars,
               c=y_stars, marker='+', vmin=vmin, vmax=vmax,
               label='Proposal Points')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, ei_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0,
                    antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.jet)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('EI')
    ax.set_title('Expected Improvement')
    ax.scatter(true_phenomenon.x_opt[:, 0], true_phenomenon.x_opt[:, 1],
               np.min(Z),
               c=true_phenomenon.f_opt, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x[:, 0], x[:, 1], np.min(Z), c=y, vmin=vmin, vmax=vmax,
               label='Training Data')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], np.min(Z),
               c=y_stars, marker='+', vmin=vmin, vmax=vmax,
               label='Proposal Points')

    x_star_history = np.concatenate((x_opt_init, x_stars), axis=0)
    y_star_history = np.append(y_opt_init, y_stars)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(np.arange(y_star_history.shape[0]), y_star_history, c='c',
            where='post', label='Proposed Points')
    ax.step(np.arange(y_star_history.shape[0]),
            np.maximum.accumulate(y_star_history), c='g', where='post',
            label='Current Optimum')
    ax.axhline(y=true_phenomenon.f_opt[0], color = 'k',
               label='True Optimum')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Observed Value')
    ax.set_title('Proposed Observations and Current Optimum v.s. Iterations')

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


if __name__ == "__main__":
    utils.misc.time_module(bayesian_optimisation)
    plt.show()