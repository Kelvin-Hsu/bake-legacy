"""
Demonstration of simple kernel embeddings.
"""
from manifold.regression import GPRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

import utils
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import time

seed = 20
x_min = np.array([-5, 0])
x_max = np.array([10, 15])
d = 2


def true_phenomenon(x):
    """
    Branin-Hoo Function.

    Parameters
    ----------
    x : numpy.ndarray
        The input to the function of size (n, 2)

    Returns
    -------
    numpy.ndarray
        The output from the function of size (n, )
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return -f

true_phenomenon.optima = np.array([[-np.pi, 12.275],
                                   [np.pi, 2.275],
                                   [9.42478, 2.475]])

true_phenomenon.optimal_value = true_phenomenon(true_phenomenon.optima)

def noisy_phenomenon(x, sigma=1.0):

    f = true_phenomenon(x)
    e = np.random.normal(loc=0, scale=sigma, size=x.shape[0])
    y = f + e
    return y


def create_training_data(n=100, sigma=1.0):

    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = noisy_phenomenon(x, sigma=1.0)
    return x, y


def bayesian_optimisation():

    # Create training data
    sigma = 0.5
    x, y = create_training_data(n=10, sigma=sigma)

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

    x_stars = []
    y_stars = []

    # Predictions
    gpr.fit(x, y)
    for i in range(20):

        if i > 0:
            # x = np.concatenate((x, x_propose), axis=0)
            # y = np.append(y, y_observe)
            gpr.update(x_star, y_star)
            if y_star > y_opt:
                x_opt = x_star
                y_opt = y_star

        y_q_exp, y_q_std = gpr.predict(x_q, return_std=True)
        y_q_exp_mesh = np.reshape(y_q_exp, (n_query, n_query))

        # Expected Improvement
        ei = gpr_expected_improvement(y_q_exp, y_q_std, y_opt)
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
    X, Y, Z = x_1_mesh, x_2_mesh, y_q_exp_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('$E[f]$')
    ax.set_title('Predictions')
    ax.scatter(true_phenomenon.optima[:, 0], true_phenomenon.optima[:, 1],
               true_phenomenon.optimal_value,
               c=true_phenomenon.optimal_value, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x[:, 0], x[:, 1], y, c=y, vmin=vmin, vmax=vmax,
               label='Training Data')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], y_stars,
               c=y_stars, marker='+', vmin=vmin, vmax=vmax,
               label='Proposal Points')
    # ax.view_init(90, 0)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = x_1_mesh, x_2_mesh, ei_mesh
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.set_zlabel('EI')
    ax.set_title('Expected Improvement')
    ax.scatter(true_phenomenon.optima[:, 0], true_phenomenon.optima[:, 1],
               np.min(Z),
               c=true_phenomenon.optimal_value, s=40, marker='x',
               vmin=vmin, vmax=vmax, label='Optima')
    ax.scatter(x[:, 0], x[:, 1], np.min(Z), c=y, vmin=vmin, vmax=vmax,
               label='Training Data')
    ax.scatter(x_stars[:, 0], x_stars[:, 1], np.min(Z),
               c=y_stars, marker='+', vmin=vmin, vmax=vmax,
               label='Proposal Points')
    # ax.view_init(90, 0)

    x_star_history = np.concatenate((x_opt_init, x_stars), axis=0)
    y_star_history = np.append(y_opt_init, y_stars)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(y_star_history.shape[0]), y_star_history, label='Proposed Points')
    ax.axhline(y=true_phenomenon.optimal_value[0], color = 'k', label='True Optimum')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Observed Value')
    ax.set_title('Proposed Observations v.s. Iterations')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(y_star_history.shape[0]), np.maximum.accumulate(y_star_history), label='Optima')
    ax.axhline(y=true_phenomenon.optimal_value[0], color = 'k', label='True Optimum')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Optima')
    ax.set_title('Optimum v.s. Iterations')


def gpr_expected_improvement(mu, std, best):

    # The difference between our predictions and the observed maximum so far
    diff = mu - best
    abs_diff = np.abs(diff)
    clip_diff = np.clip(diff, 0, np.inf)
    return clip_diff + std*norm.pdf(diff/std) - abs_diff*norm.cdf(-abs_diff/std)

if __name__ == "__main__":
    utils.misc.time_module(bayesian_optimisation)
    plt.show()