"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

seed = 200
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

def create_training_data(n=100, sigma=1.0):

    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    f = true_phenomenon(x)
    e = np.random.normal(loc=0, scale=sigma, size=n)
    y = f + e
    return x, y


def bayesian_optimisation():

    # Create training data
    x, y = create_training_data(n=100, sigma=5.0)

    # Set the hyperparameters
    theta_x, theta_y, zeta = 1.0, 5.0, 0.01

    # Generate the query points
    n_query = 250
    x_1_lim = (x_min[0], x_max[0])
    x_2_lim = (x_min[1], x_max[1])
    x_1_q, x_2_q, x_1_grid, x_2_grid = \
        utils.visuals.uniform_query(x_1_lim, x_2_lim, n_query, n_query)
    x_q = np.array([x_1_grid.ravel(), x_2_grid.ravel()]).T

    # Ground Truth
    y_q_true = true_phenomenon(x_q)

    # Conditional weights
    w_q = bake.infer.conditional_weights(x, theta_x, x_q, zeta=zeta)

    # # Weights of the density
    # w_q_norm = bake.infer.clip_normalize(w_q)

    # Probabilistic computations
    y_q_exp = bake.infer.expectance(y, w_q)
    x_star = x_q[y_q_exp.argmax()]

    # Convert to mesh
    y_q_true_mesh = np.reshape(y_q_true, (n_query, n_query))
    y_q_exp_mesh = np.reshape(y_q_exp, (n_query, n_query))

    # Plot the predictions
    plt.figure()
    plt.pcolormesh(x_1_grid, x_2_grid, y_q_true_mesh)
    plt.scatter(x[:, 0], x[:, 1], c=y, label='Training Data')
    plt.colorbar()
    plt.scatter(true_phenomenon.optima[:, 0],
                true_phenomenon.optima[:, 1],
                c='w', marker='x', label='True Optima')
    plt.xlim(x_1_lim)
    plt.ylim(x_2_lim)
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
               fontsize=8,fancybox=True).get_frame().set_alpha(0.5)
    plt.title('Ground Truth')

    # Plot the entropy
    plt.figure()
    plt.pcolormesh(x_1_grid, x_2_grid, y_q_exp_mesh)
    plt.scatter(x[:, 0], x[:, 1], c=y, label='Training Data')
    plt.colorbar()
    plt.scatter(true_phenomenon.optima[:, 0],
                true_phenomenon.optima[:, 1],
                c='w', marker='x', label='True Optima')
    plt.scatter(x_star[0], x_star[1],
                c='k', marker='x', label='Current Optima')
    plt.xlim(x_1_lim)
    plt.ylim(x_2_lim)
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.title('Kernel Embedding Regressor')

if __name__ == "__main__":
    utils.misc.time_module(bayesian_optimisation)
    plt.show()