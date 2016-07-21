"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
import numpy as np

def main():

    # Generate regression data
    # Change the noise level here
    n = 80
    x, y = utils.data.generate_two_waves(n = n, noise_level = 0.1, seed = 200)

    # Create joint data
    z = utils.data.joint_data(x, y)

    # Learn the embedding using the joint samples
    # hyper_min = ([0.01], [0.01], [0.08], [1e-3])
    # hyper_max = ([7.00], [5.00], [0.08], [1e-3])
    # hyper_init = None # ([2.0], [0.5], [1e-8], [1e-8])
    # yx = utils.data.joint_data(y, x)
    # theta_x, theta_y, zeta, sigma = bake.learn.conditional_embedding(x, y, yx,
    #     bake.kernels.dist(y, y),
    #     hyper_min, hyper_max, t_init_tuple = hyper_init, n = 2500)
    theta_x = np.array([3.0])
    theta_y = np.array([0.6])
    zeta = 0.1
    for sigma in np.logspace(-10, 2, 100):
        print(bake.learn.conditional_nlml(x, y, utils.data.joint_data(y, x), bake.kernels.dist(y, y), theta_x, theta_y, zeta, np.array([sigma])))
    theta = np.append(theta_x, theta_y)

    # This is the optimal joint embedding
    mu_z_optimal = bake.infer.embedding(z, theta)

    # This is the corresponding conditional embedding
    mu_yx_optimal = bake.infer.conditional_embedding(x, y, theta_x, theta_y,
        zeta = zeta)

    # Create a origin-centered uniform query space for visualisation
    x_margin = 0.5
    xq, yq, xq_grid, yq_grid, x_lim, y_lim = \
        utils.visuals.centred_uniform_query(x, y, 
            x_margin = x_margin, y_margin = 1, x_query = 150, y_query = 150)

    # Find the modes of the conditional embedding
    x_modes, y_modes = bake.infer.conditional_modes(mu_yx_optimal, xq, 
        [-y_lim], [+y_lim], n_modes = 3)

    # Create joint query points from the query space
    zq = utils.data.joint_data(xq_grid, yq_grid)

    # Evaluate the joint and conditional embeddings at the query points
    mu_zq_optimal = mu_z_optimal(zq)
    mu_yqxq_optimal = mu_yx_optimal(yq, xq)

    # Plot the joint embedding
    plt.figure(1)
    plt.scatter(xq_grid.ravel(), yq_grid.ravel(), c = mu_zq_optimal.ravel(), 
        s = 20, linewidths = 0)
    plt.scatter(x.ravel(), y.ravel(), c = 'k', label = 'Training Data')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.title('Optimal Joint Embedding')

    # Get rid of edge cases in the visualisation
    x_modes_points = x_modes.ravel()
    y_modes_points = y_modes.ravel()
    x_modes_points = x_modes_points[y_modes_points > -y_lim + 1]
    y_modes_points = y_modes_points[y_modes_points > -y_lim + 1]

    vmin, vmax = utils.visuals.find_bounded_extrema(mu_yqxq_optimal, 
        xq_grid, (-x_lim + x_margin, x_lim - x_margin))

    # Plot the conditional embedding
    plt.figure(2)
    plt.pcolormesh(xq_grid, yq_grid, mu_yqxq_optimal, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.scatter(x.ravel(), y.ravel(), c = 'k', label = 'Training Data')
    plt.scatter(x_modes_points, y_modes_points,
        s = 20, c = 'w', edgecolor = 'face', label = 'Mode Predictions')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.title('Optimal Conditional Embedding')

if __name__ == "__main__":
    utils.misc.time_module(main)
    plt.show()