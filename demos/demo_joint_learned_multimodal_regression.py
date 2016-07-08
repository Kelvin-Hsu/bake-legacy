"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt

def main():

    # Generate regression data
    # Change the noise level here
    x, y = utils.data.generate_two_waves(n = 80, noise_level = 0.0, seed = 100)

    # Create joint data
    z = utils.data.joint_data(x, y)

    # Learn the embedding using the joint samples
    hyper_min = ([0.01, 0.01], [0.01, 0.01], [0.0])
    hyper_max = ([5., 2.], [20., 20.], [0.1])
    theta, _, _ = bake.learn.embedding(z, hyper_min, hyper_max, n = 1500)
    theta_x, theta_y = theta[[0]], theta[[1]]

    # This is the optimal joint embedding
    mu_z_optimal = bake.infer.embedding(z, theta)

    # This is the corresponding conditional embedding
    mu_yx_optimal = bake.infer.conditional_embedding(x, y, theta_x, theta_y)

    # Create a origin-centered uniform query space for visualisation
    xq, yq, xq_grid, yq_grid, x_lim, y_lim = \
        utils.visuals.centred_uniform_query(x, y, 
            x_margin = 0, y_margin = 1, x_query = 150, y_query = 150)

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

    # Plot the conditional embedding
    plt.figure(2)
    plt.pcolormesh(xq_grid, yq_grid, mu_yqxq_optimal)
    plt.scatter(x.ravel(), y.ravel(), c = 'k', label = 'Training Data')
    plt.scatter(x_modes.ravel(), y_modes.ravel(), 
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