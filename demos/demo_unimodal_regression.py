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
    x, y = utils.data.generate_one_wave(n = 20, noise_level = 0.1, seed = 100)

    i_keep = np.logical_or(x < -3, x > 0)
    x, y = x[i_keep, np.newaxis], y[i_keep, np.newaxis]

    # Create joint data
    z = utils.data.joint_data(x, y)

    # Set the hyperparameters
    theta = [4,   0.5]
    theta_x, theta_y = theta[0], theta[1]

    # Create a origin-centered uniform query space for visualisation
    x_margin = 0.5
    xq, yq, xq_grid, yq_grid, x_lim, y_lim = \
        utils.visuals.centred_uniform_query(x, y,
            x_margin = x_margin, y_margin = 1, x_query = 300, y_query = 300)

    # # Find the modes of the conditional embedding
    # x_modes, y_modes = bake.infer.conditional_modes(mu_yx, xq,
    #     [-y_lim], [+y_lim], n_modes = 2)

    # Create joint query points from the query space
    zq = utils.data.joint_data(xq_grid, yq_grid)

    # Conditional weights
    wq = bake.infer.conditional_weights(x, y, theta_x, theta_y, xq, zeta=0.01)
    mu_yqxq = bake.infer.evaluate_embedding(yq, y, theta_y, wq)

    # Expectance and Covariance
    yq_exp = bake.infer.expectance(y, wq)[0]
    yq_var = bake.infer.variance(y, wq)[0]
    yq_std = np.sqrt(yq_var)

    # Plot the conditional embedding
    plt.figure(1)
    plt.pcolormesh(xq_grid, yq_grid, mu_yqxq)
    plt.scatter(x.ravel(), y.ravel(), c = 'k', label = 'Training Data')
    # plt.scatter(x_modes.ravel(), y_modes.ravel(),
    #     s = 20, c = 'w', edgecolor = 'face', label = 'Mode Predictions')
    plt.plot(xq.ravel(), yq_exp, c = 'k')
    plt.fill_between(xq.ravel(), yq_exp - 2 * yq_std, yq_exp + 2 * yq_std,
                     facecolor=(0.9, 0.9, 0.9), edgecolor=(0.0, 0.0, 0.0),
                     interpolate=True, alpha=0.5)
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.title('Optimal Conditional Embedding')

if __name__ == "__main__":
    utils.misc.time_module(main)
    plt.show()