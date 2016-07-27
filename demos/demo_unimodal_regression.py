"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
import numpy as np


def main(learn=True):

    # OBTAIN TRAINING DATA

    # Generate input data
    seed = 200
    n = 80
    d = 1
    x_min = -5
    x_max = +5
    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)

    # Generate output data
    # omega = np.array([1.0, 1.0])
    # phase_shift = np.array([0.0, 0.5])
    # amplitude = np.array([2.0, 1.0])
    # bias = np.array([-3., 1.])
    omega = 1.0
    phase_shift = 0.0
    amplitude = 2.0
    bias = 1.0
    noise_level = 0.4
    y = utils.data.generate_waves(x, omega, phase_shift, amplitude, bias,
                                  noise_level=noise_level, seed=seed)

    # Artificially remove some data
    # i_keep = np.logical_or(x < -2, x > 1)
    # x, y = x[i_keep, np.newaxis], y[i_keep, np.newaxis]
    #
    # i_keep = np.logical_or(x < 2, x > 3)
    # x, y = x[i_keep, np.newaxis], y[i_keep, np.newaxis]

    # HYPERPARAMETER LEARNING OR SETTING

    # Set the hyperparameters
    if learn:
        hyper_min = ([0.01], [0.01], [0.0008], [1e-6], [1e-6])
        hyper_max = ([7.00], [5.00], [0.08], [1e-3], [1e-3])
        theta_x, theta_y, zeta, psi, sigma = bake.learn.optimal_conditional_embedding(
            x, y, hyper_min, hyper_max)
        print(theta_x, theta_y, zeta, psi, sigma)
    else:
        theta_x, theta_y, zeta = 3.0, 2.0, 0.01


    # QUERY POINT GENERATION

    # Generate the query points
    x_lim = (x_min - 1, x_max + 2)
    y_lim = (y.min() - 3, y.max() + 3)
    x_q, y_q, x_grid, y_grid = utils.visuals.uniform_query(x_lim, y_lim,
                                                           n_x_query=250,
                                                           n_y_query=250)

    # COMPUTE CONDITIONAL EMBEDDING

    # Conditional weights
    w_q = bake.infer.conditional_weights(x, theta_x, x_q, zeta=zeta)

    # Conditional embedding
    mu_y_xq = bake.infer.embedding(y, theta_y, w=w_q)
    mu_yq_xq = mu_y_xq(y_q)

    # Weights of the density
    # w_q_density = bake.infer.clip_normalise(w_q)
    # w_q_density = bake.infer.density_weights(w_q, y, theta_y)

    # QUANTILE REGRESSION

    # Perform quantile regression
    probabilities = np.arange(0.1, 1.0, 0.1)
    n_quantiles, = probabilities.shape
    y_quantiles = bake.infer.multiple_quantile_regression(probabilities,
                                                          w_q, y, theta_y)

    # MOMENT REGRESSION

    # Expectance and Variance
    yq_exp = bake.infer.expectance(y, w_q)[0]
    # yq_var = bake.infer.variance(y, w_q)[0]
    # yq_std = np.sqrt(yq_var)
    # yq_lb = yq_exp - 2 * yq_std
    # yq_ub = yq_exp + 2 * yq_std

    colors = [(c - 0.1, c + 0.1, c + 0.1) for c in probabilities]

    # Plot the conditional embedding
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, mu_yq_xq)
    plt.scatter(x.ravel(), y.ravel(), c='k', label='Training Data')
    plt.plot(x_q.ravel(), yq_exp, c='b', label='Expectance')
    [plt.plot(x_q.ravel(), y_quantiles[i].ravel(),
              c=colors[i], label='p = %f' % probabilities[i])
     for i in np.arange(n_quantiles)]
    # plt.fill_between(x_q.ravel(), yq_lb, yq_ub,
    #                  facecolor=(0.9, 0.9, 0.9), edgecolor=(0.0, 0.0, 0.0),
    #                  interpolate=True, alpha=0.5)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    if learn:
        plt.title('Optimal Conditional Embedding')
    else:
        plt.title('Original Conditional Embedding')


if __name__ == "__main__":
    utils.misc.time_module(main, learn=False)
    utils.misc.time_module(main, learn=True)
    plt.show()