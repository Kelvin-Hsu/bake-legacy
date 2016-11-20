"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
import numpy as np

seed = 200

def true_phenomenon(x):

    omega = 2.0
    phase_shift = 0.0
    amplitude = 1.0
    bias = 0.0
    wave_params = (omega, phase_shift, amplitude, bias)
    noise_level = 0.0

    # Generate output data
    f = utils.data.generate_waves(x, *wave_params,
                                  noise_level=noise_level, seed=seed)
    return (f > 0).astype(float)


def noisy_phenomenon(x):

    omega = 2.0
    phase_shift = 0.0
    amplitude = 1.0
    bias = 0.0
    wave_params = (omega, phase_shift, amplitude, bias)
    noise_level = 0.1

    # Generate output data
    f = utils.data.generate_waves(x, *wave_params,
                                  noise_level=noise_level, seed=seed)
    return (f > 0).astype(float)

def create_training_data():

    # Generate input data
    n = 40
    d = 1
    x_min = -5
    x_max = +5
    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = noisy_phenomenon(x)
    return x, y

def binary_classification(x, y, learn=False):

    # Set the hyperparameters
    if learn:
        return
    else:
        theta_x, zeta = 1.5, 0.01

    # QUERY POINT GENERATION

    # Generate the query points
    x_min = -5
    x_max = +5
    x_lim = (x_min - 2, x_max + 2)
    y_lim = (y.min() - 3, y.max() + 3)
    x_q, y_q, x_grid, y_grid = utils.visuals.uniform_query(x_lim, y_lim,
                                                           n_x_query=250,
                                                           n_y_query=250)
    y_q_true = true_phenomenon(x_q)

    # COMPUTE CONDITIONAL EMBEDDING

    # Conditional weights
    w_q = bake.infer.conditional_weights(x, theta_x, x_q, zeta=zeta)
    # Conditional embedding
    mu_y_xq = bake.infer.embedding(y, None, w=w_q,
                                   k=bake.kernels.kronecker_delta)
    mu_yq_xq = mu_y_xq(y_q)

    # Weights of the density
    # w_q_density = bake.infer.clip_normalize(w_q)
    # w_q_density = bake.infer.density_weights(w_q, y, theta_y)
    # w_q_density = bake.infer.clip_normalize(w_q)
    # w_q_density = bake.infer.softmax_normalize(w_q)

    w_q_density = w_q

    # MOMENT REGRESSION

    # Expectance and Variance
    yq_exp = bake.infer.expectance(y, w_q)[0]

    # MODE REGRESSION
    x_modes, y_modes = \
        bake.infer.search_modes_from_conditional_embedding(mu_yq_xq, x_q, y_q)

    # Plot the conditional embedding
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, mu_yq_xq)
    plt.scatter(x.ravel(), y.ravel(), c='k', label='Training Data')
    plt.plot(x_q.ravel(), yq_exp, c='b', label='Expectance')
    plt.plot(x_q.ravel(), y_q_true, c='g', label='Ground Truth')
    plt.scatter(x_modes.ravel(), y_modes.ravel(), c='w', edgecolors='w',
                label='Modes')
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
               fontsize=8,fancybox=True).get_frame().set_alpha(0.5)
    plt.title('Conditional Embedding')

if __name__ == "__main__":
    x, y = create_training_data()
    utils.misc.time_module(binary_classification, x, y, learn=False)
    plt.show()