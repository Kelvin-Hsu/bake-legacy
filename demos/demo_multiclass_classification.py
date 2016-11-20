"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
import numpy as np

seed = 100
colors = ['k', 'b', 'r', 'g']

def true_phenomenon(x):

    x_min = -5
    x_max = +5
    r_max = 1.5
    n_class = 5
    c = (x_max - x_min) * np.random.rand(n_class) + x_min
    r = r_max * np.random.rand(n_class)
    y = np.zeros(x.shape)
    for i in range(n_class):
        y[np.abs(x - c[i]) < r[i]] = i
    return y

def create_training_data():

    # Generate input data
    n = 100
    d = 1
    x_min = -5
    x_max = +5
    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = true_phenomenon(x)
    return x, y

def multiclass_classification(x, y, learn=False):

    # Set the hyperparameters
    if learn:
        return
    else:
        theta_x, zeta = 0.5, 0.01

    # Generate the query points
    x_min = -5
    x_max = +5
    x_lim = (x_min - 2, x_max + 2)
    y_lim = (y.min() - 3, y.max() + 3)
    x_q, y_q, x_grid, y_grid = utils.visuals.uniform_query(x_lim, y_lim,
                                                           n_x_query=250,
                                                           n_y_query=250)
    y_q_true = true_phenomenon(x_q)

    # Conditional weights
    w_q = bake.infer.conditional_weights(x, theta_x, x_q, zeta=zeta)
    # Conditional embedding
    mu_y_xq = bake.infer.embedding(y, None, w=w_q,
                                   k=bake.kernels.kronecker_delta)
    mu_yq_xq = mu_y_xq(y_q)

    # Weights of the density
    # w_q = bake.infer.clip_normalize(w_q)

    # Probabilistic computations
    classes = np.arange(np.unique(y).shape[0])
    p = np.array([bake.infer.expectance(y == c, w_q)[0] for c in classes])
    i_pred = np.argmax(p, axis=0)
    y_pred = classes[i_pred]

    # Mode Inference
    x_modes, y_modes = \
        bake.infer.search_modes_from_conditional_embedding(mu_yq_xq, x_q, y_q)

    # Plot the conditional embedding
    plt.figure()
    # plt.pcolormesh(x_grid, y_grid, mu_yq_xq)
    plt.scatter(x.ravel(), y.ravel(), c='k', label='Training Data')
    [plt.plot(x_q.ravel(), p[c], c=colors[c],
              label='Generalised Probability of Class %d' % c) for c in classes]
    # plt.plot(x_q.ravel(), y_q_true, c='c', marker='.', label='Ground Truth')
    plt.plot(x_q.ravel(), y_pred, c='c', label='Predictions')
    # plt.scatter(x_modes.ravel(), y_modes.ravel(), c='b', edgecolors='w',
    #             label='Modes')
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
               fontsize=8,fancybox=True).get_frame().set_alpha(0.5)
    plt.title('Conditional Embedding')

if __name__ == "__main__":
    x, y = create_training_data()
    utils.misc.time_module(multiclass_classification, x, y, learn=False)
    plt.show()