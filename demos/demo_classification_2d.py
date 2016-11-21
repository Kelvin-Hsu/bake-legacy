"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

seed = 200
x_min = -5
x_max = +5
d = 2

def true_phenomenon(x):

    n_class = 5
    r_max = 4

    c = (x_max - x_min) * np.random.rand(n_class, d) + x_min
    r = r_max * np.random.rand(n_class)
    y = np.zeros((x.shape[0], 1))
    for i in range(n_class):
        y[((x - c[i])**2).sum(axis=1) < r[i]**2] = i
    return y

def create_training_data():

    n = 100
    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = true_phenomenon(x)
    return x, y

def multiclass_classification(x, y):

    # Set the hyperparameters
    theta_x, zeta = 0.4, 0.01

    # Generate the query points
    n_query = 250
    x_1_lim = (x_min, x_max)
    x_2_lim = (x_min, x_max)
    x_1_q, x_2_q, x_1_grid, x_2_grid = \
        utils.visuals.uniform_query(x_1_lim, x_2_lim, n_query, n_query)
    x_q = np.array([x_1_grid.ravel(), x_2_grid.ravel()]).T

    # Conditional weights
    w_q = bake.infer.conditional_weights(x, theta_x, x_q, zeta=zeta)

    # Weights of the density
    # w_q = bake.infer.clip_normalize(w_q)

    # Probabilistic computations
    classes = np.arange(np.unique(y).shape[0])
    p = np.array([bake.infer.expectance(y == c, w_q)[0] for c in classes])
    p_norm = bake.infer.softmax_normalize(p)
    # p_norm = p
    entropy = - np.sum(p_norm * np.log(p_norm), axis=0)
    i_pred = np.argmax(p, axis=0)
    y_pred = classes[i_pred]

    # Convert to mesh
    y_pred_mesh = np.reshape(y_pred, (n_query, n_query))
    entropy_mesh = np.reshape(entropy, (n_query, n_query))

    # Plot the predictions
    plt.figure()
    plt.pcolormesh(x_1_grid, x_2_grid, y_pred_mesh, label='Predictions')
    plt.scatter(x[:, 0], x[:, 1], c=y, label='Training Data')
    plt.colorbar()
    plt.xlim(x_1_lim)
    plt.ylim(x_2_lim)
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
               fontsize=8,fancybox=True).get_frame().set_alpha(0.5)
    plt.title('Kernel Embedding Classifier: Predictions')

    # Plot the entropy
    plt.figure()
    plt.pcolormesh(x_1_grid, x_2_grid, entropy_mesh, cmap=cm.coolwarm,
                   label='Entropy')
    plt.colorbar()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
    plt.xlim(x_1_lim)
    plt.ylim(x_2_lim)
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.title('Kernel Embedding Classifier: Entropy')

    for c in classes:
        plt.figure()
        plt.pcolormesh(x_1_grid, x_2_grid, np.reshape(p[c], (n_query, n_query)),
                       cmap=cm.coolwarm,
                       label='Probability')
        plt.colorbar()
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('Kernel Embedding Classifier: Probability for Class %d' % c)

if __name__ == "__main__":
    x, y = create_training_data()
    utils.misc.time_module(multiclass_classification, x, y)
    plt.show()