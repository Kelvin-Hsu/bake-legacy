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
x_1_lim = (x_min, x_max)
x_2_lim = (x_min, x_max)
d = 2


def create_spiral_data():

    np.random.seed(100)
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(
            N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return 5*X, y[:, np.newaxis]


def true_phenomenon(x):

    n_class = 3
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
    return x, y.astype(int)


def multiclass_classification(x, y):

    theta_x, zeta = np.array([4.23728543]),  0.644524255345
    # Generate the query points
    n_query = 250
    x_1_array = np.linspace(*x_1_lim, num=n_query)
    x_2_array = np.linspace(*x_2_lim, num=n_query)
    x_1_mesh, x_2_mesh = np.meshgrid(x_1_array, x_2_array)
    x_query = np.array([x_1_mesh.ravel(), x_2_mesh.ravel()]).T

    # classifier = bake.Classifier().fit(x, y, hyperparam=(theta_x, zeta))
    h_min = np.array([1.0, 0.01])
    h_max = np.array([2.0, 1.0])
    h_init = np.array([1.0, 0.1])
    classifier = bake.Classifier().fit(x, y, h_min=h_min, h_max=h_max,
                                       h_init=h_init, n_splits=10)
    y_query, p_query, h_query = classifier.infer(x_query)

    p_norm = bake.infer.clip_normalize(p_query)

    # Convert to mesh
    y_mesh = np.reshape(y_query, (n_query, n_query))
    h_mesh = np.reshape(h_query, (n_query, n_query))


    # Plot the predictions
    plt.figure()
    plt.pcolormesh(x_1_mesh, x_2_mesh, y_mesh, label='Predictions')
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
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
    plt.pcolormesh(x_1_mesh, x_2_mesh, h_mesh, cmap=cm.coolwarm,
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

    if classifier.n_classes == 3:

        plt.figure()
        Z = np.reshape(p_norm.T, (n_query, n_query, classifier.n_classes))
        # Z = [np.reshape(p_norm[c], (n_query, n_query)) for c in classes]
        # Z = np.swapaxes(Z, 2, 0)
        # print(Z.shape)
        plt.imshow(Z, extent=(x_min, x_max, x_min, x_max), origin="lower")
        plt.scatter(x[:, 0], x[:, 1], c=np.array(["r", "g", "b"])[y.ravel()], cmap=cm.jet, label='Training Data')
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.title('Class Probabilities')
        plt.tight_layout()
        # plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
        #     fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    else:
        for c in classifier.class_indices:
            plt.figure()
            plt.pcolormesh(x_1_mesh, x_2_mesh, np.reshape(p_norm[c], (n_query, n_query)),
                           cmap=cm.coolwarm,
                           vmin=0,
                           vmax=1,
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

    y_pred = classifier.infer(x)[0]
    print('training accuracy: %.9f' % (np.mean(y_pred == y.ravel())))

if __name__ == "__main__":
    x, y = create_spiral_data()
    utils.misc.time_module(multiclass_classification, x, y)
    plt.show()