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
    return x, y


def split_one(x, y, i, n):
    x_q = x[[i]]
    y_q = y[[i]]
    others = np.arange(n) != i
    x_t = x[others]
    y_t = y[others]
    return x_t, y_t, x_q, y_q

def train_classifier(x, y):

    n = x.shape[0]
    classes = np.arange(np.unique(y).shape[0])
    # theta, zeta = 1.0, 1e-4
    #
    # hyperparam = np.array([theta, zeta])

    def objective(hyperparam, i):

        theta, zeta = hyperparam[0], hyperparam[1]

        # p_sum = 0
        # for j in range(n):
        #     x_t, y_t, x_q, y_q = split_one(x, y, j, n)
        #     w_q = bake.infer.conditional_weights(x_t, theta, x_q, zeta=zeta)
        #     p = bake.infer.expectance(y_t == y_q[0, 0], w_q)[0][0]
        #     p_sum += p
        #
        # value = p_sum / n
        x_q = x
        w_q = bake.infer.conditional_weights(x, theta, x_q, zeta=zeta)
        p = np.array([bake.infer.expectance(y == c, w_q)[0] for c in classes])

        i_pred = np.argmax(p, axis=0)
        y_pred = classes[i_pred]
        loss = y != y_pred
        # value = np.mean(y.ravel() == y_pred) - zeta ** 2 * w_q.sum() / n

        p_want = p[y.ravel(), np.arange(n)]
        value = np.mean(p_want * loss)
        print(i, value)
        return value

    n_theta_q = 10
    n_zeta_q = 10
    theta_q = np.linspace(0.5, 6, num=n_theta_q)
    zeta_q = np.logspace(-6, 1, num=n_zeta_q)
    theta_q_grid, zeta_q_grid = np.meshgrid(theta_q, zeta_q)
    hyperparams = np.array([theta_q_grid.ravel(), zeta_q_grid.ravel()]).T
    objective_values = [objective(h, i) for i, h in enumerate(hyperparams)]
    print(hyperparams.shape)
    i_best = np.argmax(objective_values)
    hyperparam_best = hyperparams[i_best]

    plt.figure()
    values_grid = np.reshape(objective_values, (n_zeta_q, n_theta_q))
    print(theta_q_grid.shape, zeta_q_grid.shape, values_grid.shape)
    plt.pcolormesh(theta_q_grid, np.log10(zeta_q_grid), values_grid)
    plt.title('Learning Objective')
    plt.xlabel(r'\theta')
    plt.ylabel(r'\zeta')
    plt.colorbar()
    return hyperparam_best[0], hyperparam_best[1]


def train_bayes_classifier(x, y):

    n = x.shape[0]
    classes = np.arange(np.unique(y).shape[0])
    x_classes = [x[y == c] for c in classes]
    print(x_classes[0].shape)
    m = classes.shape[0]


    def objective(theta_x):
        p_classes = [bake.infer.embedding(x_c, theta_x) for x_c in x_classes]
        prior = 1/m
        # [print(y.ravel()[i]) for i in range(n)]
        print(x[[0]])
        print(p_classes[y.ravel()[0]](x[[0]]))
        log_marginal_likelihood = np.prod([p_classes[y.ravel()[i]](x[[i], :]) * prior for i in range(n)])
        return log_marginal_likelihood


    n_theta_q = 50
    theta_q = np.linspace(0.5, 6, num=n_theta_q)
    objective_values = [objective(theta) for theta in theta_q]
    i_best = np.argmax(objective_values)
    theta_x = theta_q[i_best]
    return theta_x


def log(x):
    answer = np.log(x)
    answer[x <= 0] = 0
    return answer


def train_cross_val(x, y):
    from sklearn.model_selection import KFold
    n = x.shape[0]
    classes = np.arange(np.unique(y).shape[0])

    k = 5
    X = ["a", "b", "c", "d", "e"]
    kf = KFold(n_splits=10)

    def objective(hyperparam, j):

        theta, zeta = hyperparam[0], hyperparam[1]

        total_value = 0
        for train, test in kf.split(x):
            # print(train, test)
            X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
            w_q = bake.infer.conditional_weights(X_train, theta, X_test,
                                                 zeta=zeta)
            p = np.array(
                [bake.infer.expectance(y_train == c, w_q)[0] for c in classes])
            i_pred = np.argmax(p, axis=0)
            y_pred = classes[i_pred]
            loss = y_test != y_pred
            p_want = p[y_test.ravel(), np.arange(y_test.shape[0])]
            value = np.mean(np.clip(p_want, 0, np.inf) * loss)
            total_value += value
            # plt.scatter(X_train[:, 0], X_train[:, 1], c='g')
            # plt.scatter(X_test[:, 0], X_test[:, 1], c='c')
            # plt.show()
        print(j, total_value/k)
        return total_value/k

    n_theta_q = 20
    n_zeta_q = 20
    theta_q = np.linspace(0.1, 6, num=n_theta_q)
    zeta_q = np.logspace(-6, 1, num=n_zeta_q)
    theta_q_grid, zeta_q_grid = np.meshgrid(theta_q, zeta_q)
    hyperparams = np.array([theta_q_grid.ravel(), zeta_q_grid.ravel()]).T
    objective_values = [objective(h, i) for i, h in enumerate(hyperparams)]
    print(hyperparams.shape)
    i_best = np.argmin(objective_values)
    hyperparam_best = hyperparams[i_best]

    plt.figure()
    values_grid = np.reshape(objective_values, (n_zeta_q, n_theta_q))
    print(theta_q_grid.shape, zeta_q_grid.shape, values_grid.shape)
    plt.pcolormesh(theta_q_grid, np.log10(zeta_q_grid), values_grid)
    plt.title('Learning Objective')
    plt.xlabel(r'\theta')
    plt.ylabel(r'\zeta')
    plt.colorbar()
    return hyperparam_best[0], hyperparam_best[1]

def multiclass_classification(x, y):

    # Set the hyperparameters
    theta_x, zeta = 2.77768405, 0.05
    # theta_x, zeta = train_classifier(x, y)
    # theta_x = train_bayes_classifier(x, y)
    # theta_x, zeta = train_cross_val(x, y)
    # hyper_min = ([0.1], [0.01], [1e-2])
    # hyper_max = ([5.], [5.], [5.])
    # theta_x, psi_x, sigma_x = bake.learn.optimal_joint_embedding(x,
    #                                                              hyper_min,
    #                                                              hyper_max)
    # print(theta_x, psi_x, sigma_x)
    # zeta = sigma_x
    print(theta_x, zeta)

    # Generate the query points
    n_query = 250
    x_1_array = np.linspace(*x_1_lim, num=n_query)
    x_2_array = np.linspace(*x_2_lim, num=n_query)
    x_1_mesh, x_2_mesh = np.meshgrid(x_1_array, x_2_array)
    x_query = np.array([x_1_mesh.ravel(), x_2_mesh.ravel()]).T

    classifier = bake.Classifier().fit(x, y, hyperparam=(theta_x, zeta))
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