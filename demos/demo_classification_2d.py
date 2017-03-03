"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import log_loss


def create_spiral_data():

    np.random.seed(100)
    n = 200  # number of points per class
    d = 2  # dimensionality
    m = 3  # number of classes
    x = np.zeros((n * m, d))
    y = np.zeros(n * m, dtype='uint8')
    for j in range(m):
        ix = range(n * j, n * (j + 1))
        r = np.linspace(0.05, 1.05, n)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, n) + np.random.randn(n) * 0.2
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return 2*x, y[:, np.newaxis]


def true_phenomenon(x):

    x_min = -5
    x_max = +5
    d = 2

    n_class = 3
    r_max = 4

    c = (x_max - x_min) * np.random.rand(n_class, d) + x_min
    r = r_max * np.random.rand(n_class)
    y = np.zeros((x.shape[0], 1))
    for i in range(n_class):
        y[((x - c[i])**2).sum(axis=1) < r[i]**2] = i
    return y


def create_patch_data():

    seed = 200
    x_min = -5
    x_max = +5
    d = 2

    n = 100
    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = true_phenomenon(x)
    return x, y.astype(int)


def create_iris_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]  # we only take the first two features.
    x = (x - x.mean(axis=0))/x.std(axis=0)
    x = x / np.abs(x).max(axis=0)
    y = iris.target
    return x, y[:, np.newaxis]


def multiclass_classification(x, y, x_test, y_test):

    classes = np.unique(y)

    x_1_min, x_1_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    x_2_min, x_2_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    x_1_lim = (x_1_min, x_1_max)
    x_2_lim = (x_2_min, x_2_max)

    # Generate the query points
    n_query = 250
    x_1_array = np.linspace(*x_1_lim, num=n_query)
    x_2_array = np.linspace(*x_2_lim, num=n_query)
    x_1_mesh, x_2_mesh = np.meshgrid(x_1_array, x_2_array)
    x_query = np.array([x_1_mesh.ravel(), x_2_mesh.ravel()]).T

    # kernel = bake.kernels.s_gaussian
    # h_min = np.array([0.2, 0.25, 0.001])
    # h_max = np.array([2.0, 2.0, 1.0])
    # h_init = np.array([1.0, 1.0, 0.01])

    kernel = bake.kernels.s_gaussian
    h_min = np.array([0.5, 0.5, 0.000001])
    h_max = np.array([1.5, 3.0, 1.0])
    h_init = np.array([1.0, 0.5,  0.0001])
    # h = np.array([3.98869985e-01, 3.56441404e+00, -1.73898878e-05])
    # kec = bake.Classifier(kernel=kernel).fit(x, y, h=h)
    kec = bake.Classifier(kernel=kernel).fit(x, y,
                                             h_min=h_min,
                                             h_max=h_max,
                                             h_init=h_init)
    svc = SVC(probability=True).fit(x, y)
    gp_kernel = C() * RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=gp_kernel).fit(x, y)

    kec_p_query = kec.predict_proba(x_query)
    svc_p_query = svc.predict_proba(x_query)
    gpc_p_query = gpc.predict_proba(x_query)

    kec_y_query = bake.infer.classify(kec_p_query, classes=classes)
    svc_y_query = bake.infer.classify(svc_p_query, classes=classes)
    gpc_y_query = bake.infer.classify(gpc_p_query, classes=classes)

    kec_h_query = bake.infer.entropy(kec_p_query)
    svc_h_query = bake.infer.entropy(svc_p_query)
    gpc_h_query = bake.infer.entropy(gpc_p_query)

    kec_h_query_alt = kec.predict_entropy(x_query)

    # Convert to mesh
    kec_y_mesh = np.reshape(kec_y_query, (n_query, n_query))
    kec_h_mesh = np.reshape(kec_h_query, (n_query, n_query))
    kec_h_mesh_alt = np.reshape(kec_h_query_alt, (n_query, n_query))

    svc_y_mesh = np.reshape(svc_y_query, (n_query, n_query))
    svc_h_mesh = np.reshape(svc_h_query, (n_query, n_query))

    gpc_y_mesh = np.reshape(gpc_y_query, (n_query, n_query))
    gpc_h_mesh = np.reshape(gpc_h_query, (n_query, n_query))

    visualize_classifier('Kernel Embedding Classifier', x, y, x_test, y_test,
                         x_1_mesh, x_2_mesh, x_1_lim, x_2_lim,
                         y_mesh=kec_y_mesh, h_mesh=kec_h_mesh,
                         h_mesh_alt=kec_h_mesh_alt, p=kec_p_query,
                         entropy_method='(Clip Normalized)',
                         entropy_method_alt='(RKHS Expectation)')
    visualize_classifier('Support Vector Classifier', x, y, x_test, y_test,
                         x_1_mesh, x_2_mesh, x_1_lim, x_2_lim,
                         y_mesh=svc_y_mesh, h_mesh=svc_h_mesh, p=svc_p_query)
    visualize_classifier('Gaussian Process Classifier', x, y, x_test, y_test,
                         x_1_mesh, x_2_mesh, x_1_lim, x_2_lim,
                         y_mesh=gpc_y_mesh, h_mesh=gpc_h_mesh, p=gpc_p_query)

    kec_p = kec.predict_proba(x)
    svc_p = svc.predict_proba(x)
    gpc_p = gpc.predict_proba(x)

    kec_y = bake.infer.classify(kec_p, classes=classes)
    svc_y = bake.infer.classify(svc_p, classes=classes)
    gpc_y = bake.infer.classify(svc_p, classes=classes)

    kec_p_test = kec.predict_proba(x_test)
    svc_p_test = svc.predict_proba(x_test)
    gpc_p_test = gpc.predict_proba(x_test)

    kec_y_test = bake.infer.classify(kec_p_test, classes=classes)
    svc_y_test = bake.infer.classify(svc_p_test, classes=classes)
    gpc_y_test = bake.infer.classify(gpc_p_test, classes=classes)

    print('kec training accuracy: %.9f' % (np.mean(kec_y == y.ravel())))
    print('svc training accuracy: %.9f' % (np.mean(svc_y == y.ravel())))
    print('gpc training accuracy: %.9f' % (np.mean(gpc_y == y.ravel())))

    print('kec test accuracy: %.9f' % (np.mean(kec_y_test == y_test.ravel())))
    print('svc test accuracy: %.9f' % (np.mean(svc_y_test == y_test.ravel())))
    print('gpc test accuracy: %.9f' % (np.mean(gpc_y_test == y_test.ravel())))

    print('kec training log loss: %.9f' % log_loss(y.ravel(), kec_p))
    print('svc training log loss: %.9f' % log_loss(y.ravel(), svc_p))
    print('gpc training log loss: %.9f' % log_loss(y.ravel(), gpc_p))

    print('kec test log loss: %.9f' % log_loss(y_test.ravel(), kec_p_test))
    print('svc test log loss: %.9f' % log_loss(y_test.ravel(), svc_p_test))
    print('gpc test log loss: %.9f' % log_loss(y_test.ravel(), gpc_p_test))

    y_query = classes[:, np.newaxis]
    reverse_embedding = kec.reverse_embedding(y_query, x_query) # (n_query ** 2, 3)
    i_qrep = reverse_embedding.argmax(axis=0)
    x_qrep = x_query[i_qrep]
    y_qrep = kec.predict(x_qrep)

    reverse_embedding_images = np.reshape(np.swapaxes(reverse_embedding, 0, 1),
                                          (classes.shape[0], n_query, n_query))
    reverse_weights = kec.reverse_weights(y_query) # (n, 3)
    i_rep = reverse_weights.argmax(axis=0)
    x_rep = x[i_rep]
    y_rep = y[i_rep]

    for c, image in enumerate(reverse_embedding_images):
        plt.figure()
        plt.pcolormesh(x_1_mesh, x_2_mesh, image, cmap=cm.coolwarm)
        plt.colorbar()
        plt.contour(x_1_mesh, x_2_mesh, kec_y_mesh, colors='k',
                    label='Decision Boundaries')
        # plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
        # plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x',
        #             cmap=cm.jet, label='Test Data')
        plt.scatter(x[:, 0], x[:, 1], c=reverse_weights[:, c],
                    cmap=cm.coolwarm, label='Embedding Weights')
        plt.scatter(x_rep[c, 0], x_rep[c, 1], c=y_rep[c], marker='x', s=40,
                    cmap=cm.jet, label='Representative Sample')
        plt.scatter(x_qrep[c, 0], x_qrep[c, 1], c=y_qrep[c], marker='+', s=40,
                    cmap=cm.jet, label='Representative Prediction')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('Reverse Embedding for class %d' % c)
    #
    # full_directory = './'
    # utils.misc.save_all_figures(full_directory,
    #                             axis_equal=True, tight=True,
    #                             extension='eps', rcparams=None)


def visualize_classifier(name, x, y, x_test, y_test, x_1_mesh, x_2_mesh,
                         x_1_lim, x_2_lim,
                         y_mesh=None, h_mesh=None, h_mesh_alt=None, p=None,
                         entropy_method='',
                         entropy_method_alt=''):

    # Plot the predictions
    if y_mesh is not None:
        plt.figure()
        plt.pcolormesh(x_1_mesh, x_2_mesh, y_mesh, label='Predictions')
        plt.contour(x_1_mesh, x_2_mesh, y_mesh, colors='k',
                    label='Decision Boundaries')
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x',
                    cmap=cm.jet, label='Test Data')
        plt.colorbar()
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8,fancybox=True).get_frame().set_alpha(0.5)
        plt.title('%s: Predictions' % name)

    # Plot the entropy
    if h_mesh is not None:
        plt.figure()
        plt.pcolormesh(x_1_mesh, x_2_mesh, h_mesh, cmap=cm.coolwarm,
                       label='Entropy')
        plt.colorbar()
        plt.contour(x_1_mesh, x_2_mesh, y_mesh, colors='k',
                   label='Decision Boundaries')
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x',
                    cmap=cm.jet, label='Test Data')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('%s: Entropy %s' % (name, entropy_method))

    if h_mesh_alt is not None:
        plt.figure()
        plt.pcolormesh(x_1_mesh, x_2_mesh, h_mesh_alt, cmap=cm.coolwarm,
                       label='Entropy')
        plt.colorbar()
        plt.contour(x_1_mesh, x_2_mesh, y_mesh, colors='k',
                   label='Decision Boundaries')
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', cmap=cm.jet,
                    label='Test Data')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('%s: Entropy %s' % (name, entropy_method_alt))

    classes = np.unique(y)
    n_classes = classes.shape[0]
    n_query = x_1_mesh.shape[0]

    if p is not None:
        if n_classes == 3:

            plt.figure()
            Z = np.reshape(p, (n_query, n_query, n_classes))[:, :, ::-1]
            plt.imshow(Z, extent=(x_1_lim[0], x_1_lim[1], x_2_lim[0], x_2_lim[1]), origin="lower")
            plt.contour(x_1_mesh, x_2_mesh, y_mesh, colors='k', label='Decision Boundaries')
            plt.scatter(x[:, 0], x[:, 1], c=np.array(["b", "g", "r"])[y.ravel()], label='Training Data')
            plt.scatter(x_test[:, 0], x_test[:, 1],
                        c=np.array(["b", "g", "r"])[y_test.ravel()], marker='x',
                        cmap=cm.jet, label='Test Data')
            plt.xlim(x_1_lim)
            plt.ylim(x_2_lim)
            plt.xlabel('$x_{1}$')
            plt.ylabel('$x_{2}$')
            plt.title('%s: Prediction Probabilities' % name)
            plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                fontsize=8, fancybox=True).get_frame().set_alpha(0.5)

        elif n_classes <= 10:
            for c in np.unique(y):
                plt.figure()
                plt.pcolormesh(x_1_mesh, x_2_mesh,
                               np.reshape(p.T[c], (n_query, n_query)),
                               cmap=cm.coolwarm,
                               vmin=0,
                               vmax=1,
                               label='Probability')
                plt.colorbar()
                plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet,
                            label='Training Data')
                plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm.jet,
                            marker='x', label='Test Data')
                plt.xlim(x_1_lim)
                plt.ylim(x_2_lim)
                plt.xlabel('$x_{1}$')
                plt.ylabel('$x_{2}$')
                plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                           fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
                plt.title('%s: Probability for Class %d' % (name, c))

if __name__ == "__main__":

    x, y = create_spiral_data()
    test_size = 0.25
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=0)
    utils.misc.time_module(multiclass_classification,
                           x_train, y_train, x_test, y_test)
    print('Percentage of data withheld for testing: %f%%' % (100 * test_size))
    plt.show()
