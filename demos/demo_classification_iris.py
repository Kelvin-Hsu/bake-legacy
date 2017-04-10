"""
Demonstration of simple kernel embeddings.
"""
import bake, cake
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
import datetime
import os

now = datetime.datetime.now()
now_string = '%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                    now.hour, now.minute, now.second)


def load_iris_data(a, b, normalize_features=True):
    """
    Load the iris dataset with two feature dimensions.

    Parameters
    ----------
    a : int
        First feature dimension
    b : int
        Second feature dimension
    normalize_features : bool, optional
        To normalize the features or not

    Returns
    -------
    numpy.ndarray
        The features (n, 2)
    numpy.ndarray
        The target labels (n, 1)

    """
    iris = datasets.load_iris()
    x = iris.data[:, [a, b]]
    if normalize_features:
        x -= np.min(x, axis=0)
        x /= np.max(x, axis=0)
    y = iris.target
    return x, y[:, np.newaxis]


def search_svc(x, y, kernel, hyper_search, k=1):
    losses_stack = np.zeros((k, hyper_search.shape[0]))
    for random_state in range(k):
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size,
                                                            random_state=0)
        losses_stack[random_state] = search_svc_test(x_train, y_train,
                                                     x_test, y_test,
                                                     kernel, hyper_search,
                                                     return_loss=True)
    losses = losses_stack.sum(axis=0)
    i = np.argmin(losses)
    print('\tSVC Losses: ', losses)
    print('\tSVC Lowest Loss: ', losses[i])
    return hyper_search[i]


def search_svc_test(x, y, x_test, y_test, kernel, hyper_search,
                    return_loss=False):
    print('\tSVC Kernel Parameter Search over %d possibilities'
          % hyper_search.shape[0])
    losses = [log_loss(y_test.ravel(),
                       SVC(kernel=lambda x1, x2: kernel(x1, x2, hyper),
                           probability=True).fit(x, y).predict_proba(x_test))
              for hyper in hyper_search]
    i = np.argmin(losses)
    print('\tSVC Losses: ', losses)
    print('\tSVC Lowest Loss: ', losses[i])
    if return_loss:
        return losses
    else:
        return hyper_search[i]


def iris_classification(x_train, y_train, x_test, y_test,
                        name='', directory='./'):

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    classes = np.unique(y_train)
    n_class = classes.shape[0]

    x_1_min, x_1_max = x_train[:, 0].min() - .1, x_train[:, 0].max() + .1
    x_2_min, x_2_max = x_train[:, 1].min() - .1, x_train[:, 1].max() + .1
    x_1_lim = (x_1_min, x_1_max)
    x_2_lim = (x_2_min, x_2_max)

    # Generate the query points
    n_query = 250
    x_1_array = np.linspace(*x_1_lim, num=n_query)
    x_2_array = np.linspace(*x_2_lim, num=n_query)
    x_1_mesh, x_2_mesh = np.meshgrid(x_1_array, x_2_array)
    x_query = np.array([x_1_mesh.ravel(), x_2_mesh.ravel()]).T

    # Specify the kernel and kernel parameter setup
    kernel = bake.kernels.s_gaussian

    # Train the KEC
    theta_init = np.array([1.0, 1.0, 1.0])
    zeta_init = 1e-4
    learning_rate = 0.01
    grad_tol = 0.01
    n_sgd_batch = None
    kec = cake.KEC(
        kernel=cake.kernels.s_gaussian).fit(x_train, y_train,
                                            theta=theta_init, zeta=zeta_init,
                                            learning_rate=learning_rate,
                                            grad_tol=grad_tol,
                                            n_sgd_batch=n_sgd_batch)

    kec_h = np.append(kec.theta_train, np.sqrt(kec.zeta_train))
    kec_complexity = kec.complexity_train
    kec_mean_sum_probability = kec.msp_train
    kec_steps_train = kec.steps_train
    kec_train_history = kec.train_history
    # kec_p_tf = kec.predict_proba(x_train)
    # kec_p_test_tf = kec.predict_proba(x_test)
    # kec_x_modes = kec.input_mode()
    # kec_y_modes = kec.predict(kec_x_modes)
    kec = bake.Classifier(kernel=kernel).fit(x_train, y_train, h=kec_h)
    print('KEC Hyperparameters: ', kec_h)

    # Train the SVC
    # svc_hyper_search = np.array([[s, l]
    #                              for s in np.linspace(1, 100, 50)
    #                              for l in np.linspace(0.01, 10, 50)])
    # svc_h = search_svc_test(x_train, y_train, x_test, y_test,
    #                             kernel, svc_hyper_search)
    svc_h = np.array([1., 1., 1.])
    svc = SVC(kernel=lambda x1, x2: kernel(x1, x2, svc_h),
              probability=True).fit(x_train, y_train)
    print('SVC Hyperparameters: ', svc_h)

    # Train the GPC
    gpc_kernel = C() * RBF()
    gpc = GaussianProcessClassifier(kernel=gpc_kernel).fit(x_train, y_train)
    gpc_h = gpc.kernel_.theta
    print('GPC Hyperparameters: ', gpc_h)

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

    visualize_classifier('Kernel Embedding Classifier',
                         x_train, y_train, x_test, y_test,
                         x_1_mesh, x_2_mesh, x_1_lim, x_2_lim,
                         y_mesh=kec_y_mesh, h_mesh=kec_h_mesh,
                         h_mesh_alt=kec_h_mesh_alt, p=kec_p_query,
                         entropy_method='(Clip Normalized)',
                         entropy_method_alt='(RKHS Expectation)')
    visualize_classifier('Support Vector Classifier',
                         x_train, y_train, x_test, y_test,
                         x_1_mesh, x_2_mesh, x_1_lim, x_2_lim,
                         y_mesh=svc_y_mesh, h_mesh=svc_h_mesh, p=svc_p_query)
    visualize_classifier('Gaussian Process Classifier',
                         x_train, y_train, x_test, y_test,
                         x_1_mesh, x_2_mesh, x_1_lim, x_2_lim,
                         y_mesh=gpc_y_mesh, h_mesh=gpc_h_mesh, p=gpc_p_query)

    # Predict probabilities on the training set
    kec_p = kec.predict_proba(x_train)
    svc_p = svc.predict_proba(x_train)
    gpc_p = gpc.predict_proba(x_train)

    # Predict targets on the training set
    kec_y = bake.infer.classify(kec_p, classes=classes)
    svc_y = bake.infer.classify(svc_p, classes=classes)
    gpc_y = bake.infer.classify(svc_p, classes=classes)

    # Predict probabilities on the testing set
    kec_p_test = kec.predict_proba(x_test)
    svc_p_test = svc.predict_proba(x_test)
    gpc_p_test = gpc.predict_proba(x_test)

    # Predict targets on the testing set
    kec_y_test = bake.infer.classify(kec_p_test, classes=classes)
    svc_y_test = bake.infer.classify(svc_p_test, classes=classes)
    gpc_y_test = bake.infer.classify(gpc_p_test, classes=classes)

    # Report the performance of each classifier
    kec_train_accuracy = np.mean(kec_y == y_train.ravel())
    svc_train_accuracy = np.mean(svc_y == y_train.ravel())
    gpc_train_accuracy = np.mean(gpc_y == y_train.ravel())
    kec_test_accuracy = np.mean(kec_y_test == y_test.ravel())
    svc_test_accuracy = np.mean(svc_y_test == y_test.ravel())
    gpc_test_accuracy = np.mean(gpc_y_test == y_test.ravel())
    kec_train_log_loss = log_loss(y_train.ravel(), kec_p)
    svc_train_los_loss = log_loss(y_train.ravel(), svc_p)
    gpc_train_los_loss = log_loss(y_train.ravel(), gpc_p)
    kec_test_log_loss = log_loss(y_test.ravel(), kec_p_test)
    svc_test_los_loss = log_loss(y_test.ravel(), svc_p_test)
    gpc_test_los_loss = log_loss(y_test.ravel(), gpc_p_test)
    print('kec training accuracy: %.9f' % kec_train_accuracy)
    print('svc training accuracy: %.9f' % svc_train_accuracy)
    print('gpc training accuracy: %.9f' % gpc_train_accuracy)
    print('kec test accuracy: %.9f' % kec_test_accuracy)
    print('svc test accuracy: %.9f' % svc_test_accuracy)
    print('gpc test accuracy: %.9f' % gpc_test_accuracy)
    print('kec training log loss: %.9f' % kec_train_log_loss)
    print('svc training log loss: %.9f' % svc_train_los_loss)
    print('gpc training log loss: %.9f' % gpc_train_los_loss)
    print('kec test log loss: %.9f' % kec_test_log_loss)
    print('svc test log loss: %.9f' % svc_test_los_loss)
    print('gpc test log loss: %.9f' % gpc_test_los_loss)

    y_query = classes[:, np.newaxis]
    reverse_embedding = kec.reverse_embedding(y_query, x_query)
    i_qrep = reverse_embedding.argmax(axis=0)
    x_qrep = x_query[i_qrep]
    y_qrep = kec.predict(x_qrep)

    reverse_embedding_images = np.reshape(np.swapaxes(reverse_embedding, 0, 1),
                                          (classes.shape[0], n_query, n_query))
    reverse_weights = kec.reverse_weights(y_query) # (n, 3)
    i_rep = reverse_weights.argmax(axis=0)
    x_rep = x_train[i_rep]
    y_rep = y_train[i_rep]

    for c, image in enumerate(reverse_embedding_images):
        plt.figure()
        plt.pcolormesh(x_1_mesh, x_2_mesh, image, cmap=cm.coolwarm)
        plt.colorbar()
        plt.contour(x_1_mesh, x_2_mesh, kec_y_mesh, colors='k',
                    label='Decision Boundaries')
        plt.scatter(x_train[:, 0], x_train[:, 1], c=reverse_weights[:, c], s=40,
                    cmap=cm.coolwarm, label='Embedding Weights')
        plt.scatter(x_rep[c, 0], x_rep[c, 1], c=y_rep[c], marker='x', s=40,
                    cmap=cm.jet, label='Representative Sample')
        plt.scatter(x_qrep[c, 0], x_qrep[c, 1], c=y_qrep[c], marker='+', s=40,
                    cmap=cm.jet, label='Enumerated Representative Prediction')
        # plt.scatter(kec_x_modes[c, 0], kec_x_modes[c, 1], c=kec_y_modes[c],
        #             marker='*', s=40,
        #             cmap=cm.jet, label='Learned Representative Prediction')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('Reverse Embedding for class %d' % c)

    # Plot Training History
    # self.train_history = {'iterations': _train_history[0, :],
    #                       'complexity': _train_history[1, :],
    #                       'accuracy': _train_history[2, :],
    #                       'cross_entropy_loss': _train_history[3, :],
    #                       'gradient_norm': _train_history[4, :],
    #                       'kernel_hypers': _train_history[5:-1, :],
    #                       'regularisation': _train_history[-1, :]}
    th = kec_train_history

    hyper_history = \
        np.concatenate((th['kernel_hypers'],
                        np.sqrt(th['regularisation'][:, np.newaxis])), axis=1)
    ph = performance_history(hyper_history, kernel, classes,
                             x_train, y_train, x_test, y_test)

    fig = plt.figure()
    plt.subplot(6, 1, 1)
    plt.plot(ph['iterations'], ph['complexity'], c='g',
             label='Training Complexity')
    plt.plot(th['iterations'], th['complexity'], c='c', linestyle='--',
             label='Batch Complexity')
    plt.xlim((0, th['iterations'][-1]))
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.subplot(6, 1, 2)
    plt.plot(ph['iterations'], 100 * ph['train_accuracy'], c='g',
             label='Training Accuracy (%)')
    plt.plot(ph['iterations'], 100 * ph['test_accuracy'], c='r',
             label='Test Accuracy (%)')
    plt.plot(th['iterations'], 100 * th['accuracy'], c='c', linestyle='--',
             label='Batch Accuracy (%)')
    plt.xlim((0, th['iterations'][-1]))
    plt.ylim((0, 100))
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.subplot(6, 1, 3)
    plt.plot(ph['iterations'], ph['train_cross_entropy_loss'], c='g',
             label='Training Cross Entropy Loss')
    plt.plot(ph['iterations'], ph['test_cross_entropy_loss'], c='r',
             label='Test Cross Entropy Loss')
    plt.plot(th['iterations'], th['cross_entropy_loss'], c='b', linestyle='--',
             label='Batch Raw-Predicted Cross Entropy Loss')
    plt.plot(th['iterations'], th['valid_cross_entropy_loss'],
             c='c', linestyle='--',
             label='Batch Clip-Normalized Cross Entropy Loss')
    plt.xlim((0, th['iterations'][-1]))
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.subplot(6, 1, 4)
    plt.plot(th['iterations'], th['gradient_norm'], c='c', linestyle='--',
             label='Gradient Norm')
    plt.xlim((0, th['iterations'][-1]))
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.subplot(6, 1, 5)
    plt.plot(th['iterations'], th['kernel_hypers'][:, 0],
             label='Kernel Sensitivity')
    plt.plot(th['iterations'], th['kernel_hypers'][:, 1:],
             label=['Kernel Length Scales %d' % (i + 1)
                    for i in range(th['kernel_hypers'].shape[1])])
    plt.xlim((0, th['iterations'][-1]))
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    plt.subplot(6, 1, 6)
    plt.plot(th['iterations'], np.log(th['regularisation']),
             label='Log Regularisation')
    plt.xlim((0, th['iterations'][-1]))
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    fig.set_size_inches(18, 20, forward=True)

    f = open('%s%s_results.txt' % (directory, name), 'w')
    f.write('There are %d classes for digits: %s\n' % (n_class, str(classes)))
    f.write('Training on %d images\n' % n_train)
    f.write('Testing on %d images\n' % n_test)
    f.write('-----\n')
    f.write('Kernel Embedding Classifier Training Setup:\n')
    f.write('Initial Hyperparameters: {} {}\n'.format(theta_init, zeta_init))
    f.write('Learning Rate: %f\n' % learning_rate)
    f.write('Gradient Error Tolerance: %f\n' % grad_tol)
    if n_sgd_batch:
        f.write('Batch Size for Stochastic Gradient Descent: %d\n'
                % n_sgd_batch)
    else:
        f.write('Batch Size for Stochastic Gradient Descent: Full Dataset')
    f.write('-----\n')
    f.write('Kernel Embedding Classifier Final Training Configuration:\n')
    f.write('Training Iterations: %d\n' % kec_steps_train)
    f.write('Hyperparameters: %s\n' % str(kec_h))
    f.write('Model Complexity: %f\n' % kec_complexity)
    f.write('Training Accuracy: %f\n' % kec_train_accuracy)
    f.write('Mean Sum of Probabilities: %f\n' % kec_mean_sum_probability)
    f.write('-----\n')
    f.write('SVC Hyperparameters: %s\n' % str(svc_h))
    f.write('GPC Hyperparameters: %s\n' % str(gpc_h))
    f.write('-----\n')
    f.write('kec training accuracy: %.9f\n' % kec_train_accuracy)
    f.write('svc training accuracy: %.9f\n' % svc_train_accuracy)
    f.write('gpc training accuracy: %.9f\n' % gpc_train_accuracy)
    f.write('kec test accuracy: %.9f\n' % kec_test_accuracy)
    f.write('svc test accuracy: %.9f\n' % svc_test_accuracy)
    f.write('gpc test accuracy: %.9f\n' % gpc_test_accuracy)
    f.write('kec training log loss: %.9f\n' % kec_train_log_loss)
    f.write('svc training log loss: %.9f\n' % svc_train_los_loss)
    f.write('gpc training log loss: %.9f\n' % gpc_train_los_loss)
    f.write('kec test log loss: %.9f\n' % kec_test_log_loss)
    f.write('svc test log loss: %.9f\n' % svc_test_los_loss)
    f.write('gpc test log loss: %.9f\n' % gpc_test_los_loss)
    f.write('-----\n')
    f.close()


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
        plt.scatter(x[:, 0], x[:, 1], c=y,  s=40, cmap=cm.jet,
                    label='Training Data')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='D', s=40,
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
        plt.scatter(x[:, 0], x[:, 1], c=y,  s=40, cmap=cm.jet,
                    label='Training Data')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='D', s=40,
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
        plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=cm.jet,
                    label='Training Data')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='D', s=40,
                    cmap=cm.jet, label='Test Data')
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
            plt.imshow(Z, extent=(x_1_lim[0], x_1_lim[1],
                                  x_2_lim[0], x_2_lim[1]), origin="lower")
            plt.contour(x_1_mesh, x_2_mesh, y_mesh, colors='k',
                        label='Decision Boundaries')
            plt.scatter(x[:, 0], x[:, 1],
                        c=np.array(["b", "g", "r"])[y.ravel()],
                        s=40, label='Training Data')
            plt.scatter(x_test[:, 0], x_test[:, 1],
                        c=np.array(["b", "g", "r"])[y_test.ravel()],
                        marker='D', s=40, cmap=cm.jet, label='Test Data')
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
                plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=40,
                            cmap=cm.jet, marker='D', label='Test Data')
                plt.xlim(x_1_lim)
                plt.ylim(x_2_lim)
                plt.xlabel('$x_{1}$')
                plt.ylabel('$x_{2}$')
                plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=8,
                           fancybox=True).get_frame().set_alpha(0.5)
                plt.title('%s: Probability for Class %d' % (name, c))


def cross_entropy(y, p):
    p_want = p[np.arange(y.shape[0]), y]
    return -np.mean(np.log(p_want))


def performance_history(hyper_history, kernel, classes,
                        x_train, y_train, x_test, y_test):

    # Make sure to translate the regularisation parameter through a sqrt
    # before this
    n_steps = hyper_history.shape[0]
    complexity_history = np.zeros(n_steps)
    train_accuracy_history = np.zeros(n_steps)
    train_cel_history = np.zeros(n_steps)
    test_accuracy_history = np.zeros(n_steps)
    test_cel_history = np.zeros(n_steps)
    for i, h in enumerate(hyper_history):
        kec = bake.Classifier(kernel=kernel).fit(x_train, y_train, h=h)
        complexity = kec.compute_complexity()
        kec_p_train = kec.predict_proba(x_train)
        kec_y_train = bake.infer.classify(kec_p_train, classes=classes)
        kec_train_accuracy = np.mean(kec_y_train == y_train.ravel())
        kec_train_log_loss = log_loss(y_train, kec_p_train)

        kec_p_test = kec.predict_proba(x_test)
        kec_y_test = bake.infer.classify(kec_p_test, classes=classes)
        kec_test_accuracy = np.mean(kec_y_test == y_test.ravel())
        kec_test_log_loss = log_loss(y_test, kec_p_test)

        complexity_history[i] = complexity
        train_accuracy_history[i] = kec_train_accuracy
        train_cel_history[i] = kec_train_log_loss
        test_accuracy_history[i] = kec_test_accuracy
        test_cel_history[i] = kec_test_log_loss

    return {'iterations': np.arange(n_steps) + 1,
            'complexity': complexity_history,
            'train_accuracy': train_accuracy_history,
            'train_cross_entropy_loss': train_cel_history,
            'test_accuracy': test_accuracy_history,
            'test_cross_entropy_loss': test_cel_history}




if __name__ == "__main__":

    n_attr = 4
    test_size = 0.1

    for a in np.arange(n_attr):
        for b in np.arange(a + 1, n_attr):

            full_directory = './iris_%s_attributes_%d_%d/' % (now_string, a, b)
            os.mkdir(full_directory)
            print('Results will be saved in "%s"' % full_directory)

            x, y = load_iris_data(a, b)
            np.random.seed(0)
            x_train, x_test, y_train, y_test = \
                train_test_split(x, y, test_size=test_size, random_state=0)
            utils.misc.time_module(iris_classification,
                                   x_train, y_train, x_test, y_test,
                                   directory=full_directory)

            # Save all figures and show all figures
            utils.misc.save_all_figures(full_directory,
                                        axis_equal=False, tight=True,
                                        extension='eps', rcparams=None)
            plt.close("all")
