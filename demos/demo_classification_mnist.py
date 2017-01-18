"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import log_loss
from tensorflow.examples.tutorials.mnist import input_data


def create_mnist_data():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x = mnist.train.images
    y = mnist.train.labels
    n = x.shape[0]
    images = np.reshape(x, (n, 28, 28))

    n_sample = 20000
    x = x[:n_sample]
    y = y[:n_sample]
    images = images[:n_sample]
    x = x / np.max(x)
    return x, y[:, np.newaxis], images


def digit_classification():

    x_all, y_all, images = create_mnist_data()
    n_samples = x_all.shape[0]
    classes = np.unique(y_all)

    i_break = int(0.05 * n_samples)
    np.random.seed(0)
    indices = np.random.permutation(n_samples)
    i_train = indices[:i_break]
    i_test = indices[i_break:]

    test_size = i_test.shape[0] / n_samples

    x = x_all[i_train]
    y = y_all[i_train]
    x_test = x_all[i_test]
    y_test = y_all[i_test]

    kernel = bake.kernels.s_gaussian

    # h_impose = np.array([0.92, 2.25, 0.08])
    # kec = bake.Classifier(kernel=kernel).fit(x, y, hyperparam=h_impose)

    h_min = np.array([0.2, 0.25, 0.001])
    h_max = np.array([2.0, 5.0, 0.1])
    h_init = np.array([1.0, 2.0, 0.01])

    kec = bake.Classifier(kernel=kernel).fit(x, y,
                                             h_min=h_min,
                                             h_max=h_max,
                                             h_init=h_init)
    print('KEC Training Finished')
    svc = SVC(probability=True).fit(x, y)
    print('SVC Training Finished')
    gpc = GaussianProcessClassifier().fit(x, y)
    print('GPC Training Finished')

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

    kec_p_pred = kec_p_test[np.arange(x_test.shape[0]), kec_y_test]
    svc_p_pred = svc_p_test[np.arange(x_test.shape[0]), svc_y_test]
    gpc_p_pred = gpc_p_test[np.arange(x_test.shape[0]), gpc_y_test]

    print('Percentage of data withheld for testing: %f%%' % (100 * test_size))

    fig = plt.figure(0)
    n_row = 3
    n_col = 5
    n_pic = n_row * n_col
    images_and_labels = list(zip(images[i_test], y_test,
                                 kec_y_test, kec_p_pred,
                                 svc_y_test, svc_p_pred,
                                 gpc_y_test, gpc_p_pred))
    for index, (image, label,
                kec_pred, kec_p,
                svc_pred, svc_p,
                gpc_pred, gpc_p) in enumerate(images_and_labels[:n_pic]):
        plt.subplot(n_row, n_col, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Truth: %i'
                  '\nKEC: %i (%.1f%%)'
                  '\nSVC: %i (%.1f%%)'
                  '\nGPC: %i (%.1f%%)' % (label,
                                          kec_pred, 100*kec_p,
                                          svc_pred, 100*svc_p,
                                          gpc_pred, 100*gpc_p))

    fig.set_size_inches(18, 14, forward=True)
    full_directory = './'
    utils.misc.save_all_figures(full_directory,
                                axis_equal=True, tight=True,
                                extension='eps', rcparams=None)


if __name__ == "__main__":

    utils.misc.time_module(digit_classification)
    plt.show()
