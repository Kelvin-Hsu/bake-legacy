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

    x_train = mnist.train.images
    y_train = mnist.train.labels[:, np.newaxis]
    n_train, d = x_train.shape
    images_train = np.reshape(x_train, (n_train, 28, 28))

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels[:, np.newaxis]
    n_valid, d = x_valid.shape
    images_valid = np.reshape(x_valid, (n_valid, 28, 28))

    x = np.concatenate((x_train, x_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)
    images = np.concatenate((images_train, images_valid), axis=0)

    x_test = mnist.test.images
    y_test = mnist.test.labels[:, np.newaxis]
    n_test, d = x_test.shape
    images_test = np.reshape(x_test, (n_test, 28, 28))

    n_sample = 1000
    x = x[:n_sample]
    y = y[:n_sample]
    images = images[:n_sample]

    x /= np.max(x)
    print('Training Size: %d, Validation Size: %d, Testing Size: %d'
          % (n_train, n_valid, n_test))
    return x, y, images, x_test, y_test, images_test


def digit_classification():

    x_train, y_train, images, x_test, y_test, images_test = create_mnist_data()
    classes = np.unique(y_train)

    # kernel = bake.kernels.s_gaussian
    # h_min = np.array([0.2, 0.25, 0.001])
    # h_max = np.array([2.0, 5.0, 0.1])
    # h_init = np.array([1.0, 2.0, 0.01])

    kernel = bake.kernels.gaussian
    h_min = np.array([0.25, 0.001])
    h_max = np.array([5.0, 0.1])
    h_init = np.array([2.0, 0.01])

    # h_impose = np.array([1, 1.0, 0.01])
    # kec = bake.Classifier(kernel=kernel).fit(x_train, y_train, h=h_impose)

    kec = bake.Classifier(kernel=kernel).fit(x_train, y_train,
                                             h_min=h_min,
                                             h_max=h_max,
                                             h_init=h_init)
    print('KEC Training Finished')
    svc = SVC(probability=True).fit(x_train, y_train)
    print('SVC Training Finished')
    gpc = GaussianProcessClassifier().fit(x_train, y_train)
    print('GPC Training Finished')

    kec_p = kec.predict_proba(x_train)
    svc_p = svc.predict_proba(x_train)
    gpc_p = gpc.predict_proba(x_train)

    kec_y = bake.infer.classify(kec_p, classes=classes)
    svc_y = bake.infer.classify(svc_p, classes=classes)
    gpc_y = bake.infer.classify(svc_p, classes=classes)

    kec_p_test = kec.predict_proba(x_test)
    svc_p_test = svc.predict_proba(x_test)
    gpc_p_test = gpc.predict_proba(x_test)

    kec_y_test = bake.infer.classify(kec_p_test, classes=classes)
    svc_y_test = bake.infer.classify(svc_p_test, classes=classes)
    gpc_y_test = bake.infer.classify(gpc_p_test, classes=classes)

    print('Training on %d images' % y_train.shape[0])
    print('Testing on %d images' % y_test.shape[0])

    print('kec training accuracy: %.9f' % (np.mean(kec_y == y_train.ravel())))
    print('svc training accuracy: %.9f' % (np.mean(svc_y == y_train.ravel())))
    print('gpc training accuracy: %.9f' % (np.mean(gpc_y == y_train.ravel())))

    print('kec test accuracy: %.9f' % (np.mean(kec_y_test == y_test.ravel())))
    print('svc test accuracy: %.9f' % (np.mean(svc_y_test == y_test.ravel())))
    print('gpc test accuracy: %.9f' % (np.mean(gpc_y_test == y_test.ravel())))

    print('kec training log loss: %.9f' % log_loss(y_train.ravel(), kec_p))
    print('svc training log loss: %.9f' % log_loss(y_train.ravel(), svc_p))
    print('gpc training log loss: %.9f' % log_loss(y_train.ravel(), gpc_p))

    print('kec test log loss: %.9f' % log_loss(y_test.ravel(), kec_p_test))
    print('svc test log loss: %.9f' % log_loss(y_test.ravel(), svc_p_test))
    print('gpc test log loss: %.9f' % log_loss(y_test.ravel(), gpc_p_test))

    kec_p_pred = kec_p_test[np.arange(x_test.shape[0]), kec_y_test]
    svc_p_pred = svc_p_test[np.arange(x_test.shape[0]), svc_y_test]
    gpc_p_pred = gpc_p_test[np.arange(x_test.shape[0]), gpc_y_test]

    fig = plt.figure(0)
    n_row = 3
    n_col = 5
    n_pic = n_row * n_col
    images_and_labels = list(zip(images_test, y_test,
                                 kec_y_test, kec_p_pred,
                                 svc_y_test, svc_p_pred,
                                 gpc_y_test, gpc_p_pred))
    plt.figure()
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

    input_variance = kec.input_variance()
    input_variance_images = np.reshape(input_variance, (10, 28, 28))

    plt.figure()
    for index, image in enumerate(input_variance_images):
        plt.subplot(3, 4, index)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Variations for %d' % index)

    fig.set_size_inches(18, 14, forward=True)
    full_directory = './'
    utils.misc.save_all_figures(full_directory,
                                axis_equal=True, tight=True,
                                extension='eps', rcparams=None)


if __name__ == "__main__":

    utils.misc.time_module(digit_classification)
    plt.show()
