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
import sys, pickle


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    else:
        raise Exception('No Python %d Available' % sys.version_info.major)
    fp.close()
    return data


def load_cifar_10_data(s):

    cifar_10 = unpickle("CIFAR_10_data/%s" % s)
    x = cifar_10['data']
    y = cifar_10['labels']
    n, d = x.shape
    print('%s has %d data points' % (s, n))
    images = np.reshape(x, (n, 3, 32, 32))[:, :, ::-1]
    images = np.swapaxes(np.swapaxes(images, 1, 3), 1, 2)

    x = x / np.max(x)
    return np.array(x), np.array(y)[:, np.newaxis], images


def load_cifar_10_names():

    cifar_10_names = unpickle("CIFAR_10_data/batches.meta")
    label_names = cifar_10_names['label_names']
    return label_names


def image_classification():

    x_train, y_train, images_train = load_cifar_10_data('data_batch_1')

    x_test, y_test, images_test = load_cifar_10_data('test_batch')

    classes = np.unique(y_train)

    label_names = load_cifar_10_names()

    n_sample_train = 2000
    x_train = x_train[:n_sample_train]
    y_train = y_train[:n_sample_train]
    images_train = images_train[:n_sample_train]
    #
    # n_sample_test = 20000
    # x_test = x_test[:n_sample_test]
    # y_test = y_test[:n_sample_test]
    # images_test = images_test[:n_sample_test]

    kernel = bake.kernels.s_gaussian

    # h_impose = np.array([0.5, 0.5, 0.01])
    # kec = bake.Classifier(kernel=kernel).fit(x_train, y_train, h=h_impose)

    h_min = np.array([0.2, 0.1, 0.001])
    h_max = np.array([2.0, 5.0, 10.0])
    h_init = np.array([1.0, 2.0, 0.01])

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
    for index, (image, label,
                kec_pred, kec_p,
                svc_pred, svc_p,
                gpc_pred, gpc_p) in enumerate(images_and_labels[:n_pic]):
        plt.subplot(n_row, n_col, index + 1)
        plt.axis('off')
        plt.imshow(image, origin="lower")
        plt.title('Truth: %s'
                  '\nKEC: %s (%.1f%%)'
                  '\nSVC: %s (%.1f%%)'
                  '\nGPC: %s (%.1f%%)' % (label_names[label],
                                          label_names[kec_pred], 100*kec_p,
                                          label_names[svc_pred], 100*svc_p,
                                          label_names[gpc_pred], 100*gpc_p))

    fig.set_size_inches(18, 14, forward=True)
    full_directory = './'
    utils.misc.save_all_figures(full_directory,
                                axis_equal=True, tight=True,
                                extension='eps', rcparams=None)


if __name__ == "__main__":

    utils.misc.time_module(image_classification)
    plt.show()
