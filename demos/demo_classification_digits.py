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
from sklearn.model_selection import train_test_split
from sklearn import datasets


def create_digits_data():
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    x = x / np.max(x)
    return x, y[:, np.newaxis], digits.images


def log(x):
    answer = np.log(x)
    answer[x <= 0] = 0
    return answer


def digit_classification():

    x_all, y_all, images = create_digits_data()
    n_samples = x_all.shape[0]

    i_break = int(0.2 * n_samples)
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

    h_impose = np.array([0.92, 2.25, 0.08])
    kec = bake.Classifier(kernel=kernel).fit(x, y, hyperparam=h_impose)

    # h_min = np.array([0.5, 0.25, 0.001])
    # h_max = np.array([2.0, 5.0, 1.0])
    # h_init = np.array([1.0, 2.0, 0.01])
    # kec = bake.Classifier(kernel=kernel).fit(x, y, h_min=h_min, h_max=h_max, h_init=h_init)

    kec_p_test= kec.probability(x_test)
    kec_p_test = bake.infer.clip_normalize(kec_p_test).T
    kec_h_test = -np.sum(kec_p_test * log(kec_p_test), axis=1)

    svc = SVC(probability=True).fit(x, y)
    svc_p_test= svc.predict_proba(x_test)
    svc_h_test = -np.sum(svc_p_test * log(svc_p_test), axis=1)

    gpc = GaussianProcessClassifier().fit(x, y)
    gpc_p_test= gpc.predict_proba(x_test)
    gpc_h_test = -np.sum(gpc_p_test * log(gpc_p_test), axis=1)

    kec_y_pred = kec.predict(x)
    svc_y_pred = svc.predict(x)
    gpc_y_pred = gpc.predict(x)
    print('kec training accuracy: %.9f' % (np.mean(kec_y_pred == y.ravel())))
    print('svc training accuracy: %.9f' % (np.mean(svc_y_pred == y.ravel())))
    print('gpc training accuracy: %.9f' % (np.mean(gpc_y_pred == y.ravel())))

    kec_y_pred = kec.predict(x_test)
    svc_y_pred = svc.predict(x_test)
    gpc_y_pred = gpc.predict(x_test)
    print('kec test accuracy: %.9f' % (np.mean(kec_y_pred == y_test.ravel())))
    print('svc test accuracy: %.9f' % (np.mean(svc_y_pred == y_test.ravel())))
    print('gpc test accuracy: %.9f' % (np.mean(gpc_y_pred == y_test.ravel())))

    kec_p_pred = kec_p_test[np.arange(x_test.shape[0]), kec_y_pred]
    svc_p_pred = svc_p_test[np.arange(x_test.shape[0]), svc_y_pred]
    gpc_p_pred = gpc_p_test[np.arange(x_test.shape[0]), gpc_y_pred]

    print('Percentage of data withheld for testing: %f%%' % (100 * test_size))

    fig = plt.figure(0)
    n_row = 3
    n_col = 6
    n_pic = n_row * n_col
    images_and_labels = list(zip(images[i_test], y_test, kec_y_pred, kec_p_pred, svc_y_pred, svc_p_pred, gpc_y_pred, gpc_p_pred))
    for index, (image, label, kec_pred, kec_p, svc_pred, svc_p, gpc_pred, gpc_p) in enumerate(images_and_labels[:n_pic]):
        plt.subplot(n_row, n_col, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Truth: %i\nKEC: %i (%.1f%%)\nSVC: %i (%.1f%%)\nGPC: %i (%.1f%%)' % (label, kec_pred, 100*kec_p, svc_pred, 100*svc_p, gpc_pred, 100*gpc_p))

    fig.set_size_inches(18, 14, forward=True)
    full_directory = './'
    utils.misc.save_all_figures(full_directory,
                                axis_equal=True, tight=True,
                                extension='eps', rcparams=None)


if __name__ == "__main__":

    utils.misc.time_module(digit_classification)
    plt.show()
