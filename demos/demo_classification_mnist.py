"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
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

    # digits = np.array([4, 7, 9])
    #
    # indices = np.any(y == digits, axis=1)
    # x = x[indices]
    # y = y[indices]
    # images = images[indices]
    #
    # indices = np.any(y_test == digits, axis=1)
    # x_test = x_test[indices]
    # y_test = y_test[indices]
    # images_test = images_test[indices]

    n_sample = 500
    x = x[:n_sample]
    y = y[:n_sample]
    images = images[:n_sample]

    print('Training Size: %d, Testing Size: %d' % (x.shape[0], x_test.shape[0]))
    return x, y, images, x_test, y_test, images_test


def digit_classification():

    x_train, y_train, images, x_test, y_test, images_test = create_mnist_data()
    classes = np.unique(y_train)

    # kernel = bake.kernels.s_gaussian
    # h_min = np.array([0.2, 0.25, 0.001])
    # h_max = np.array([2.0, 5.0, 0.1])
    # h_init = np.array([1.0, 2.0, 0.01])

    # kernel = bake.kernels.gaussian
    # h_min = np.array([0.25, 0.001])
    # h_max = np.array([5.0, 0.1])
    # h_init = np.array([2.0, 0.01])

    kernel = bake.kernels.s_gaussian

    # FOR ISOTROPIC TEST
    # h_min = np.array([0.5, 0.75, 0.001])
    # h_max = np.array([2.0, 4.0, 0.1])
    # h_init = np.array([1.0, 1.0, 0.01])

    # FOR ANISOTROPIC TEST
    t_min = 0.2 * np.ones(x_train.shape[1])
    t_max = 20.0 * np.ones(x_train.shape[1])
    t_init = 2.0 * np.ones(x_train.shape[1])

    h_min = np.concatenate(([0.5], t_min, [1e-10]))
    h_max = np.concatenate(([5.0], t_max, [1]))
    h_init = np.concatenate(([1.0], t_init, [1e-2]))

    # FOR ANISOTROPIC TEST
    # file = np.load('./anisotropic_500_digits.npz')
    # h_impose = file['h']
    # kec = bake.Classifier(kernel=kernel).fit(x_train, y_train, h=h_impose)

    # FOR ISOTROPIC TEST
    # h_impose = np.array([0.88, 1.73, 0.078])  # Matern
    # h_impose = np.array([0.95,  3.2,  0.06])  # Gaussian
    # kec = bake.Classifier(kernel=kernel).fit(x_train, y_train, h=h_impose)

    kec = bake.Classifier(kernel=kernel).fit(x_train, y_train,
                                             h_min=h_min,
                                             h_max=h_max,
                                             h_init=h_init)
    np.savez('anisotropic_500_digits.npz', h=np.append(kec.theta, kec.zeta))
    print('KEC Training Finished')
    svc = SVC(probability=True).fit(x_train, y_train)
    print('SVC Training Finished')
    gp_kernel = RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=gp_kernel).fit(x_train, y_train)
    print('Gaussian Process Hyperparameters: ', gpc.kernel_.theta)
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

    kec_p_pred = kec_p_test[np.arange(x_test.shape[0]), np.argmax(kec_p_test, axis=1)]
    svc_p_pred = svc_p_test[np.arange(x_test.shape[0]), np.argmax(svc_p_test, axis=1)]
    gpc_p_pred = gpc_p_test[np.arange(x_test.shape[0]), np.argmax(gpc_p_test, axis=1)]

    n_row = 3
    n_col = 5
    n_pic = n_row * n_col
    images_and_labels = list(zip(images_test, y_test,
                                 kec_y_test, kec_p_pred,
                                 svc_y_test, svc_p_pred,
                                 gpc_y_test, gpc_p_pred))

    for j in range(12):
        fig = plt.figure()
        for index, (image, label,
                    kec_pred, kec_p,
                    svc_pred, svc_p,
                    gpc_pred, gpc_p) in enumerate(
            images_and_labels[j*n_pic:(j + 1)*n_pic]):
            plt.subplot(n_row, n_col, index + 1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Truth: %d'
                      '\nKEC: %d (%.1f%%)'
                      '\nSVC: %d (%.1f%%)'
                      '\nGPC: %d (%.1f%%)' % (label,
                                              kec_pred, 100*kec_p,
                                              svc_pred, 100*svc_p,
                                              gpc_pred, 100*gpc_p))
        fig.set_size_inches(18, 14, forward=True)

    for j in range(10):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,
                               width_ratios=[1, 2.5],
                               height_ratios=[1, 1])
        plt.subplot(gs[0])
        plt.axis('off')
        plt.imshow(images_test[j], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Truth: %d' % y_test[j])
        plt.subplot(gs[1])
        bar_width = 0.2
        opacity = 0.4
        plt.bar(classes, tuple(svc_p_test[j]), bar_width,
                alpha=opacity,
                color='r',
                label='SVC')
        plt.bar(classes + bar_width, tuple(kec_p_test[j]), bar_width,
                alpha=opacity,
                color='g',
                label='KEC')
        plt.bar(classes + 2 * bar_width, tuple(gpc_p_test[j]), bar_width,
                alpha=opacity,
                color='b',
                label='GPC')
        plt.xlabel('Classes')
        plt.ylabel('Probabilities')
        plt.ylim((0, 1))
        plt.title('Prediction Probabilities')
        plt.xticks(classes + 1.5 * bar_width, tuple([str(c) for c in classes]))
        # Shrink current axis's height by 10% on the bottom
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0 + box.height * 0.1,
                               box.width, box.height * 0.9])
        # Put a legend below current axis
        plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                         fancybox=True, shadow=True, ncol=3)
        plt.tight_layout()
        fig.set_size_inches(18, 4, forward=True)

    x_train_mean = np.array([np.mean(x_train[y_train.ravel() == c], axis=0)
                             for c in classes])
    x_train_var = np.array([np.var(x_train[y_train.ravel() == c], axis=0)
                            for c in classes])

    x_train_mean_images = np.reshape(x_train_mean, (classes.shape[0], 28, 28))
    x_train_var_images = np.reshape(x_train_var, (classes.shape[0], 28, 28))

    fig = plt.figure()
    for index, image in enumerate(x_train_mean_images):
        plt.subplot(1, classes.shape[0], index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Empirical Expectance')
    fig.set_size_inches(18, 4, forward=True)

    fig = plt.figure()
    for index, image in enumerate(x_train_var_images):
        plt.subplot(1, classes.shape[0], index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Empirical Variance')
    fig.set_size_inches(18, 4, forward=True)

    input_expectance = kec.input_expectance()
    input_expectance_images = np.reshape(input_expectance, (classes.shape[0], 28, 28))

    fig = plt.figure()
    for index, image in enumerate(input_expectance_images):
        plt.subplot(1, classes.shape[0], index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Input Expectance')
    fig.set_size_inches(18, 4, forward=True)

    input_variance = kec.input_variance()
    input_variance_images = np.reshape(input_variance, (classes.shape[0], 28, 28))

    fig = plt.figure()
    for index, image in enumerate(input_variance_images):
        plt.subplot(1, classes.shape[0], index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Input Variance')
    fig.set_size_inches(18, 4, forward=True)

    input_mode = kec.input_mode()
    input_mode_images = np.reshape(input_mode, (classes.shape[0], 28, 28))

    fig = plt.figure()
    for index, image in enumerate(input_mode_images):
        plt.subplot(1, classes.shape[0], index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Input Modes')
    fig.set_size_inches(18, 4, forward=True)

    if kec.theta.shape[0] == 28*28 + 1:
        fig = plt.figure()
        theta_image = np.reshape(kec.theta[1:], (28, 28))
        plt.axis('off')
        plt.imshow(theta_image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Pixel Relevance for Classifying Digits ' + str(classes))
        fig.set_size_inches(18, 9, forward=True)

    full_directory = './'
    utils.misc.save_all_figures(full_directory,
                                axis_equal=True, tight=True,
                                extension='eps', rcparams=None)


if __name__ == "__main__":

    utils.misc.time_module(digit_classification)
    plt.show()
