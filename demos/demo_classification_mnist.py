"""
Application of the kernel embedding classifier on the MNIST dataset.
"""
import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import log_loss
import os
from scipy.spatial.distance import cdist
import datetime

now = datetime.datetime.now()
now_string = '%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                    now.hour, now.minute, now.second)


def create_mnist_data(load_from_tf=False):
    """
    Load the mnist dataset.

    Returns
    -------
    numpy.ndarray
        The training inputs (n_train, 28 * 28)
    numpy.ndarray
        The training outputs (n_train, 1)
    numpy.ndarray
        The training images (n_train, 28, 28)
    numpy.ndarray
        The test inputs (n_test, 28 * 28)
    numpy.ndarray
        The test outputs (n_test, 1)
    numpy.ndarray
        The test images (n_test, 28, 28)
    """
    if load_from_tf:
        from tensorflow.examples.tutorials.mnist import input_data
        # Load the MNIST dataset
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

        # Load the training data
        x_train = mnist.train.images
        y_train = mnist.train.labels[:, np.newaxis]
        n_train, d = x_train.shape
        images_train = np.reshape(x_train, (n_train, 28, 28))

        # Load the validation data
        x_valid = mnist.validation.images
        y_valid = mnist.validation.labels[:, np.newaxis]
        n_valid, d = x_valid.shape
        images_valid = np.reshape(x_valid, (n_valid, 28, 28))

        # Add the validation data to the training data
        x = np.concatenate((x_train, x_valid), axis=0)
        y = np.concatenate((y_train, y_valid), axis=0)
        images = np.concatenate((images_train, images_valid), axis=0)

        # Load the testing data
        x_test = mnist.test.images
        y_test = mnist.test.labels[:, np.newaxis]
        n_test, d = x_test.shape
        images_test = np.reshape(x_test, (n_test, 28, 28))

        np.savez('./MNIST_data/mnist_data.npz',
                 x=x,
                 y=y,
                 images=images,
                 x_test=x_test,
                 y_test=y_test,
                 images_test=images_test)
    else:
        file = np.load('./MNIST_data/mnist_data.npz')
        x = file['x']
        y = file['y']
        images = file['images']
        x_test = file['x_test']
        y_test = file['y_test']
        images_test = file['images_test']

    return x, y, images, x_test, y_test, images_test


def process_mnist_data(x, y, images, x_test, y_test, images_test,
                       digits=np.arange(10), n_sample=500, sample_before=True):
    """
    Process and load a subset of the mnist dataset.

    Parameters
    ----------
    digits : numpy.ndarray
        An array of the digit classes to restrict the dataset to
    n_sample : int
        The number of training samples to sample
    sample_before : bool
        Whether to reduce the training set first before restricting digits

    Returns
    -------
    numpy.ndarray
        The training inputs (n_train, 28 * 28)
    numpy.ndarray
        The training outputs (n_train, 1)
    numpy.ndarray
        The training images (n_train, 28, 28)
    numpy.ndarray
        The test inputs (n_test, 28 * 28)
    numpy.ndarray
        The test outputs (n_test, 1)
    numpy.ndarray
        The test images (n_test, 28, 28)
    """
    # Limit the training set to only the specified number of data points
    if sample_before:
        x = x[:n_sample]
        y = y[:n_sample]
        images = images[:n_sample]

    # Limit the training set to only the specified digits
    indices = np.any(y == digits, axis=1)
    x = x[indices]
    y = y[indices]
    images = images[indices]

    # Limit the testing set to only the specified digits
    indices = np.any(y_test == digits, axis=1)
    x_test = x_test[indices]
    y_test = y_test[indices]
    images_test = images_test[indices]

    # Limit the training set to only the specified number of data points
    if not sample_before:
        x = x[:n_sample]
        y = y[:n_sample]
        images = images[:n_sample]

    print('Digits extracted: ', digits)
    print('Training Size: %d, Testing Size: %d'
          % (x.shape[0], x_test.shape[0]))
    return x, y, images, x_test, y_test, images_test


def digit_classification(x_train, y_train, images_train,
                         x_test, y_test, images_test):
    """
    Performs the digit classification task and saves results.

    Parameters
    ----------
    x_train : numpy.ndarray
        The training inputs (n_train, 28 * 28)
    y_train : numpy.ndarray
        The training outputs (n_train, 1)
    images_train : numpy.ndarray
        The training images (n_train, 28, 28)
    x_test : numpy.ndarray
        The test inputs (n_test, 28 * 28)
    y_test : numpy.ndarray
        The test outputs (n_test, 1)
    images_test : numpy.ndarray
        The test images (n_test, 28, 28)

    Returns
    -------
    None
    """
    # Determine the classes, number of classes, and input dimensions for
    # this digit classification test
    classes = np.unique(y_train)
    n_class = classes.shape[0]
    d = x_train.shape[1]
    print('\n--------------------------------------------------------------\n')
    print('There are %d classes for digits: ' % n_class, classes)

    # Report the number of training and testing points used in this test
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    print('Training on %d images' % n_train)
    print('Testing on %d images' % n_test)

    # Create full directory
    digits_str = ''.join([str(i) for i in classes])
    full_directory = './%s_mnist_digits_%s_with_%d_training_images/' \
                     % (now_string, digits_str, n_train)
    os.mkdir(full_directory)
    print('Results will be saved in "%s"' % full_directory)

    # Save the training and test data
    np.savez('%strain_test_data.npz' % full_directory,
             x_train=x_train,
             y_train=y_train,
             images_train=images_train,
             x_test=x_test,
             y_test=y_test,
             images_test=images_test)

    # Specify the kernel used for the classifier
    kec_kernel = bake.kernels.s_gaussian

    # Specify settings for hyperparameter learning
    t_min = 0.2 * np.ones(d)
    t_max = 20.0 * np.ones(d)
    t_init = 2.0 * np.ones(d)
    h_min = np.concatenate(([0.5], t_min, [1e-10]))
    h_max = np.concatenate(([5.0], t_max, [1]))
    h_init = np.concatenate(([1.0], t_init, [1e-2]))

    # Alternative: Load learned hyperparameter from a file
    # file = np.load('./mnist_training_results.npz')
    # h_fixed = file['h']
    # kec = bake.Classifier(kernel=kec_kernel).fit(x_train, y_train, h=h_fixed)

    # Train the kernel embedding classifier
    kec = bake.Classifier(kernel=kec_kernel).fit(x_train, y_train,
                                                 h_min=h_min,
                                                 h_max=h_max,
                                                 h_init=h_init)
    kec_h = np.append(kec.theta, kec.zeta)
    kec_f_train = kec._f_train
    kec_a_train = kec._a_train
    kec_p_train = kec._p_train
    kec_h_train = kec._h_train
    kec_complexity = kec.complexity
    kec_mean_sum_probability = kec.mean_sum_probability
    print('KEC Training Finished')

    # Train the support vector classifier
    svc = SVC(probability=True).fit(x_train, y_train)
    print('SVC Training Finished')

    # Train the gaussian process classifier
    gpc_kernel = C() * RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=gpc_kernel).fit(x_train, y_train)
    gpc_h = gpc.kernel_.theta
    print('Gaussian Process Hyperparameters: ', gpc_h)
    print('GPC Training Finished')

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

    # Obtain the probabilities of the predictions only
    kec_p_pred = kec_p_test[np.arange(n_test), np.argmax(kec_p_test, axis=1)]
    svc_p_pred = svc_p_test[np.arange(n_test), np.argmax(svc_p_test, axis=1)]
    gpc_p_pred = gpc_p_test[np.arange(n_test), np.argmax(gpc_p_test, axis=1)]

    # Compute the empirical expectance and variance for each class
    x_train_mean = np.array([np.mean(x_train[y_train.ravel() == c], axis=0)
                             for c in classes])
    x_train_var = np.array([np.var(x_train[y_train.ravel() == c], axis=0)
                            for c in classes])

    # Compute the input expectance, variance, and mode from KEC
    input_expectance = kec.input_expectance()
    input_variance = kec.input_variance()
    input_mode = kec.input_mode()
    kec_x_modes = kec.x_modes
    kec_mu_modes = kec.mu_modes

    # Determine the nearest training image
    x_train_mode = np.zeros(x_train_mean.shape)
    for i, c in enumerate(classes):
        ind = cdist(input_mode[[i]], x_train[y_train.ravel() == c],
                    'euclidean').ravel().argmin()
        x_train_mode[i, :] = x_train[y_train.ravel() == c][ind]

    # Determine the distance to the closest training image to check that
    # the mode is not simply a training input
    for i, c in enumerate(classes):
        dist = cdist(input_mode[[i]], x_train[y_train.ravel() == c],
                     'euclidean').min()
        print('Euclidean distance from mode to closest training image '
              'for class %d: %f || Embedding Value: %f'
              % (c, dist, kec_mu_modes[i]))

    # Save the training results for the kernel embedding classifier
    np.savez('%smnist_training_results.npz' % full_directory,
             classes=classes,
             n_class=n_class,
             h_min=h_min,
             h_max=h_max,
             h_init=h_init,
             kec_h=kec_h,
             gpc_h=gpc_h,
             kec_f_train=kec_f_train,
             kec_a_train=kec_a_train,
             kec_p_train=kec_p_train,
             kec_h_train=kec_h_train,
             kec_x_modes=kec_x_modes,
             kec_mu_modes=kec_mu_modes,
             kec_p=kec_p,
             svc_p=svc_p,
             gpc_p=gpc_p,
             kec_y=kec_y,
             svc_y=svc_y,
             gpc_y=gpc_y,
             kec_p_test=kec_p_test,
             svc_p_test=svc_p_test,
             gpc_p_test=gpc_p_test,
             kec_y_test=kec_y_test,
             svc_y_test=svc_y_test,
             gpc_y_test=gpc_y_test,
             kec_complexity=kec_complexity,
             kec_mean_sum_probability=kec_mean_sum_probability,
             kec_train_accuracy=kec_train_accuracy,
             svc_train_accuracy=svc_train_accuracy,
             gpc_train_accuracy=gpc_train_accuracy,
             kec_test_accuracy=kec_test_accuracy,
             svc_test_accuracy=svc_test_accuracy,
             gpc_test_accuracy=gpc_test_accuracy,
             kec_train_log_loss=kec_train_log_loss,
             svc_train_los_loss=svc_train_los_loss,
             gpc_train_los_loss=gpc_train_los_loss,
             kec_test_log_loss=kec_test_log_loss,
             svc_test_los_loss=svc_test_los_loss,
             gpc_test_los_loss=gpc_test_los_loss,
             kec_p_pred=kec_p_pred,
             svc_p_pred=svc_p_pred,
             gpc_p_pred=gpc_p_pred,
             x_train_mean=x_train_mean,
             x_train_var=x_train_var,
             input_expectance=input_expectance,
             input_variance=input_variance,
             input_mode=input_mode)

    # Convert the above into image form
    x_train_mean_images = np.reshape(x_train_mean, (n_class, 28, 28))
    x_train_var_images = np.reshape(x_train_var, (n_class, 28, 28))
    input_expectance_images = np.reshape(input_expectance, (n_class, 28, 28))
    input_variance_images = np.reshape(input_variance, (n_class, 28, 28))
    input_mode_images = np.reshape(input_mode, (n_class, 28, 28))
    x_train_mode_images = np.reshape(x_train_mode, (n_class, 28, 28))

    # Visualise the predictions on the testing set
    prediction_results = list(zip(images_test, y_test,
                                  kec_y_test, kec_p_pred,
                                  svc_y_test, svc_p_pred,
                                  gpc_y_test, gpc_p_pred))
    n_figures = 20
    n_row = 3
    n_col = 5
    n_pic = n_row * n_col
    for j in range(n_figures):
        fig = plt.figure()
        for index, (image, label,
                    kec_pred, kec_p,
                    svc_pred, svc_p,
                    gpc_pred, gpc_p) in enumerate(
                    prediction_results[j * n_pic:(j + 1) * n_pic]):
            plt.subplot(n_row, n_col, index + 1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Truth: %d'
                      '\nKEC: %d (%.1f%%)'
                      '\nSVC: %d (%.1f%%)'
                      '\nGPC: %d (%.1f%%)' % (label,
                                              kec_pred, 100 * kec_p,
                                              svc_pred, 100 * svc_p,
                                              gpc_pred, 100 * gpc_p))
        fig.set_size_inches(18, 14, forward=True)

    # Visualise the probability distribution of each prediction
    n_figures = 20
    for j in range(n_figures):
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
        plt.bar(np.arange(n_class), tuple(svc_p_test[j]),
                bar_width,
                alpha=opacity,
                color='r',
                label='SVC')
        plt.bar(np.arange(n_class) + bar_width, tuple(kec_p_test[j]),
                bar_width,
                alpha=opacity,
                color='g',
                label='KEC')
        plt.bar(np.arange(n_class) + 2 * bar_width, tuple(gpc_p_test[j]),
                bar_width,
                alpha=opacity,
                color='b',
                label='GPC')
        plt.xlabel('Classes')
        plt.ylabel('Probabilities')
        plt.ylim((0, 1))
        plt.title('Prediction Probabilities')
        plt.xticks(np.arange(n_class) + 1.5 * bar_width,
                   tuple([str(c) for c in classes]))
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0 + box.height * 0.1,
                               box.width, box.height * 0.9])
        plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                         fancybox=True, shadow=True, ncol=3)
        plt.tight_layout()
        fig.set_size_inches(18, 3, forward=True)

    # Show the empirical expectance
    fig = plt.figure()
    for index, image in enumerate(x_train_mean_images):
        plt.subplot(1, n_class, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Empirical Expectance')
    fig.set_size_inches(18, 4, forward=True)

    # Show the empirical variance
    fig = plt.figure()
    for index, image in enumerate(x_train_var_images):
        plt.subplot(1, n_class, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Empirical Variance')
    fig.set_size_inches(18, 4, forward=True)

    # Show the input expectance
    fig = plt.figure()
    for index, image in enumerate(input_expectance_images):
        plt.subplot(1, n_class, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Input Expectance')
    fig.set_size_inches(18, 4, forward=True)

    # Show the input variance
    fig = plt.figure()
    for index, image in enumerate(input_variance_images):
        plt.subplot(1, n_class, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Input Variance')
    fig.set_size_inches(18, 4, forward=True)

    # Show the input mode
    fig = plt.figure()
    for index, image in enumerate(input_mode_images):
        plt.subplot(1, n_class, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Input Modes')
    fig.set_size_inches(18, 4, forward=True)

    # Show the closest mode
    fig = plt.figure()
    for index, image in enumerate(x_train_mode_images):
        plt.subplot(1, n_class, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%d' % classes[index])
    plt.suptitle('Closest Training Image to Modes')
    fig.set_size_inches(18, 4, forward=True)

    # If the classifier was anisotropic, show the pixel relevance
    if kec_h.shape[0] == 28*28 + 2:
        fig = plt.figure()
        theta_image = np.reshape(kec_h[1:-1], (28, 28))
        plt.axis('off')
        plt.imshow(theta_image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Pixel Relevance for Classifying Digits ' + str(classes))
        fig.set_size_inches(8, 8, forward=True)

    for i in plt.get_fignums():
        fig = plt.figure(i)
        plt.gca().set_aspect('equal', adjustable='box')
        fig.tight_layout()

    # Plot the objective and constraints history
    fig = plt.figure()
    iters = np.arange(kec_f_train.shape[0])
    plt.plot(iters, kec_f_train, c='c',
             label='KEC Complexity (Scale: $\log_{e}$)')
    plt.plot(iters, kec_a_train, c='r', label='KEC Training Accuracy')
    plt.plot(iters, kec_p_train, c='g', label='KEC Mean Sum of Probabilities')
    plt.title('Training History: Objectives and Constraints')
    plt.xlabel('Iterations')
    leg = plt.legend(loc='center', bbox_to_anchor=(0.95, 0.9),
                     fancybox=True, shadow=True)
    leg.get_frame().set_alpha(0.5)
    fig.set_size_inches(18, 4, forward=True)

    # Plot the hyperparameter history
    fig = plt.figure()
    iters = np.arange(kec_h_train.shape[0])
    plt.subplot(2, 1, 1)
    plt.plot(iters, kec_h_train[:, 0], c='c', label='Sensitivity')
    plt.plot(iters, kec_h_train[:, 1:-1].mean(axis=1), c='g',
             label='(Mean) Length Scale')
    if kec_h_train.shape[1] == 3:
        kernel_type = 'Isotropic'
    else:
        kernel_type = 'Anisotropic'
    plt.title('Training History: %s Kernel Hyperparameters' % kernel_type)
    plt.gca().xaxis.set_ticklabels([])
    leg = plt.legend(loc='center', bbox_to_anchor=(0.95, 0.9),
                     fancybox=True, shadow=True)
    leg.get_frame().set_alpha(0.5)
    plt.subplot(2, 1, 2)
    plt.plot(iters, np.log10(kec_h_train[:, -1]), c='b',
             label='Regularization (Scale: $\log_{10}$)')
    plt.title('Training History: Regularization Parameter')
    plt.xlabel('Iterations')
    leg = plt.legend(loc='center', bbox_to_anchor=(0.95, 0.9),
                     fancybox=True, shadow=True)
    leg.get_frame().set_alpha(0.5)
    fig.set_size_inches(18, 4, forward=True)

    # Save all figures and show all figures
    utils.misc.save_all_figures(full_directory,
                                axis_equal=False, tight=False,
                                extension='eps', rcparams=None)
    plt.close("all")

    f = open('%sresults.txt' % full_directory, 'w')
    f.write('There are %d classes for digits: %s\n' % (n_class, str(classes)))
    f.write('Training on %d images\n' % n_train)
    f.write('Testing on %d images\n' % n_test)
    f.write('-----\n')
    f.write('Kernel Embedding Classifier Final Training Configuration:\n')
    f.write('Hyperparameters: %s\n' % str(kec_h))
    f.write('Model Complexity: %f\n' % kec_complexity)
    f.write('Training Accuracy: %f\n' % kec_train_accuracy)
    f.write('Mean Sum of Probabilities: %f\n' % kec_mean_sum_probability)
    f.write('-----\n')
    f.write('Gaussian Process Hyperparameters: %s\n' % str(gpc_h))
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
    for i, c in enumerate(classes):
        dist = cdist(input_mode[[i]], x_train[y_train.ravel() == c],
                     'euclidean').min()
        f.write('Euclidean distance from mode to closest training image '
                'for class %d: %f || Embedding Value: %f\n'
                % (c, dist, kec_mu_modes[i]))
    f.close()


def main():
    """Runs the digit classification task through different scenarios."""
    n_sample = 500
    digits_list = [np.array([1, 4, 9]),
                   np.arange(10),
                   np.array([0, 6]),
                   np.array([0, 8]),
                   np.array([1, 7]),
                   np.array([2, 3]),
                   np.array([3, 5]),
                   np.array([3, 8]),
                   np.array([4, 7]),
                   np.array([4, 9]),
                   np.array([5, 6]),
                   np.array([6, 8]),
                   np.array([6, 9]),
                   np.array([7, 9]),
                   np.array([8, 9]),
                   np.array([0, 6, 8]),
                   np.array([1, 4, 7]),
                   np.array([2, 3, 5]),
                   np.array([3, 5, 8]),
                   np.array([4, 7, 9]),
                   np.array([5, 6, 8]),
                   np.array([6, 8, 9]),
                   np.array([7, 8, 9]),
                   np.array([0, 6, 8, 9]),
                   np.array([1, 4, 7, 9]),
                   np.array([2, 3, 5, 6]),
                   np.array([3, 5, 6, 8]),
                   np.array([4, 5, 7, 9]),
                   np.array([5, 6, 8, 9]),
                   np.array([6, 7, 8, 9])]

    for digits in digits_list:
        raw_mnist_data = create_mnist_data(load_from_tf=False)
        mnist_data = process_mnist_data(*raw_mnist_data,
                                        digits=digits, n_sample=n_sample)
        utils.misc.time_module(digit_classification, *mnist_data)


if __name__ == "__main__":
    main()
