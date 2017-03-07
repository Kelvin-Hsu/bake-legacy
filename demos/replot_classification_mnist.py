import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def create_mnist_data(digits=np.arange(10), n_sample=500, sample_before=True):
    """
    Load a subset of the mnist dataset.

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


def replot(now_string, digits, n_sample=500):

    x_train, y_train, images_train, x_test, y_test, images_test = \
        create_mnist_data(digits=digits, n_sample=n_sample)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    digits_str = ''.join([str(i) for i in digits])
    full_directory = './%s_mnist_digits_%s_with_%d_training_images/' \
                     % (now_string, digits_str, n_train)
    file = np.load('%smnist_training_results.npz' % full_directory)

    classes = file['classes']
    n_class = file['n_class']
    h_min = file['h_min']
    h_max = file['h_max']
    h_init = file['h_init']
    kec_h = file['kec_h']
    gpc_h = file['gpc_h']
    kec_f_train = file['kec_f_train']
    kec_a_train = file['kec_a_train']
    kec_p_train = file['kec_p_train']
    kec_h_train = file['kec_h_train']
    kec_x_modes = file['kec_x_modes']
    kec_mu_modes = file['kec_mu_modes']
    kec_p = file['kec_p']
    svc_p = file['svc_p']
    gpc_p = file['gpc_p']
    kec_y = file['kec_y']
    svc_y = file['svc_y']
    gpc_y = file['gpc_y']
    kec_p_test = file['kec_p_test']
    svc_p_test = file['svc_p_test']
    gpc_p_test = file['gpc_p_test']
    kec_y_test = file['kec_y_test']
    svc_y_test = file['svc_y_test']
    gpc_y_test = file['gpc_y_test']
    # kec_complexity = file['kec_complexity']
    # kec_mean_sum_probability = file['kec_mean_sum_probability']
    kec_train_accuracy = file['kec_train_accuracy']
    svc_train_accuracy = file['svc_train_accuracy']
    gpc_train_accuracy = file['gpc_train_accuracy']
    kec_test_accuracy = file['kec_test_accuracy']
    svc_test_accuracy = file['svc_test_accuracy']
    gpc_test_accuracy = file['gpc_test_accuracy']
    kec_train_log_loss = file['kec_train_log_loss']
    svc_train_los_loss = file['svc_train_los_loss']
    gpc_train_los_loss = file['gpc_train_los_loss']
    kec_test_log_loss = file['kec_test_log_loss']
    svc_test_los_loss = file['svc_test_los_loss']
    gpc_test_los_loss = file['gpc_test_los_loss']
    kec_p_pred = file['kec_p_pred']
    svc_p_pred = file['svc_p_pred']
    gpc_p_pred = file['gpc_p_pred']
    x_train_mean = file['x_train_mean']
    x_train_var = file['x_train_var']
    input_expectance = file['input_expectance']
    input_variance = file['input_variance']
    input_mode = file['input_mode']

    # MISTAKE 1
    kec_f_train = kec_f_train[0]
    kec_a_train = kec_a_train[0]
    kec_p_train = kec_p_train[0]
    kec_h_train = kec_h_train[0]

    n_class = int(n_class)

    # Convert the above into image form
    x_train_mean_images = np.reshape(x_train_mean, (n_class, 28, 28))
    x_train_var_images = np.reshape(x_train_var, (n_class, 28, 28))
    input_expectance_images = np.reshape(input_expectance, (n_class, 28, 28))
    input_variance_images = np.reshape(input_variance, (n_class, 28, 28))
    input_mode_images = np.reshape(input_mode, (n_class, 28, 28))

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

    # If the classifier was anisotropic, show the pixel relevance
    if kec_h.shape[0] == 28 * 28 + 2:
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
                                extension='eps', rcparams=None,
                                verbose=True)
    plt.close("all")


def main():

    now_string = '2017_3_6_21_26_39'
    n_sample = 500
    digits_list = [np.arange(10),
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
                   # np.arange(2),
                   # np.arange(3),
                   # np.arange(4),
                   # np.arange(5),
                   # np.arange(6),
                   # np.arange(7),
                   # np.arange(8),
                   # np.arange(9)]

    for digits in digits_list:
        utils.misc.time_module(replot, now_string, digits, n_sample=n_sample)

if __name__ == "__main__":
    main()