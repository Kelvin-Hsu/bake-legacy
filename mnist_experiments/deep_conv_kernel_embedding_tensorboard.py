import numpy as np
import tensorflow as tf
import datetime
import cake
import os
from sklearn.metrics import log_loss


now = datetime.datetime.now()
now_string = '%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                    now.hour, now.minute, now.second)


def mnist_classification(x_train, y_train, images_train,
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
    print('\n--------------------------------------------------------------\n')
    print('There are %d classes for digits: ' % n_class, classes)
    for c in classes:
        print('There are %d observations for class %d' % (np.sum(y_train == c), c))

    # Report the number of training and testing points used in this test
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    print('Training on %d images' % n_train)
    print('Testing on %d images' % n_test)

    # Specify the parameters
    kernel = cake.kernels.s_gaussian
    theta_init = np.array([10., 50])
    zeta_init = 1e-4
    learning_rate = 0.001
    grad_tol = 0.01
    n_sgd_batch = 50
    max_iter = 2000
    param_string = 'theta_10_50_zeta_en4_lr_0p001_grad_tol_0p1_sgd_50_max_iter_2000'
    # Create full directory
    full_directory = './%s_mnist_%d_tf_tute_arch_%s/' % (now_string, n_train, param_string)
    os.mkdir(full_directory)
    print('Results will be saved in "%s"' % full_directory)

    tensorboard_directory = full_directory + 'tensorboard/'
    os.mkdir(tensorboard_directory)

    # Train the kernel embedding classifier
    kec = cake.MNISTClassifier_TensorFlowTutorialArchitecture(kernel=kernel).fit(
        x_train, y_train, x_test, y_test, theta=theta_init, zeta=zeta_init, learning_rate=learning_rate, grad_tol=grad_tol, max_iter=max_iter, n_sgd_batch=n_sgd_batch, tensorboard_directory=tensorboard_directory)

def create_mnist_data(load_from_tf=True):
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
        x = file['x_train']
        y = file['y_train']
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
    # If all the training dataset is to be used
    if n_sample == 0:
        return x, y, images, x_test, y_test, images_test

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


def main():
    """Runs the digit classification task through different scenarios."""
    n_sample = 5000
    digits_list = [np.arange(10)]

    for digits in digits_list:
        raw_mnist_data = create_mnist_data(load_from_tf=True)
        mnist_data = process_mnist_data(*raw_mnist_data,
                                        digits=digits, n_sample=n_sample)
        mnist_classification(*mnist_data)


if __name__ == "__main__":
    main()