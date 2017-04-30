import numpy as np
from experiments import run_experiment


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

name = 'mnist'

x_train, y_train, images_train, x_test, y_test, images_test = \
    create_mnist_data()

x_train = x_train[:5000]
y_train = y_train[:5000]

run_experiment(x_train, y_train, x_test, y_test,
               name='%s_5000_anisotropic_full_training' % name,
               s_init=1.,
               fix_s=False,
               l_init=np.ones(x_train.shape[1]),
               zeta_init=1e-4,
               learning_rate=0.1,
               grad_tol=0.00,
               max_iter=300,
               n_sgd_batch=None,
               n_train_limit=10000,
               objective='full',
               sequential_batch=False,
               log_hypers=True,
               to_train=True,
               save_step=1)

run_experiment(x_train, y_train, x_test, y_test,
               name='%s_5000_anisotropic_cross_entropy_loss_training' % name,
               s_init=1.,
               fix_s=False,
               l_init=np.ones(x_train.shape[1]),
               zeta_init=1e-4,
               learning_rate=0.1,
               grad_tol=0.00,
               max_iter=300,
               n_sgd_batch=None,
               n_train_limit=10000,
               objective='cross_entropy_loss',
               sequential_batch=False,
               log_hypers=True,
               to_train=True,
               save_step=1)

run_experiment(x_train, y_train, x_test, y_test,
               name='%s_5000_anisotropic_complexity_training' % name,
               s_init=1.,
               fix_s=False,
               l_init=np.ones(x_train.shape[1]),
               zeta_init=1e-4,
               learning_rate=0.1,
               grad_tol=0.00,
               max_iter=300,
               n_sgd_batch=None,
               n_train_limit=10000,
               objective='complexity',
               sequential_batch=False,
               log_hypers=True,
               to_train=True,
               save_step=1)
