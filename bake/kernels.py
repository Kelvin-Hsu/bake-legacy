"""
Kernel Function Module.

These are the definitions of commonly used characteristic kernels.
"""
import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf
from scipy.stats import logistic


# TO DO: Add the following kernels
#   Rational Quadratic Kernel
#   Linear Kernel
#   Periodic Kernel
#   Locally Periodic Kernel
#   Matern Kernels
#   Chi-Squared Kernel


def s_gaussian(x_p, x_q, theta):
    s = theta[0]
    l = theta[1:]
    return s**2 * gaussian(x_p, x_q, l)


def s_matern3on2(x_p, x_q, theta):
    s = theta[0]
    l = theta[1:]
    return s**2 * matern3on2(x_p, x_q, l)


def gaussian(x_p, x_q, theta):
    """
    Defines the Gaussian or squared exponential kernel.

    The hyperparameters are the length scales of the kernel. There can either
    be m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    # c = np.prod(np.sqrt(2*np.pi)*theta*np.ones(x_p.shape[1]))
    if x_q is None:
        return np.ones(x_p.shape[0])
    return np.exp(-0.5 * cdist(x_p/theta, x_q/theta, 'sqeuclidean'))


def laplace(x_p, x_q, theta):
    """
    Defines the Laplace or exponential kernel.

    The hyperparameters are the length scales of the kernel. There can either
    be m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    return np.exp(-cdist(x_p/theta, x_q/theta, 'euclidean'))


def matern3on2(x_p, x_q, theta):
    """
    Defines the Matern 3/2 kernel.

    The hyperparameters are the length scales of the kernel. There can either
    be m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    r = cdist(x_p/theta, x_q/theta, 'euclidean')
    return (1.0 + r) * np.exp(-r)


def kronecker_delta(x_p, x_q, *args):
    """
    Defines the Kronecker delta kernel.

    The Kronecker delta does not need any hyperparameters. Passing
    hyperparameter arguments do not change the kernel behaviour.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    return (cdist(x_p, x_q, 'sqeuclidean') == 0).astype(float)


def general_kronecker_delta(x_p, x_q, theta):
    """
    Defines the general Kronecker delta kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    s = np.array([theta, 1]).ravel()
    return s[(cdist(x_p, x_q) == 0).astype(int)].astype(float)


def logistic_kronecker_delta(x_p, x_q, theta):
    """
    Defines the logistic Kronecker delta kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    s = np.array([1 / (1 + np.exp(theta)), 1]).ravel()
    return s[(cdist(x_p, x_q) == 0).astype(int)].astype(float)


def tf_sq_dist(x_p, x_q, theta):
    z_p = tf.divide(x_p, theta)  # (n_p, d)
    z_q = tf.divide(x_q, theta)  # (n_q, d)
    d_pq = tf.matmul(z_p, tf.transpose(z_q))  # (n_p, n_q)
    d_p = tf.reduce_sum(tf.square(z_p), axis=1)  # (n_p,)
    d_q = tf.reduce_sum(tf.square(z_q), axis=1)  # (n_q,)
    return tf.transpose(d_p + tf.transpose(-2 * d_pq + d_q))


def tf_gaussian(x_p, x_q, theta):
    return tf.exp(-0.5 * tf_sq_dist(x_p, x_q, theta))


def tf_s_gaussian(x_p, x_q, theta):
    s = theta[0]
    l = theta[1:]
    return tf.multiply(tf.square(s), tf_gaussian(x_p, x_q, l))


def tf_matern3on2(x_p, x_q, theta):
    r = tf.sqrt(tf_sq_dist(x_p, x_q, theta))
    return tf.multiply(1.0 + r, tf.exp(-r))


def tf_s_matern3on2(x_p, x_q, theta):
    s = theta[0]
    l = theta[1:]
    return tf.multiply(tf.square(s), tf_matern3on2(x_p, x_q, l))


gaussian.tf = tf_gaussian
s_gaussian.tf = tf_s_gaussian
matern3on2.tf = tf_matern3on2
s_matern3on2.tf = tf_s_matern3on2