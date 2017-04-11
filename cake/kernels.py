"""
Kernel Function Module.

These are the definitions of commonly used characteristic kernels.
"""
import tensorflow as tf
import numpy as np

tf_float_type = tf.float64
tf_int_type = tf.int64


def sqdist(x_p, x_q, theta):
    """
    Pairwise squared Euclidean distanced between datasets under length scaling.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]

    Returns
    -------
    tensorflow.Tensor
        The pairwise squared Euclidean distance under length scaling (n_p, n_q)
    """
    z_p = tf.divide(x_p, theta)  # (n_p, d)
    z_q = tf.divide(x_q, theta)  # (n_q, d)
    d_pq = tf.matmul(z_p, tf.transpose(z_q))  # (n_p, n_q)
    d_p = tf.reduce_sum(tf.square(z_p), axis=1)  # (n_p,)
    d_q = tf.reduce_sum(tf.square(z_q), axis=1)  # (n_q,)
    return d_p[:, tf.newaxis] - 2 * d_pq + d_q  # (n_p, n_q)


def linear(x_p, x_q, *args):
    """
    Define the linear kernel.

    The linear kernel does not need any hyperparameters.
    Passing hyperparameter arguments do not change the kernel behaviour.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    return tf.matmul(x_p, tf.transpose(x_q))


def gaussian(x_p, x_q, theta):
    """
    Define the Gaussian or squared exponential kernel.

    The hyperparameters are the length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    return tf.exp(-0.5 * sqdist(x_p, x_q, theta))


def s_gaussian(x_p, x_q, theta):
    """
    Define the sensitised Gaussian or squared exponential kernel.

    The hyperparameters are the sensitivity and length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(1 + 1,), (1 + d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    s = theta[0]
    l = theta[1:]
    return tf.multiply(tf.square(s), gaussian(x_p, x_q, l))


def matern3on2(x_p, x_q, theta):
    """
    Define the Matern 3/2 kernel.

    The hyperparameters are the length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    r = tf.sqrt(sqdist(x_p, x_q, theta))
    return tf.multiply(1.0 + r, tf.exp(-r))


def s_matern3on2(x_p, x_q, theta):
    """
    Define the Matern 3/2 kernel.

    The hyperparameters are the sensitivity and length scales of the kernel.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1 + 1,), (1 + d,)]

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    s = theta[0]
    l = theta[1:]
    return tf.multiply(tf.square(s), matern3on2(x_p, x_q, l))


def kronecker_delta(y_p, y_q, *args):
    """
    Define the Kronecker delta kernel.

    The Kronecker delta kernel does not need any hyperparameters.
    Passing hyperparameter arguments do not change the kernel behaviour.

    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)

    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    return tf.cast(tf.equal(y_p, tf.reshape(y_q, [-1])), tf_float_type)


def perceptron(x, w, b):
    """
    Define a single layer perceptron.

    Parameters
    ----------
    x : tensorflow.Tensor
        A dataset of size (n, d)
    w : tensorflow.Tensor
        A weight matrix of size (d, d')
    b : tensorflow.Tensor
        A bias vector of size (d',)

    Returns
    -------
    tensorflow.Tensor
        The perceptron matrix of size (n, d')
    """
    return tf.nn.relu(tf.matmul(x, w) + b)


def linear_mlp1(x_p, x_q, w, b):
    phi_p = perceptron(x_p, w, b)
    phi_q = perceptron(x_q, w, b)
    return linear(phi_p, phi_q)


def gaussian_mlp1(x_p, x_q, theta, w, b):
    phi_p = perceptron(x_p, w, b)
    phi_q = perceptron(x_q, w, b)
    return gaussian(phi_p, phi_q, theta)


def s_gaussian_mlp1(x_p, x_q, theta, w, b):
    phi_p = perceptron(x_p, w, b)
    phi_q = perceptron(x_q, w, b)
    return s_gaussian(phi_p, phi_q, theta)


def linear_mlp2(x_p, x_q, w_1, b_1, w_2, b_2):
    phi_p_1 = perceptron(x_p, w_1, b_1)
    phi_q_1 = perceptron(x_q, w_1, b_1)
    phi_p_2 = perceptron(phi_p_1, w_2, b_2)
    phi_q_2 = perceptron(phi_q_1, w_2, b_2)
    return linear(phi_p_2, phi_q_2)


def gaussian_mlp2(x_p, x_q, theta, w_1, b_1, w_2, b_2):
    phi_p_1 = perceptron(x_p, w_1, b_1)
    phi_q_1 = perceptron(x_q, w_1, b_1)
    phi_p_2 = perceptron(phi_p_1, w_2, b_2)
    phi_q_2 = perceptron(phi_q_1, w_2, b_2)
    return gaussian(phi_p_2, phi_q_2, theta)


def s_gaussian_mlp2(x_p, x_q, theta, w_1, b_1, w_2, b_2):
    phi_p_1 = perceptron(x_p, w_1, b_1)
    phi_q_1 = perceptron(x_q, w_1, b_1)
    phi_p_2 = perceptron(phi_p_1, w_2, b_2)
    phi_q_2 = perceptron(phi_q_1, w_2, b_2)
    return s_gaussian(phi_p_2, phi_q_2, theta)
