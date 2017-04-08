"""
Kernel Function Module.

These are the definitions of commonly used characteristic kernels.
"""
import tensorflow as tf

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

    The Kronecker delta does not need any hyperparameters.
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