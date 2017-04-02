"""
Kernel Function Module.

These are the definitions of commonly used characteristic kernels.
"""
import tensorflow as tf

float_type = tf.float32


def sqdist(x_p, x_q, theta):
    z_p = tf.divide(x_p, theta)  # (n_p, d)
    z_q = tf.divide(x_q, theta)  # (n_q, d)
    d_pq = tf.matmul(z_p, tf.transpose(z_q))  # (n_p, n_q)
    d_p = tf.reduce_sum(tf.square(z_p), axis=1)  # (n_p,)
    d_q = tf.reduce_sum(tf.square(z_q), axis=1)  # (n_q,)
    return d_p[:, tf.newaxis] - 2 * d_pq + d_q  # (n_p, n_q)


def gaussian(x_p, x_q, theta):
    return tf.exp(-0.5 * sqdist(x_p, x_q, theta))


def s_gaussian(x_p, x_q, theta):
    s = theta[0]
    l = theta[1:]
    return tf.multiply(tf.square(s), gaussian(x_p, x_q, l))


def matern3on2(x_p, x_q, theta):
    r = tf.sqrt(sqdist(x_p, x_q, theta))
    return tf.multiply(1.0 + r, tf.exp(-r))


def s_matern3on2(x_p, x_q, theta):
    s = theta[0]
    l = theta[1:]
    return tf.multiply(tf.square(s), matern3on2(x_p, x_q, l))


def kronecker_delta(y_p, y_q, *args):
    # Assume both are 2d arrays with size (?, 1)
    return tf.cast(tf.equal(y_p, tf.reshape(y_q, [-1])), float_type)

# def general_kronecker_delta(y_p, y_q, theta):
#     t = theta[0]
#     s = tf.array([theta, 1]).ravel()
#     tf.equal(y_p, tf.reshape(y_q, [-1]))
