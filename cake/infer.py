"""
Inference Module.

These are the core but simple inference algorithms used by kernel embeddings.
"""
import tensorflow as tf
import numpy as np

tf_float_type = tf.float32
tf_int_type = tf.int32


def expectance(y, w):
    """
    Obtain the expectance from an empirical embedding.

    Parameters
    ----------
    y : tensorflow.Tensor
        The training outputs (n, d_y)
    w : tensorflow.Tensor
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    tensorflow.Tensor
        The conditional expected value of the output (n_q, d_y)
    """
    return tf.transpose(tf.matmul(tf.transpose(y), w))


def variance(y, w):
    """
    Obtain the variance from an empirical embedding.

    Parameters
    ----------
    y : tensorflow.Tensor
        The training outputs (n, d_y)
    w : tensorflow.Tensor
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    tensorflow.Tensor
        The conditional covariance value of the output (n_q, d_y)
    """
    # Compute the expectance (d_y, n_q)
    y_q_exp = expectance(y, w)

    # Compute the expectance of squares (d_y, n_q)
    y_q_exp_sq = expectance(tf.square(y), w)

    # Compute the variance (n_q, d_y)
    return y_q_exp_sq - tf.square(y_q_exp)


def clip_normalize(w):
    """
    Clip-normalise over the first axis of a tensor.

    Parameters
    ----------
    w : tensorflow.Tensor
        Any tensor

    Returns
    -------
    tensorflow.Tensor
        The clip-normalised tensor of the same size as the input
    """
    w_clip = tf.clip_by_value(w, 0, np.inf)
    return tf.divide(w_clip, tf.reduce_sum(w_clip, axis=0))


def classify(p, classes=None):
    """
    Classify or predict based on a discrete probability distribution.

    Parameters
    ----------
    p : tensorflow.Tensor
        Discrete probability distribution of size (n, m)
    classes : tensorflow.Tensor, optional
        The unique class labels of size (m,); the default is [0, ..., m - 1]

    Returns
    -------
    tensorflow.Tensor
        The classification predictions of size (n,)
    """
    if classes is None:
        classes = tf.range(tf.shape(p)[1])
    return tf.gather(classes, tf.cast(tf.argmax(p, axis=1), tf_int_type))


def adjust_prob(p):
    """
    Adjust invalid probabilities for entropy computations.

    Parameters
    ----------
    p : tensorflow.Tensor
        Discrete probability distribution of size (n, m)

    Returns
    -------
        Discrete probability distribution of size (n, m)
    """
    invalid = tf.less_equal(p, 0)
    return tf.where(invalid, tf.cast(tf.ones(tf.shape(p)), tf_float_type), p)


def entropy(p):
    """
    Compute the entropy of a discrete probability distribution.

    Parameters
    ----------
    p : tensorflow.Tensor
        Discrete probability distribution of size (n, m)

    Returns
    -------
    tensorflow.Tensor
        The entropy of size (n,)
    """
    p_adjust = adjust_prob(p)
    return -tf.reduce_sum(tf.multiply(p_adjust, tf.log(p_adjust)), axis=1)
