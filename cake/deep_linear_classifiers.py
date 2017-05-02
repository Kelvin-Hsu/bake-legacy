"""
Define deep linear kernel embedding classifier.
"""
import tensorflow as tf
from .linear_classifier import LinearKernelEmbeddingClassifier


def perceptron(x, w, b, name='perceptron'):
    """
    Define a single layer perceptron.
    Parameters
    ----------
    x : tensorflow.Tensor
        A dataset of size (n, d)
    w : tensorflow.Variable
        A weight matrix of size (d, d')
    b : tensorflow.Variable
        A bias vector of size (d',)
    Returns
    -------
    tensorflow.Tensor
        The perceptron matrix of size (n, d')
    """
    with tf.name_scope(name):
        return tf.nn.relu(tf.matmul(x, w) + b)


def weight_variable(shape, name='weight'):
    with tf.name_scope(name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape, name='bias'):
    with tf.name_scope(name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


class TwoLayerDeepLinearKernelEmbeddingClassifier(LinearKernelEmbeddingClassifier):

    def __init__(self, d_1=20, d_2=10, seed=0):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        super().__init__()
        self.d_1 = d_1
        self.d_2 = d_2
        self.seed = seed

    def initialise_deep_parameters(self):
        """Define the deep parameters of the kernel embedding network."""
        with tf.name_scope('deep_parameters'):

            tf.set_random_seed(self.seed)
            self.w_1 = weight_variable([self.d, self.d_1])
            self.b_1 = bias_variable([self.d_1])
            self.w_2 = weight_variable([self.d_1, self.d_2])
            self.b_2 = bias_variable([self.d_2])
            self.deep_var_list = [self.w_1, self.b_1, self.w_2, self.b_2]

    def features(self, x):
        """
        Define the deep features of the kernel embedding network.

        Parameters
        ----------
        x : tensorflow.Tensor
            An input example of size (n, d)
        return_all_layers : bool
            To return all the intermediate layers or only the final output

        Returns
        -------
        tensorflow.Tensor or list
            The output of the final layer or a list of outputs of each layer
        """
        with tf.name_scope('deep_features'):
            h1 = perceptron(x, self.w_1, self.b_1)
            z = perceptron(h1, self.w_2, self.b_2)
            return z


class DeepLinearKernelEmbeddingClassifier(LinearKernelEmbeddingClassifier):

    def __init__(self, n_layer=2, hidden_units=[20, 10], seed=0):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        super().__init__()
        self.n_layer = n_layer
        self.hidden_units = hidden_units
        self.seed = seed

    def initialise_deep_parameters(self):
        """Define the deep parameters of the kernel embedding network."""
        with tf.name_scope('deep_parameters'):

            tf.set_random_seed(self.seed)

            self.weight_list = []
            self.bias_list = []

            n_previous_hidden = self.d
            for l in range(self.n_layer):
                n_hidden = self.hidden_units[l]
                self.weight_list.append(weight_variable([n_previous_hidden, n_hidden]))
                self.bias_list.append(bias_variable([n_hidden]))
                n_previous_hidden = n_hidden

            self.deep_var_list = self.weight_list + self.bias_list

    def features(self, x):
        """
        Define the deep features of the kernel embedding network.

        Parameters
        ----------
        x : tensorflow.Tensor
            An input example of size (n, d)
        return_all_layers : bool
            To return all the intermediate layers or only the final output

        Returns
        -------
        tensorflow.Tensor or list
            The output of the final layer or a list of outputs of each layer
        """
        with tf.name_scope('deep_features'):

            h = x
            for l in range(self.n_layer):
                h = perceptron(h, self.weight_list[l], self.bias_list[l])
            return h
