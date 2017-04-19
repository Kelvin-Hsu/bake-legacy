"""
Define deep convolutional kernel embedding classifier.
"""
import tensorflow as tf
from .kernels import s_gaussian as _s_gaussian
from .data_type_def import *
from .base_classifier import KernelEmbeddingClassifier


class DeepConvolutionalKernelEmbeddingClassifier3NoPool(KernelEmbeddingClassifier):

    def __init__(self, kernel=_s_gaussian, width=28, height=28, channels_in=1,
                 channels_1=4, channels_2=8, channels_3=12,
                 k_size_1=5, k_size_2=5, k_size_3=4,
                 stride_1=1, stride_2=2, stride_3=2,
                 weights_std=0.1, bias_init=0.1, seed=0):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        super().__init__()
        self.out_kernel = kernel

        self.width = width
        self.height = height

        self.channels_in = channels_in
        self.channels_1 = channels_1
        self.channels_2 = channels_2
        self.channels_3 = channels_3
        self.k_size_1 = k_size_1
        self.k_size_2 = k_size_2
        self.k_size_3 = k_size_3

        self.stride_1 = stride_1  # output is 28x28
        self.stride_2 = stride_2  # output is 14x14
        self.stride_3 = stride_3  # output is 7x7

        self.weights_std = weights_std
        self.bias_init = bias_init
        self.seed = seed

    def initialise_deep_parameters(self):
        """Define the deep parameters of the kernel embedding network."""
        with tf.name_scope('deep_parameters'):

            tf.set_random_seed(self.seed)
            weights_std = self.weights_std
            bias_init = self.bias_init
            w1_init = tf.cast(tf.truncated_normal([self.k_size_1, self.k_size_1, self.channels_in, self.channels_1], stddev=weights_std), tf_float_type)
            w2_init = tf.cast(tf.truncated_normal([self.k_size_2, self.k_size_2, self.channels_1, self.channels_2], stddev=weights_std), tf_float_type)
            w3_init = tf.cast(tf.truncated_normal([self.k_size_3, self.k_size_3, self.channels_2, self.channels_3], stddev=weights_std), tf_float_type)
            b1_init = tf.cast(tf.ones([self.channels_1]) * bias_init, tf_float_type)
            b2_init = tf.cast(tf.ones([self.channels_2]) * bias_init, tf_float_type)
            b3_init = tf.cast(tf.ones([self.channels_3]) * bias_init, tf_float_type)
            self.w1 = tf.Variable(w1_init, name='w1')
            self.b1 = tf.Variable(b1_init, name='b1')
            self.w2 = tf.Variable(w2_init, name='w2')
            self.b2 = tf.Variable(b2_init, name='b2')
            self.w3 = tf.Variable(w3_init, name='w3')
            self.b3 = tf.Variable(b3_init, name='b3')
            tf.summary.histogram('w1', self.w1)
            tf.summary.histogram('b1', self.b1)
            tf.summary.histogram('w2', self.w2)
            tf.summary.histogram('b2', self.b2)
            tf.summary.histogram('w3', self.w3)
            tf.summary.histogram('b3', self.b3)
            self.deep_var_list = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def features(self, x, return_all_layers=False):
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
            width = self.width
            height = self.height
            stride_1 = self.stride_1
            stride_2 = self.stride_2
            stride_3 = self.stride_3

            # Reshape the input back to an image
            x_image = tf.reshape(x, shape=[-1, width, height, 1], name='x_image')

            # Build the deep layers
            phi_1 = tf.nn.relu(tf.nn.conv2d(x_image, self.w1, strides=[1, stride_1, stride_1, 1], padding='SAME') + self.b1, name='phi_1')
            phi_2 = tf.nn.relu(tf.nn.conv2d(phi_1, self.w2, strides=[1, stride_2, stride_2, 1], padding='SAME') + self.b2, name='phi_2')
            phi_3 = tf.nn.relu(tf.nn.conv2d(phi_2, self.w3, strides=[1, stride_3, stride_3, 1], padding='SAME') + self.b3, name='phi_3')

            width_1 = int(width / stride_1)
            height_1 = int(height / stride_1)
            width_2 = int(width / (stride_1 * stride_2))
            height_2 = int(height / (stride_1 * stride_2))
            width_3 = int(width / (stride_1 * stride_2 * stride_3))
            height_3 = int(height / (stride_1 * stride_2 * stride_3))

            # Reshape the last layer to standard form
            final_width = int(width / (stride_1 * stride_2 * stride_3))
            final_height = int(height / (stride_1 * stride_2 * stride_3))
            n_flat = int(final_width * final_height * self.channels_3)
            z = tf.reshape(phi_3, shape=[-1, n_flat], name='z')

            for i in range(self.channels_1):
                phi_1_i = tf.reshape(phi_1[:, :, :, i], shape=[-1, width_1, height_1, 1], name='phi_1_%d' % i)
                tf.summary.image('phi_1_%d' % i, phi_1_i, max_outputs=3)
            for i in range(self.channels_2):
                phi_2_i = tf.reshape(phi_2[:, :, :, i], shape=[-1, width_2, height_2, 1], name='phi_2_%d' % i)
                tf.summary.image('phi_2_%d' % i, phi_2_i, max_outputs=3)
            for i in range(self.channels_3):
                phi_3_i = tf.reshape(phi_3[:, :, :, i], shape=[-1, width_3, height_3, 1], name='phi_3_%d' % i)
                tf.summary.image('phi_3_%d' % i, phi_3_i, max_outputs=3)

            # Return
            if return_all_layers:
                return [phi_1, phi_2, phi_3, z]
            else:
                return z

    def compute_features(self, x):
        """
        Compute the deep features of the kernel embedding network.

        Parameters
        ----------
        x : numpy.ndarray
            The input of size (n, d)

        Returns
        -------
        list
            A list of the output of each layer, including the final output
        """
        x_input = tf.placeholder(tf_float_type, list(x.shape))
        phi_list = self.features(x_input, return_all_layers=True)
        return self.sess.run(phi_list, feed_dict={x_input: x})