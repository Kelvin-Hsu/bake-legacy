"""
Define the base kernel embedding classifier.
"""
import tensorflow as tf
import numpy as np
from cake.infer import clip_normalize as _clip_normalize
from cake.infer import classify as _classify
from cake.infer import decode_one_hot as _decode_one_hot
from cake.data_type_def import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class MNISTLinearKernelEmbeddingClassifier():

    def __init__(self):
        """Initialize the classifier."""
        pass

    def initialise_deep_parameters(self):
        """Define the deep parameters of the kernel embedding network."""
        with tf.name_scope('all_parameters'):

            tf.set_random_seed(0)
            self.zeta_init = 1.
            self.log_zeta = tf.Variable(np.log(np.atleast_1d(self.zeta_init)).astype(np_float_type), name="log_zeta")
            self.zeta = tf.exp(self.log_zeta, name="zeta")

            self.w_conv_1 = weight_variable([5, 5, 1, 32])
            self.b_conv_1 = bias_variable([32])

            self.w_conv_2 = weight_variable([5, 5, 32, 64])
            self.b_conv_2 = bias_variable([64])

            self.w_fc_1 = weight_variable([7 * 7 * 64, 1024])
            self.b_fc_1 = bias_variable([1024])

            self.var_list = [self.log_zeta, self.w_conv_1, self.b_conv_1, self.w_conv_2, self.b_conv_2, self.w_fc_1, self.b_fc_1]

            self.dropout = tf.placeholder(tf_float_type)

    def features(self, x):

        with tf.name_scope('features'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            # (n, 28, 28, 32)
            h_conv_1 = tf.nn.relu(conv2d(x_image, self.w_conv_1) + self.b_conv_1)
            # (n, 14, 14, 32)
            h_pool_1 = max_pool_2x2(h_conv_1)

            # (n, 14, 14, 64)
            h_conv_2 = tf.nn.relu(conv2d(h_pool_1, self.w_conv_2) + self.b_conv_2)
            # (n, 7, 7, 64)
            h_pool_2 = max_pool_2x2(h_conv_2)

            # (n, 7 * 7 * 64)
            h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
            # (n, 1024)
            h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, self.w_fc_1) + self.b_fc_1)

            # (n, 1024)
            h_fc_1_dropout = tf.nn.dropout(h_fc_1, self.dropout)
            return h_fc_1_dropout

    def fit(self, x_train, y_train, x_test, y_test,
            learning_rate=0.1,
            dropout=0.5,
            grad_tol=0.00,
            max_iter=60000,
            n_sgd_batch=2048,
            objective='full',
            sequential_batch=False,
            to_train=True,
            save_step=1,
            directory='./'):

        with tf.name_scope('class_data'):

            # Determine the classes
            classes = np.unique(y_train)
            class_indices = np.arange(classes.shape[0])
            self.classes = tf.cast(tf.constant(classes), tf_float_type, name='classes')
            self.class_indices = tf.cast(tf.constant(class_indices), tf_int_type, name='class_indices')
            self.n_classes = classes.shape[0]
            self.n = x_train.shape[0]
            self.d = x_train.shape[1]
            self.n_features = self.d

        with tf.name_scope('core_graph'):
            # Setup the core graph
            self._setup_core_graph()

        with tf.name_scope('query_graph'):
            # Setup the training graph
            self._setup_query_graph()

        with tf.name_scope('optimisation'):
            # Setup the lagrangian objective
            if objective == 'full':
                self.lagrangian = self.query_cross_entropy_loss + self.complexity
            elif objective == 'cross_entropy_loss':
                self.lagrangian = self.query_cross_entropy_loss
            elif objective == 'complexity':
                self.lagrangian = self.complexity
            else:
                raise ValueError('No such objective named %s' % objective)

            self.grad = tf.gradients(self.lagrangian, self.var_list)

            # Setup the training optimisation program
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train = opt.minimize(self.lagrangian, var_list=self.var_list)

            # Run the optimisation
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # Run the optimisation
            print('Starting Training')
            print('Batch size for stochastic gradient descent: %d' % n_sgd_batch) if n_sgd_batch else print('Using full dataset for gradient descent')
            self.training_iterations = 0
            batch_grad_norm = grad_tol + 1 if to_train else 0
            np.set_printoptions(precision=2)
            test_feed_dict = {self.x_train: x_train, self.y_train: y_train, self.x_query: x_test, self.y_query: y_test, self.dropout: 1.0}

            while batch_grad_norm > grad_tol and self.training_iterations < max_iter:

                # Sample the data batch for this training iteration
                if n_sgd_batch:
                    sgd_indices = np.arange(self.training_iterations * n_sgd_batch, (self.training_iterations + 1) * n_sgd_batch) % self.n if sequential_batch else np.random.choice(self.n, n_sgd_batch, replace=False)
                    x_batch = x_train[sgd_indices]
                    y_batch = y_train[sgd_indices]
                    batch_feed_dict = {self.x_train: x_batch, self.y_train: y_batch, self.x_query: x_batch, self.y_query: y_batch, self.dropout: dropout}
                    batch_test_feed_dict = {self.x_train: x_batch, self.y_train: y_batch, self.x_query: x_test, self.y_query: y_test, self.dropout: 1.0}

                zeta = self.sess.run(self.zeta)
                w_conv_1 = self.sess.run(self.w_conv_1)
                b_conv_1 = self.sess.run(self.b_conv_1)
                w_conv_2 = self.sess.run(self.w_conv_2)
                b_conv_2 = self.sess.run(self.b_conv_2)
                w_fc_1 = self.sess.run(self.w_fc_1)
                b_fc_1 = self.sess.run(self.b_fc_1)

                batch_acc = self.sess.run(self.query_accuracy, feed_dict=batch_feed_dict)
                batch_cel = self.sess.run(self.query_cross_entropy_loss, feed_dict=batch_feed_dict)
                batch_cel_valid = self.sess.run(self.query_cross_entropy_loss_valid, feed_dict=batch_feed_dict)
                batch_msp = self.sess.run(self.query_msp, feed_dict=batch_feed_dict)
                batch_complexity = self.sess.run(self.complexity, feed_dict=batch_feed_dict)

                batch_grad = self.sess.run(self.grad, feed_dict=batch_feed_dict)
                batch_grad_norms = np.array([np.max(np.abs(grad_i)) for grad_i in batch_grad])
                batch_grad_norm = np.max(batch_grad_norms)

                print('Step %d' % self.training_iterations,
                      '|Reg:', zeta[0],
                      '|BC:', batch_complexity,
                      '|BACC:', batch_acc,
                      '|BCEL:', batch_cel,
                      '|BCELV:', batch_cel_valid,
                      '|BMSP:', batch_msp,
                      '|Batch Gradient Norms:', batch_grad_norms)

                np.savez('%sbatch_info_%d.npz' % (directory, self.training_iterations),
                         zeta=zeta,
                         w_conv_1=w_conv_1,
                         b_conv_1=b_conv_1,
                         w_conv_2=w_conv_2,
                         b_conv_2=b_conv_2,
                         w_fc_1=w_fc_1,
                         b_fc_1=b_fc_1,
                         batch_acc=batch_acc,
                         batch_cel=batch_cel,
                         batch_cel_valid=batch_cel_valid,
                         batch_msp=batch_msp,
                         batch_complexity=batch_complexity,
                         batch_grad_norms=batch_grad_norms,
                         batch_grad_norm=batch_grad_norm)

                # Log and save the progress every so iterations
                if self.training_iterations % save_step == 0:

                    test_acc = self.sess.run(self.query_accuracy, feed_dict=batch_test_feed_dict)
                    test_cel = self.sess.run(self.query_cross_entropy_loss, feed_dict=batch_test_feed_dict)
                    test_cel_valid = self.sess.run(self.query_cross_entropy_loss_valid, feed_dict=batch_test_feed_dict)
                    test_msp = self.sess.run(self.query_msp, feed_dict=batch_test_feed_dict)

                    print('Step %d' % self.training_iterations,
                          '|TACC:', test_acc,
                          '|TCEL:', test_cel,
                          '|TCELV:', test_cel_valid,
                          '|TMSP:', test_msp)

                    np.savez('%stest_info_%d.npz' % (directory, self.training_iterations),
                             zeta=zeta,
                             w_conv_1=w_conv_1,
                             b_conv_1=b_conv_1,
                             w_conv_2=w_conv_2,
                             b_conv_2=b_conv_2,
                             w_fc_1=w_fc_1,
                             b_fc_1=b_fc_1,
                             batch_acc=batch_acc,
                             batch_cel=batch_cel,
                             batch_cel_valid=batch_cel_valid,
                             batch_msp=batch_msp,
                             batch_complexity=batch_complexity,
                             batch_grad_norms=batch_grad_norms,
                             batch_grad_norm=batch_grad_norm,
                             test_acc=test_acc,
                             test_cel=test_cel,
                             test_cel_valid=test_cel_valid,
                             test_msp=test_msp)

                # Run a training step
                for i in range(100):
                    self.sess.run(train, feed_dict=batch_feed_dict)
                    self.training_iterations += 1
                    print('\t%d' % i)

        return self

    def _setup_core_graph(self):
        """Setup the core computational graph."""
        with tf.name_scope('train_data'):

            # Setup the data
            self.x_train = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_train')
            self.y_train = tf.placeholder(tf_float_type, shape=[None, 1], name='y_train')
            self.n_train = tf.shape(self.x_train)[0]

            # Determine the one hot encoding and index form of the training labels
            self.y_train_one_hot = tf.equal(self.y_train, self.classes, name='y_train_one_hot')
            self.y_train_indices = _decode_one_hot(self.y_train_one_hot, name='y_train_indices')

        with tf.name_scope('train_features'):
            self.z_train = self.features(self.x_train)

        with tf.name_scope('regularisation_matrix'):
            # The regulariser matrix
            i = tf.cast(tf.eye(tf.shape(self.z_train)[1]), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n_train, tf_float_type), tf.multiply(self.zeta, i), name='reg')

        with tf.name_scope('weights'):
            zt = tf.transpose(self.z_train)
            z = self.z_train
            ztz_reg = tf.matmul(zt, z) + reg
            self.chol_ztz_reg = tf.cholesky(ztz_reg, name='chol_ztz_reg')
            self.weights = tf.cholesky_solve(self.chol_ztz_reg, tf.matmul(zt, tf.cast(self.y_train_one_hot, tf_float_type)), name='weights')

        with tf.name_scope('complexity'):
            # The model complexity of the classifier
            self.complexity = self._define_complexity(name='complexity')

    def _setup_query_graph(self):
        """Setup the querying computational graph."""
        with tf.name_scope('query_input'):

            self.x_query = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_query')
            self.y_query = tf.placeholder(tf_float_type, shape=[None, 1], name='y_query')
            self.n_query = tf.shape(self.x_query)[0]
            self.y_query_one_hot = tf.equal(self.y_query, self.classes, name='y_query_one_hot')
            self.y_query_indices = _decode_one_hot(self.y_query_one_hot, name='y_query_indices')

        with tf.name_scope('query_features'):

            self.z_query = self.features(self.x_query)

        with tf.name_scope('query_decision_probabilities'):
            # The decision probabilities
            self.query_p = tf.matmul(self.z_query, self.weights, name='query_p')

            # The clip-normalised valid decision probabilities
            self.query_p_valid = tf.transpose(_clip_normalize(tf.transpose(self.query_p)), name='query_p_valid')

        with tf.name_scope('query_predictions'):
            # The predictions
            self.query_y = _classify(self.query_p, classes=self.classes, name='query_y')

        with tf.name_scope('query_accuracy'):
            # The accuracy
            self.query_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.query_y, tf.reshape(self.y_query, [-1])), tf_float_type), name='query_accuracy')

        with tf.name_scope('query_cross_entropy_loss'):

            # The prediction probabilities on the actual label
            query_indices = tf.cast(tf.range(self.n_query), tf_int_type, name='query_indices')
            self.query_p_y = tf.gather_nd(self.query_p, tf.stack([query_indices, self.y_query_indices], axis=1), name='query_p_y')

            # self.query_p_y = np.where()
            # The cross entropy loss over the querying data
            self.query_cross_entropy_loss = tf.reduce_mean(tf_info(self.query_p_y), name='query_cross_entropy_loss')

            # The clip-normalised valid decision probabilities on the actual label
            self.query_p_y_valid = tf.gather_nd(self.query_p_valid, tf.stack([query_indices, self.y_query_indices], axis=1), name='query_p_y_valid')

            # The valid cross entropy loss over the querying data
            self.query_cross_entropy_loss_valid = tf.reduce_mean(tf_info(self.query_p_y_valid), name='query_cross_entropy_loss_valid')

        with tf.name_scope('other'):
            # Other interesting quantities
            self.query_msp = tf.reduce_mean(tf.reduce_sum(self.query_p, axis=1), name='query_msp')

    def _define_complexity(self, name='complexity_definition'):
        """
        Define the kernel embedding classifier model complexity.

        Returns
        -------
        float
            The model complexity
        """
        with tf.name_scope(name):
            return tf.sqrt(tf.reduce_sum(tf.square(self.weights)))

def tf_info(p):
    """
    Compute information.

    Parameters
    ----------
    p : tensorflow.Tensor
        A tensor of probabilities of any shape

    Returns
    -------
    tensorflow.Tensor
        A tensor of information of the same shape as the input probabilities
    """
    return - tf.log(tf.clip_by_value(p, 1e-15, np.inf))
