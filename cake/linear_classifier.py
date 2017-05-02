"""
Define the base kernel embedding classifier.
"""
import tensorflow as tf
import numpy as np
from .infer import clip_normalize as _clip_normalize
from .infer import classify as _classify
from .infer import decode_one_hot as _decode_one_hot
from .kernels import linear as _linear
from .data_type_def import *


class LinearKernelEmbeddingClassifier():

    def __init__(self):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        self.out_kernel = _linear

    def initialise_deep_parameters(self):
        """Define the deep parameters of the kernel embedding network."""
        with tf.name_scope('deep_parameters'):
            self.deep_var_list = []

    def features(self, x):
        """
        Define the features of the kernel embedding network.

        Parameters
        ----------
        x : tensorflow.Tensor
            An input example of size (n_train, d)

        Returns
        -------
        tensorflow.Tensor
            The features of the kernel embedding classifier
        """
        with tf.name_scope('features'):
            bias = False
            if bias:
                b = tf.ones([tf.shape(x)[0], 1])
                return tf.concat([x, b], axis=1)
            else:
                return x

    def define_summary_graph(self):
        """Setup the summary graph for tensorboard."""
        with tf.name_scope('hyperparameter_summary'):
            theta_str = tf.summary.histogram('theta', self.theta)
            zeta_str = tf.summary.histogram('zeta', self.zeta[0])
            theta_scalar_str = [tf.summary.scalar('theta_%d' % i, self.theta[i]) for i in range(self.d_theta)]
            zeta_scalar_str = tf.summary.scalar('zeta', self.zeta[0])
            self.summary_hypers_str = theta_scalar_str + [zeta_scalar_str] + [theta_str, zeta_str]

        with tf.name_scope('train_summary'):
            train_str_1 = tf.summary.scalar('train_accuracy', self.train_accuracy)
            train_str_2 = tf.summary.scalar('train_cross_entropy_loss', self.train_cross_entropy_loss)
            train_str_3 = tf.summary.scalar('train_cross_entropy_loss_valid', self.train_cross_entropy_loss_valid)
            train_str_4 = tf.summary.scalar('train_msp', self.train_msp)
            train_str_5 = tf.summary.scalar('complexity', self.complexity)
            self.summary_train_str = [train_str_1, train_str_2, train_str_3, train_str_4, train_str_5]

        with tf.name_scope('test_summary'):
            test_str_1 = tf.summary.scalar('test_accuracy', self.test_accuracy)
            test_str_2 = tf.summary.scalar('test_cross_entropy_loss', self.test_cross_entropy_loss)
            test_str_3 = tf.summary.scalar('test_cross_entropy_loss_valid', self.test_cross_entropy_loss_valid)
            test_str_4 = tf.summary.scalar('test_msp', self.test_msp)
            self.summary_test_str = [test_str_1, test_str_2, test_str_3, test_str_4]

    def results(self):
        """Compute relevant results."""
        theta = self.sess.run(self.theta)
        zeta = self.sess.run(self.zeta)

        train_acc = self.sess.run(self.train_accuracy, feed_dict=self.feed_dict)
        train_cel = self.sess.run(self.train_cross_entropy_loss, feed_dict=self.feed_dict)
        train_cel_valid = self.sess.run(self.train_cross_entropy_loss_valid, feed_dict=self.feed_dict)
        train_msp = self.sess.run(self.train_msp, feed_dict=self.feed_dict)
        complexity = self.sess.run(self.complexity, feed_dict=self.feed_dict)

        test_acc = self.sess.run(self.test_accuracy, feed_dict=self.feed_dict)
        test_cel = self.sess.run(self.test_cross_entropy_loss, feed_dict=self.feed_dict)
        test_cel_valid = self.sess.run(self.test_cross_entropy_loss_valid, feed_dict=self.feed_dict)
        test_msp = self.sess.run(self.test_msp, feed_dict=self.feed_dict)

        result = {'training_iterations': self.training_iterations,
                  'theta': theta,
                  'zeta': zeta,
                  'train_acc': train_acc,
                  'train_cel': train_cel,
                  'train_cel_valid': train_cel_valid,
                  'train_msp': train_msp,
                  'complexity': complexity,
                  'test_acc': test_acc,
                  'test_cel': test_cel,
                  'test_cel_valid': test_cel_valid,
                  'test_msp': test_msp}
        return result

    def kernel(self, x_p, x_q, name=None):
        """
        Build a general kernel.

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
        with tf.name_scope('kernel'):
            return self.out_kernel(self.features(x_p), self.features(x_q), self.theta, name=name)

    def fit(self, x_train, y_train, x_test, y_test,
            theta=np.array([1., 1.]),
            zeta=1e-4,
            learning_rate=0.1,
            grad_tol=0.00,
            max_iter=1000,
            n_sgd_batch=None,
            objective='full',
            sequential_batch=False,
            log_hypers=True,
            to_train=True,
            save_step=10,
            tensorboard_directory=None):
        """
        Fit the kernel embedding classifier.

        Parameters
        ----------
        x_train : numpy.ndarray
            The training inputs of size (n_train, d)
        y_train : numpy.ndarray
            The training outputs of size (n_train, 1)
        x_test : numpy.ndarray
            The testing inputs of size (n_test, d)
        y_test : numpy.ndarray
            The testing outputs of size (n_test, 1)
        theta : numpy.ndarray, optional
            The initial kernel hyperparameters (?,)
        zeta : float or numpy.ndarray, optional
            The initial regularisation parameter (scalar or 1 element array)
        learning_rate : float, optional
            The learning rate for the gradient descent optimiser
        grad_tol : float, optional
            The gradient error tolerance for the stopping criteria
        max_iter : int, optional
            The maximum number of iterations for training
        n_sgd_batch : int, optional
            The number of batches used for stochastic gradient descent
        objective : str, optional
            The training objective ['full', 'cross_entropy_loss', 'complexity']
        log_hypers : bool, optional
            To train over the log-space of the hyperparameters instead
        to_train : bool, optional
            The train the hyperparameters or not
        save_step : int, optional
            The number of steps to wait before saving tensorboard results again
        tensorboard_directory : str, optional
            A directory to store all the tensorboard information

        Returns
        -------
        KernelEmbeddingClassifier
            The trained classifier
        """
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
            self.d_theta = theta.shape[0]

        with tf.name_scope('train_data'):

            # Setup the data
            self.x_train = tf.placeholder(tf_float_type, shape=[None, x_train.shape[1]], name='x_train')
            self.y_train = tf.placeholder(tf_float_type, shape=[None, y_train.shape[1]], name='y_train')
            self.feed_dict = {self.x_train: x_train, self.y_train: y_train}
            self.n_train = tf.shape(self.x_train)[0]

            # Determine the one hot encoding and index form of the training labels
            self.y_train_one_hot = tf.equal(self.y_train, self.classes, name='y_train_one_hot')
            self.y_train_indices = _decode_one_hot(self.y_train_one_hot, name='y_train_indices')

        with tf.name_scope('test_data'):

            # Setup the data
            self.x_test = tf.placeholder(tf_float_type, shape=[None, x_test.shape[1]], name='x_test')
            self.y_test = tf.placeholder(tf_float_type, shape=[None, y_test.shape[1]], name='y_test')
            self.feed_dict.update({self.x_test: x_test, self.y_test: y_test})
            self.n_test = tf.shape(self.x_test)[0]

            # Determine the one hot encoding and index form of the testing labels
            self.y_test_one_hot = tf.equal(self.y_test, self.classes, name='y_test_one_hot')
            self.y_test_indices = _decode_one_hot(self.y_test_one_hot, name='y_test_indices')

        with tf.name_scope('hyperparameters'):
            # Setup the output kernel hyperparameters and regulariser
            if log_hypers:
                self.log_theta = tf.Variable(np.log(theta).astype(np_float_type), name="log_theta")
                self.log_zeta = tf.Variable(np.log(np.atleast_1d(zeta)).astype(np_float_type), name="log_zeta")
                self.theta = tf.exp(self.log_theta, name="theta")
                self.zeta = tf.exp(self.log_zeta, name="zeta")
                var_list = [self.log_zeta]
            else:
                self.theta = tf.Variable(theta.astype(np_float_type), name="theta")
                self.zeta = tf.Variable(np.atleast_1d(zeta).astype(np_float_type), name="zeta")
                var_list = [self.zeta]

            # Setup deep hyperparameters
            self.initialise_deep_parameters()
            var_list += self.deep_var_list

        with tf.name_scope('core_graph'):
            # Setup the core graph
            self._setup_core_graph()

        with tf.name_scope('train_graph'):
            # Setup the training graph
            self._setup_train_graph()

        with tf.name_scope('test_graph'):
            # Setup the testing graph
            self._setup_test_graph()

        with tf.name_scope('optimisation'):
            # Setup the lagrangian objective
            if objective == 'full':
                self.lagrangian = self.train_cross_entropy_loss + self.complexity
            elif objective == 'cross_entropy_loss':
                self.lagrangian = self.train_cross_entropy_loss
            elif objective == 'complexity':
                self.lagrangian = self.complexity
            else:
                raise ValueError('No such objective named %s' % objective)

            self.grad = tf.gradients(self.lagrangian, var_list)

            # Setup the training optimisation program
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train = opt.minimize(self.lagrangian, var_list=var_list)

            # Run the optimisation
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # Merge all the summaries
            if tensorboard_directory:

                self.define_summary_graph()
                self.summary_hypers = tf.summary.merge(self.summary_hypers_str)
                self.summary_train = tf.summary.merge(self.summary_train_str)
                self.summary_test = tf.summary.merge(self.summary_test_str)

                if n_sgd_batch:
                    writer_batch = tf.summary.FileWriter(tensorboard_directory + 'batch/')
                    writer_batch.add_graph(self.sess.graph)
                    writer_whole = tf.summary.FileWriter(tensorboard_directory + 'whole/')
                    writer_whole.add_graph(self.sess.graph)
                else:
                    writer = tf.summary.FileWriter(tensorboard_directory)
                    writer.add_graph(self.sess.graph)

            # Run the optimisation
            print('Starting Training')
            print('Batch size for stochastic gradient descent: %d' % n_sgd_batch) if n_sgd_batch else print('Using full dataset for gradient descent')
            batch_feed_dict = self.feed_dict.copy()
            step = 0
            self.training_iterations = step
            grad_norm = grad_tol + 1 if to_train else 0
            np.set_printoptions(precision=2)
            while grad_norm > grad_tol and step < max_iter:

                # Sample the data batch for this training iteration
                if n_sgd_batch:
                    sgd_indices = np.arange(step * n_sgd_batch, (step + 1) * n_sgd_batch) % self.n if sequential_batch else np.random.choice(self.n, n_sgd_batch, replace=False)
                    batch_feed_dict.update({self.x_train: x_train[sgd_indices], self.y_train: y_train[sgd_indices]})

                # Log and save the progress every so iterations
                if step % save_step == 0:
                    theta = self.sess.run(self.theta)
                    zeta = self.sess.run(self.zeta)

                    train_acc = self.sess.run(self.train_accuracy, feed_dict=batch_feed_dict)
                    train_cel = self.sess.run(self.train_cross_entropy_loss, feed_dict=batch_feed_dict)
                    train_cel_valid = self.sess.run(self.train_cross_entropy_loss_valid, feed_dict=batch_feed_dict)
                    train_msp = self.sess.run(self.train_msp, feed_dict=batch_feed_dict)
                    complexity = self.sess.run(self.complexity, feed_dict=batch_feed_dict)

                    test_acc = self.sess.run(self.test_accuracy, feed_dict=self.feed_dict)
                    test_cel = self.sess.run(self.test_cross_entropy_loss, feed_dict=self.feed_dict)
                    test_cel_valid = self.sess.run(self.test_cross_entropy_loss_valid, feed_dict=self.feed_dict)
                    test_msp = self.sess.run(self.test_msp, feed_dict=self.feed_dict)

                    grad = self.sess.run(self.grad, feed_dict=batch_feed_dict)
                    grad_norm = compute_grad_norm(grad)

                    print('Step %d' % step,
                          '|H:', zeta[0],
                          '|C:', complexity,
                          '|ACC:', train_acc,
                          '|CEL:', train_cel,
                          '|CELV:', train_cel_valid,
                          '|MSP:', train_msp,
                          '|TACC:', test_acc,
                          '|TCEL:', test_cel,
                          '|TCELV:', test_cel_valid,
                          '|TMSP:', test_msp,
                          '|Gradient Norm:', grad_norm)
                    print('Gradient Norms:', np.array([np.max(np.abs(grad_i))
                                                       for grad_i in grad]))

                    if tensorboard_directory:

                        if n_sgd_batch:
                            whole_summary = self.sess.run(self.summary_hypers, feed_dict=self.feed_dict)
                            writer_whole.add_summary(whole_summary, step)
                            whole_summary = self.sess.run(self.summary_train, feed_dict=self.feed_dict)
                            writer_whole.add_summary(whole_summary, step)
                            whole_summary = self.sess.run(self.summary_test, feed_dict=self.feed_dict)
                            writer_whole.add_summary(whole_summary, step)

                            batch_summary = self.sess.run(self.summary_hypers, feed_dict=batch_feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                            batch_summary = self.sess.run(self.summary_train, feed_dict=batch_feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                            batch_summary = self.sess.run(self.summary_test, feed_dict=batch_feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                        else:
                            summary = self.sess.run(self.summary_hypers, feed_dict=batch_feed_dict)
                            writer.add_summary(summary, step)
                            summary = self.sess.run(self.summary_train, feed_dict=batch_feed_dict)
                            writer.add_summary(summary, step)
                            summary = self.sess.run(self.summary_test, feed_dict=batch_feed_dict)
                            writer.add_summary(summary, step)

                # We can still print out the hyperparameters and batch training performance
                else:

                    theta = self.sess.run(self.theta)
                    zeta = self.sess.run(self.zeta)
                    train_acc = self.sess.run(self.train_accuracy, feed_dict=batch_feed_dict)
                    train_cel = self.sess.run(self.train_cross_entropy_loss, feed_dict=batch_feed_dict)
                    train_cel_valid = self.sess.run(self.train_cross_entropy_loss_valid, feed_dict=batch_feed_dict)
                    train_msp = self.sess.run(self.train_msp, feed_dict=batch_feed_dict)
                    complexity = self.sess.run(self.complexity, feed_dict=batch_feed_dict)

                    grad = self.sess.run(self.grad, feed_dict=batch_feed_dict)
                    grad_norm = compute_grad_norm(grad)
                    print('Step %d' % step,
                          '|H:', zeta[0],
                          '|C:', complexity,
                          '|BACC:', train_acc,
                          '|BCEL:', train_cel,
                          '|BCELV:', train_cel_valid,
                          '|BMSP:', train_msp,
                          '|Gradient Norm:', grad_norm,
                          '|Gradient Norms:', np.array([np.max(np.abs(grad_i)) for grad_i in grad]))

                    if tensorboard_directory:

                        if n_sgd_batch:
                            batch_summary = self.sess.run(self.summary_hypers, feed_dict=batch_feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                            batch_summary = self.sess.run(self.summary_train, feed_dict=batch_feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                            batch_summary = self.sess.run(self.summary_test, feed_dict=batch_feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                        else:
                            summary = self.sess.run(self.summary_hypers, feed_dict=batch_feed_dict)
                            writer.add_summary(summary, step)
                            summary = self.sess.run(self.summary_train, feed_dict=batch_feed_dict)
                            writer.add_summary(summary, step)
                            summary = self.sess.run(self.summary_test, feed_dict=batch_feed_dict)
                            writer.add_summary(summary, step)

                # Run a training step
                self.sess.run(train, feed_dict=batch_feed_dict)
                step += 1
                self.training_iterations = step

        return self

    def _setup_core_graph(self):
        """Setup the core computational graph."""
        with tf.name_scope('train_features'):
            self.z_train = self.features(self.x_train)

        with tf.name_scope('test_features'):
            self.z_test = self.features(self.x_test)

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

    def _setup_train_graph(self):
        """Setup the training computational graph."""
        with tf.name_scope('train_decision_probabilities'):
            # The decision probabilities on the training datatrain_cross_entropy_loss
            self.train_p = tf.matmul(self.z_train, self.weights, name='train_p')

            # The clip-normalised valid decision probabilities
            self.train_p_valid = tf.transpose(_clip_normalize(tf.transpose(self.train_p)), name='train_p_valid')

        with tf.name_scope('train_predictions'):
            # The predictions on the training data
            self.train_y = _classify(self.train_p, classes=self.classes, name='train_y')

        with tf.name_scope('train_accuracy'):
            # The training accuracy
            self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.train_y, tf.reshape(self.y_train, [-1])), tf_float_type), name='train_accuracy')

        with tf.name_scope('train_cross_entropy_loss'):
            # The prediction probabilities on the actual label
            train_indices = tf.cast(tf.range(self.n_train), tf_int_type, name='train_indices')
            self.train_p_y = tf.gather_nd(self.train_p, tf.stack([train_indices, self.y_train_indices], axis=1), name='train_p_y')

            # self.train_p_y = np.where()
            # The cross entropy loss over the training data
            self.train_cross_entropy_loss = tf.reduce_mean(tf_info(self.train_p_y), name='train_cross_entropy_loss')

            # The clip-normalised valid decision probabilities on the actual label
            self.train_p_y_valid = tf.gather_nd(self.train_p_valid, tf.stack([train_indices, self.y_train_indices], axis=1), name='train_p_y_valid')

            # The valid cross entropy loss over the training data
            self.train_cross_entropy_loss_valid = tf.reduce_mean(tf_info(self.train_p_y_valid), name='train_cross_entropy_loss_valid')

        with tf.name_scope('other'):
            # Other interesting quantities
            self.train_msp = tf.reduce_mean(tf.reduce_sum(self.train_p, axis=1), name='train_msp')

        with tf.name_scope('complexity'):
            # The model complexity of the classifier
            self.complexity = self._define_complexity(name='complexity')

    def _setup_test_graph(self):
        """Setup the testing computational graph."""
        with tf.name_scope('test_decision_probabilities'):
            # Decision Probability
            self.test_p = tf.matmul(self.z_test, self.weights, name='test_p')

            # The clip-normalised valid decision probabilities
            self.test_p_valid = tf.transpose(_clip_normalize(tf.transpose(self.test_p)), name='test_p_valid')

        with tf.name_scope('test_predictions'):
            # Prediction
            self.test_y = _classify(self.test_p, classes=self.classes, name='test_y')

        with tf.name_scope('test_accuracy'):
            # The testing accuracy
            self.test_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.test_y, tf.reshape(self.y_test, [-1])), tf_float_type), name='test_accuracy')

        with tf.name_scope('test_cross_entropy_loss'):
            # The prediction probabilities on the actual label
            test_indices = tf.cast(tf.range(self.n_test), tf_int_type, name='indices')
            self.test_p_y = tf.gather_nd(self.test_p, tf.stack([test_indices, self.y_test_indices], axis=1), name='test_p_y')

            # The cross entropy loss over the testing data
            self.test_cross_entropy_loss = tf.reduce_mean(tf_info(self.test_p_y), name='test_cross_entropy_loss')

            # The clip-normalised valid decision probabilities on the actual label
            self.test_p_y_valid = tf.gather_nd(self.test_p_valid, tf.stack([test_indices, self.y_test_indices], axis=1), name='test_p_y_valid')

            # The valid cross entropy loss over the testing data
            self.test_cross_entropy_loss_valid = tf.reduce_mean(tf_info(self.test_p_y_valid), name='test_cross_entropy_loss_valid')

        with tf.name_scope('others'):
            # Other interesting quantities
            self.test_msp = tf.reduce_mean(tf.reduce_sum(self.test_p, axis=1), name='test_msp')

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


def compute_grad_norm(grad):
    """
    Compute the L1 norm of a list of gradients in any shape.

    Parameters
    ----------
    grad : list
        The list of gradients of any shape

    Returns
    -------
    float
        The L1 norm of the gradient
    """
    return np.max([np.max(np.abs(grad_i)) for grad_i in grad])


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
