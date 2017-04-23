"""
Define the base kernel embedding classifier.
"""
import tensorflow as tf
import numpy as np
from .infer import expectance as _expectance
from .infer import variance as _variance
from .infer import clip_normalize as _clip_normalize
from .infer import adjust_prob as _adjust_prob
from .infer import classify as _classify
from .infer import decode_one_hot as _decode_one_hot
from .kernels import s_gaussian as _s_gaussian
from .kernels import kronecker_delta as _kronecker_delta
from .data_type_def import *


class KernelEmbeddingClassifier():

    def __init__(self, kernel=_s_gaussian):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        self.out_kernel = kernel

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
            return x

    def define_summary_graph(self):
        """Setup the summary graph for tensorboard."""
        with tf.name_scope('summary'):
            theta_str = tf.summary.histogram('theta', self.theta)
            zeta_str = tf.summary.histogram('zeta', self.zeta[0])

            train_str_1 = tf.summary.scalar('train_accuracy', self.train_accuracy)
            train_str_2 = tf.summary.scalar('train_cross_entropy_loss', self.train_cross_entropy_loss)
            train_str_3 = tf.summary.scalar('train_cross_entropy_loss_valid', self.train_cross_entropy_loss_valid)
            train_str_4 = tf.summary.scalar('train_msp', self.train_msp)
            train_str_5 = tf.summary.scalar('complexity', self.complexity)

            test_str_1 = tf.summary.scalar('test_accuracy', self.test_accuracy)
            test_str_2 = tf.summary.scalar('test_cross_entropy_loss', self.test_cross_entropy_loss)
            test_str_3 = tf.summary.scalar('test_cross_entropy_loss_valid', self.test_cross_entropy_loss_valid)
            test_str_4 = tf.summary.scalar('test_msp', self.test_msp)

            self.summary_hypers_str = [theta_str, zeta_str]
            self.summary_train_str = [train_str_1, train_str_2, train_str_3, train_str_4, train_str_5]
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

        result = {'theta': theta,
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
            theta=np.array([1., 1.]), zeta=0.01,
            learning_rate=0.01, grad_tol=0.01, max_iter=100, n_sgd_batch=None,
            sequential_batch=False,
            log_hypers=True, to_train=True,
            save_step=100,
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
            The number of batches used for stochastic gradient descent.
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
                var_list = [self.log_theta, self.log_zeta]
            else:
                self.theta = tf.Variable(theta.astype(np_float_type), name="theta")
                self.zeta = tf.Variable(np.atleast_1d(zeta).astype(np_float_type), name="zeta")
                var_list = [self.theta, self.zeta]

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
            n_limit = 2000
            if self.n > n_limit:
                n_basis = 1000
                self._setup_fast_test_graph(n_basis=n_basis)
            else:
                self._setup_test_graph()

        with tf.name_scope('query_graph'):
            # Setup the query graph
            self._setup_query_graph()

        with tf.name_scope('optimisation'):

            # Setup the lagrangian objective
            self.lagrangian = self.complexity
            self.grad = tf.gradients(self.lagrangian, var_list)

            # Setup the training optimisation program
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train = opt.minimize(self.lagrangian, var_list=var_list)
            # train_complexity = opt.minimize(self.complexity, var_list=var_list)

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
            feed_dict = self.feed_dict.copy()
            step = 0
            grad_norm = grad_tol + 1 if to_train else 0
            np.set_printoptions(precision=2)
            while grad_norm > grad_tol and step < max_iter:

                # Sample the data batch for this training iteration
                if n_sgd_batch:
                    sgd_indices = np.arange(step * n_sgd_batch, (step + 1) * n_sgd_batch) % self.n if sequential_batch else np.random.choice(self.n, n_sgd_batch, replace=False)
                    feed_dict.update({self.x_train: x_train[sgd_indices], self.y_train: y_train[sgd_indices]})

                # Log and save the progress every so iterations
                if step % save_step == 0:
                    theta = self.sess.run(self.theta)
                    zeta = self.sess.run(self.zeta)

                    train_acc = self.sess.run(self.train_accuracy, feed_dict=feed_dict)
                    train_cel = self.sess.run(self.train_cross_entropy_loss, feed_dict=feed_dict)
                    train_cel_valid = self.sess.run(self.train_cross_entropy_loss_valid, feed_dict=feed_dict)
                    train_msp = self.sess.run(self.train_msp, feed_dict=feed_dict)
                    complexity = self.sess.run(self.complexity, feed_dict=feed_dict)

                    test_acc = self.sess.run(self.test_accuracy, feed_dict=self.feed_dict)
                    test_cel = self.sess.run(self.test_cross_entropy_loss, feed_dict=self.feed_dict)
                    test_cel_valid = self.sess.run(self.test_cross_entropy_loss_valid, feed_dict=self.feed_dict)
                    test_msp = self.sess.run(self.test_msp, feed_dict=self.feed_dict)

                    grad = self.sess.run(self.grad, feed_dict=feed_dict)
                    grad_norm = compute_grad_norm(grad)

                    print('Step %d' % step,
                          '|H:', theta, zeta[0],
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
                            if self.n < n_limit:
                                whole_summary = self.sess.run(self.summary_train, feed_dict=self.feed_dict)
                                writer_whole.add_summary(whole_summary, step)
                            whole_summary = self.sess.run(self.summary_test, feed_dict=self.feed_dict)
                            writer_whole.add_summary(whole_summary, step)

                            batch_summary = self.sess.run(self.summary_hypers, feed_dict=feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                            batch_summary = self.sess.run(self.summary_train, feed_dict=feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                            batch_summary = self.sess.run(self.summary_test, feed_dict=feed_dict)
                            writer_batch.add_summary(batch_summary, step)
                        else:
                            summary = self.sess.run(self.summary_hypers, feed_dict=feed_dict)
                            writer.add_summary(summary, step)
                            summary = self.sess.run(self.summary_train, feed_dict=feed_dict)
                            writer.add_summary(summary, step)
                            summary = self.sess.run(self.summary_test, feed_dict=feed_dict)
                            writer.add_summary(summary, step)

                # Run a training step
                # if self.sess.run(self.train_accuracy, feed_dict=feed_dict) >= 1.0:
                #     print('Training only on complexity')
                #     self.sess.run(train_complexity, feed_dict=feed_dict)
                # else:
                #     self.sess.run(train, feed_dict=feed_dict)
                self.sess.run(train, feed_dict=feed_dict)

                print('Step %d' % step)
                step += 1

        return self

    def _setup_core_graph(self):
        """Setup the core computational graph."""
        with tf.name_scope('regularisation_matrix'):
            # The regulariser matrix
            i = tf.cast(tf.eye(self.n_train), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n_train, tf_float_type), tf.multiply(self.zeta, i), name='reg')

        with tf.name_scope('output_gram_matrix'):
            # The gram matrix and regularised version thereof for the output space
            self.l = _kronecker_delta(self.y_train, self.y_train, name='l')
            self.l_reg = tf.add(self.l, reg, name='l_reg')
            self.chol_l_reg = tf.cholesky(self.l_reg, name='chol_l_reg')

        with tf.name_scope('input_gram_matrix'):
            # The gram matrix and regularised version thereof for the input space
            self.k = self.kernel(self.x_train, self.x_train, name='k')
            self.k_reg = tf.add(self.k, reg, name='k_reg')
            self.chol_k_reg = tf.cholesky(self.k_reg, name='chol_k_reg')

    def _setup_train_graph(self):
        """Setup the training computational graph."""
        with tf.name_scope('train_embedding_weights'):
            # The embedding weights on the training data
            self.train_k = self.k
            self.train_w = tf.cholesky_solve(self.chol_k_reg, self.train_k, name='train_w')

        with tf.name_scope('train_decision_probabilities'):
            # The decision probabilities on the training datatrain_cross_entropy_loss
            self.train_p = _expectance(tf.cast(tf.equal(self.y_train, self.classes), tf_float_type), self.train_w, name='train_p')
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

            # The cross entropy loss over the training data
            self.train_cross_entropy_loss = tf.reduce_mean(- tf.log(self.train_p_y), name='train_cross_entropy_loss')

            # The clip-normalised valid decision probabilities on the actual label
            self.train_p_y_valid = tf.gather_nd(self.train_p_valid, tf.stack([train_indices, self.y_train_indices], axis=1), name='train_p_y_valid')

            # The valid cross entropy loss over the training data
            self.train_cross_entropy_loss_valid = tf.reduce_mean(- tf.log(self.train_p_y_valid), name='train_cross_entropy_loss_valid')

        with tf.name_scope('other'):
            # Other interesting quantities
            self.train_msp = tf.reduce_mean(tf.reduce_sum(self.train_p, axis=1), name='train_msp')

        with tf.name_scope('complexity'):
            # The model complexity of the classifier
            self.complexity = self._define_complexity(name='complexity')

    def _setup_test_graph(self):
        """Setup the testing computational graph."""
        with tf.name_scope('test_embedding_weights'):
            # Conditional Embedding Weights
            self.test_k = self.kernel(self.x_train, self.x_test, name='test_k')
            self.test_w = tf.cholesky_solve(self.chol_k_reg, self.test_k, name='test_w')

        with tf.name_scope('test_decision_probabilities'):
            # Decision Probability
            self.test_p = _expectance(tf.cast(tf.equal(self.y_train, self.classes), tf_float_type), self.test_w, name='test_p')
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
            self.test_cross_entropy_loss = tf.reduce_mean(- tf.log(tf.clip_by_value(self.test_p_y, 1e-15, np.inf)), name='test_cross_entropy_loss')

            # The clip-normalised valid decision probabilities on the actual label
            self.test_p_y_valid = tf.gather_nd(self.test_p_valid, tf.stack([test_indices, self.y_test_indices], axis=1), name='test_p_y_valid')

            # The valid cross entropy loss over the testing data
            self.test_cross_entropy_loss_valid = tf.reduce_mean(- tf.log(tf.clip_by_value(self.test_p_y_valid, 1e-15, np.inf)), name='test_cross_entropy_loss_valid')

        with tf.name_scope('others'):
            # Other interesting quantities
            self.test_msp = tf.reduce_mean(tf.reduce_sum(self.test_p, axis=1), name='test_msp')

    def _setup_fast_test_graph(self, n_basis=None):

        assert self.out_kernel == _s_gaussian

        print('Using fast test graph with random fourier features')

        sensitivity = self.theta[0]
        length_scale = self.theta[1:]

        with tf.name_scope('random_fourier_feature'):
            d = tf.shape(self.features(self.x_train[:1]))[1]
            shape = d * tf.ones(2)
            w = tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf_float_type, seed=0) / length_scale

            def phi(x, name='random_fourier_features'):
                """
                Define the approximate feature map.

                Parameters
                ----------
                x : tensorflow.Tensor
                    A collection of inputs of size (n, d)

                Returns
                -------
                tensorflow.Tensor
                    The approximate feature map of size (2 * d, n)
                """
                with tf.name_scope(name):
                    z = self.features(x)
                    c = tf.cos(tf.matmul(w, tf.transpose(z))) / tf.sqrt(d)
                    s = tf.sin(tf.matmul(w, tf.transpose(z))) / tf.sqrt(d)
                    return tf.concat([c, s], 0)

        with tf.name_scope('test_decision_probabilities'):

            # The approximate feature matrix (2 * d, n)
            q_train = phi(self.x_train, name='train_random_fourier_features')

            # The reduced kernel
            qqt = tf.matmul(q_train, tf.transpose(q_train), name='rff_reduced_gram_matrix')

            # Compute the regularisation matrix
            i = tf.cast(tf.eye(2 * d), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n_train, tf_float_type), tf.multiply(self.zeta, i), name='reg')
            reg_rff = tf.divide(reg, tf.square(sensitivity), name='sensitivity_adjusted_reg')

            # Compute the cholesky matrix
            l_rff = tf.cholesky(tf.add(qqt, reg_rff), name='reduced_cholesky')

            # The approximate feature matrix for
            q_test = phi(self.x_test, name='test_random_fourier_features')

            # Compute the reduced weights
            w_rff = tf.cholesky_solve(l_rff, q_test, name='reduced_weights')

            # The one hot encoded outputs
            b = tf.cast(tf.equal(self.y_train, self.classes), tf_float_type, name='one_hot_encoded_labels')

            # The transformed one hot encoded outputs
            b_rff = tf.matmul(q, b, name='transformed_one_hot_encoded_labels')

            # Decision Probability
            self.test_p = tf.matmul(tf.transpose(b_rff), w_rff)

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
            self.test_cross_entropy_loss = tf.reduce_mean(- tf.log(tf.clip_by_value(self.test_p_y, 1e-15, np.inf)), name='test_cross_entropy_loss')

            # The clip-normalised valid decision probabilities on the actual label
            self.test_p_y_valid = tf.gather_nd(self.test_p_valid, tf.stack([test_indices, self.y_test_indices], axis=1), name='test_p_y_valid')

            # The valid cross entropy loss over the testing data
            self.test_cross_entropy_loss_valid = tf.reduce_mean(- tf.log(tf.clip_by_value(self.test_p_y_valid, 1e-15, np.inf)), name='test_cross_entropy_loss_valid')

        with tf.name_scope('others'):
            # Other interesting quantities
            self.test_msp = tf.reduce_mean(tf.reduce_sum(self.test_p, axis=1), name='test_msp')

    def _setup_query_graph(self):
        """Setup the query inference computational graph."""
        with tf.name_scope('query_data'):
            # Query on x_train
            self.x_query = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_query')

        with tf.name_scope('query_embedding_weights'):
            # Conditional Embedding Weights
            self.query_k = self.kernel(self.x_train, self.x_query, name='query_k')
            self.query_w = tf.cholesky_solve(self.chol_k_reg, self.query_k, name='query_w')

        with tf.name_scope('query_decision_probabilities'):
            # Decision Probability
            self.query_p = _expectance(tf.cast(tf.equal(self.y_train, self.classes), tf_float_type), self.query_w, name='query_p')
            # The clip-normalised valid decision probabilities
            self.query_p_valid = tf.transpose(_clip_normalize(tf.transpose(self.query_p)), name='query_p_valid')

        with tf.name_scope('query_predictions'):
            # Prediction
            self.query_y = _classify(self.query_p, classes=self.classes, name='query_y')

        with tf.name_scope('query_information_entropy'):
            # Information Entropy
            ind = tf.reshape(self.y_train_indices, [-1])
            self.query_p_field = tf.transpose(tf.gather(tf.transpose(self.query_p), ind), name='query_p_field')  # (n_query, n_train)
            self.query_p_field_adjusted = _adjust_prob(self.query_p_field, name='query_p_field_adjusted')  # (n_query, n_train)
            self.query_u = tf.subtract(0., tf.log(self.query_p_field_adjusted), name='query_u')
            # TODO: Einstein Sum for diagonal of matrix product
            # self.h_query = tf.einsum('ij,ji->i', u, self.w_query)
            self.query_h = tf.diag_part(tf.matmul(self.query_u, self.query_w), name='query_h')
            self.query_h_valid = tf.clip_by_value(self.query_h, 0, np.inf, name='query_h_valid')

        # # Query on y_train
        # self.y_query = tf.placeholder(tf_float_type, shape=[None, 1], name='y_query')
        #
        # # Reverse Inference Gram Matrix
        # self.l_query = _kronecker_delta(self.y_train, self.y_query, name='l_query')
        #
        # # Reverse Conditional Embedding Weights
        # self.v_query = tf.cholesky_solve(self.chol_l_reg, self.l_query, name='v_query')
        #
        # # Reverse Embedding
        # self.mu_xy = tf.matmul(self.kernel(self.x_query, self.x_train), self.v_query, name='mu_xy')
        #
        # # Input Expectance
        # self.x_exp = _expectance(self.x_train, self.v_query, name='x_exp')
        #
        # # Input Variance
        # self.x_var = _variance(self.x_train, self.v_query, name='x_var')

    def _define_complexity(self, complexity='Global Rademacher Complexity', name=None):
        """
        Define the kernel embedding classifier model complexity.

        Returns
        -------
        float
            The model complexity
        """
        with tf.name_scope('complexity_definition'):
            if complexity == 'Embedding Trace':
                return tf.trace(self.train_w, name=name)
            elif complexity == 'Global Rademacher Complexity':
                b = _kronecker_delta(self.y_train, self.classes[:, tf.newaxis])
                v = tf.cholesky_solve(self.chol_k_reg, b)
                wtw = tf.matmul(tf.transpose(v), tf.matmul(self.k, v))
                return tf.sqrt(tf.trace(wtw), name=name)
            else:
                raise ValueError('No complexity measure named "%s"' % complexity)

    def predict_weights(self, x_query):
        """
        Predict the query weights for classification.

        Parameters
        ----------
        x_query : numpy.ndarray
            The query points of size (n_query, d)

        Returns
        -------
        numpy.ndarray
            The query weights of size (n_train, n_query)
        """
        self.feed_dict.update({self.x_query: x_query})
        return self.sess.run(self.query_w, feed_dict=self.feed_dict)

    def predict_proba(self, x_query, normalize=True):
        """
        Predict the probabilities for classification.

        Parameters
        ----------
        x_query : numpy.ndarray
            The query points of size (n_query, d)
        normalize : bool
            Normalize the probabilities properly

        Returns
        -------
        numpy.ndarray
            The query probability estimates of size (n_query, n_class)
        """
        self.feed_dict.update({self.x_query: x_query})
        return self.sess.run(self.query_p_valid if normalize else self.query_p, feed_dict=self.feed_dict)

    def predict(self, x_query):
        """
        Predict the targets for classification.

        Parameters
        ----------
        x_query : numpy.ndarray
            The query points of size (n_query, d)

        Returns
        -------
        numpy.ndarray
            The query targets of size (n_query,)
        """
        self.feed_dict.update({self.x_query: x_query})
        return self.sess.run(self.query_y, feed_dict=self.feed_dict)

    def predict_entropy(self, x_query, clip=True, tensorflow=False):
        """
        Predict the entropy of the predictions.

        Parameters
        ----------
        x_query : numpy.ndarray
            The query points of size (n_query, d)
        clip : bool
            Make sure the entropy estimate is non-negative

        Returns
        -------
        numpy.ndarray
            The query entropy of size (n_query,)
        """
        self.feed_dict.update({self.x_query: x_query})
        if tensorflow:
            return self.sess.run(self.query_h_valid if clip else self.query_h, feed_dict=self.feed_dict)
        else:
            query_u = self.sess.run(self.query_u, feed_dict=self.feed_dict)
            query_w = self.sess.run(self.query_w, feed_dict=self.feed_dict)
            query_h = np.einsum('ij,ji->i', query_u, query_w)
            return np.clip(query_h, 0, np.inf) if clip else query_h

    # def reverse_weights(self, y_query):
    #     """
    #     Compute the weights for the reverse embedding.
    #
    #     Parameters
    #     ----------
    #     y_query : numpy.ndarray
    #         The query outputs to condition the reverse embedding on
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The weights for the reverse embedding (n_train, n_query)
    #     """
    #     self.feed_dict.update({self.y_query: y_query})
    #     return self.sess.run(self.v_query, feed_dict=self.feed_dict)
    #
    # def reverse_embedding(self, y_query, x_query):
    #     """
    #     Evaluate the reverse embedding.
    #
    #     Parameters
    #     ----------
    #     y_query : numpy.ndarray
    #         The query outputs to condition the reverse embedding on
    #     x_query : numpy.ndarray
    #         The query inputs to evaluate the reverse embedding on
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The evaluated reverse embedding of size (n_x_query, n_y_query)
    #     """
    #     self.feed_dict.update({self.x_query: x_query, self.y_query: y_query})
    #     return self.sess.run(self.mu_xy, feed_dict=self.feed_dict)
    #
    # def input_expectance(self):
    #     """
    #     Predict the expectance of the input conditioned on a class.
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The expectance of each feature for each class (n_class, d)
    #     """
    #     return self.sess.run(self.x_exp, feed_dict=self.feed_dict)
    #
    # def input_variance(self):
    #     """
    #     Predict the variance of the input conditioned on a class.
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The variance of each feature for each class (n_class, d)
    #     """
    #     return self.sess.run(self.x_var, feed_dict=self.feed_dict)
    #
    # def input_mode(self, learning_rate=0.01, grad_tol=0.01):
    #     """
    #     Predict the mode of the input conditioned on a class.
    #
    #     Parameters
    #     ----------
    #     learning_rate : float
    #         The learning rate for the gradient descent optimiser
    #     grad_tol : float
    #         The gradient error tolerance for the stopping criteria
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The mode of each feature for each class (n_class, d)
    #     """
    #     raise ValueError('Implementation not complete.')
    #     # TODO: This optimisation does not converge sometimes. Investigate.
    #     # # For each class, compute the mode
    #     # classes = self.sess.run(self.classes)
    #     # x_mode = np.zeros((self.n_classes, self.d))
    #     # for c in range(self.n_classes):
    #     #
    #     #     # Choose an initial starting point
    #     #     self.feed_dict.update(
    #     #         {self.y_query: self.sess.run(self.classes)[:, np.newaxis]})
    #     #     x_init = self.sess.run(self.x_exp, feed_dict=self.feed_dict)
    #     #
    #     #     # Define a candidate and initialise it to the starting point (d,)
    #     #     x_cand = tf.Variable(x_init[c])
    #     #
    #     #     # Inference Gram Matrix
    #     #     k_cand = self.kernel(self.x_train, x_cand[tf.newaxis, :], self.theta)
    #     #
    #     #     # Conditional Embedding Weights
    #     #     w_cand = tf.cholesky_solve(self.chol_k_reg, k_cand)
    #     #
    #     #     # Embedding
    #     #     mu_yx = tf.matmul(_kronecker_delta(self.classes[:, tf.newaxis],
    #     #                                        self.y_train), w_cand)
    #     #
    #     #     # Embedding Gradients
    #     #     grad = tf.gradients(mu_yx, [x_cand])
    #     #     grad_norm = tf.reduce_max(tf.abs(tf.concat(grad, axis=0)))
    #     #
    #     #     # Begin optimisation
    #     #     self.sess.run(tf.global_variables_initializer())
    #     #     print('Starting Mode Decoding for Class %d' % classes[c])
    #     #     tf.assign(x_cand, x_init[c])
    #     #     opt = tf.train.GradientDescentOptimizer(
    #     #         learning_rate=learning_rate)
    #     #     mu = mu_yx[c, 0]
    #     #     print(mu)
    #     #     train = opt.minimize(mu, var_list=[x_cand])
    #     #     grad_eps = grad_tol + 1
    #     #     i = 1
    #     #     while grad_eps > grad_tol:
    #     #         grad_eps = self.sess.run(grad_norm, feed_dict=self.feed_dict)
    #     #         print('Class %d || Step %d || Mode Candidate: '
    #     #               % (classes[c], i), self.sess.run(x_cand),
    #     #               '|| Gradient Norm: %f' % grad_eps)
    #     #         self.sess.run(train, feed_dict=self.feed_dict)
    #     #         i += 1
    #     #     x_mode[c] = self.sess.run(x_cand)
    #     #     print('Mode Decoded for Class %d:' % classes[c], x_mode[c])
    #     #     print('Embedding Value at Mode: %f'
    #     #           % self.sess.run(mu_yx[c, 0], feed_dict=self.feed_dict))
    #     # print('All Modes Decoded: \n_train', x_mode)
    #     # return x_mode


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