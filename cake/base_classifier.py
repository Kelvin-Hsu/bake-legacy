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
            An input example of size (n, d)

        Returns
        -------
        tensorflow.Tensor or list
            The features of the kernel embedding classifier
        """
        with tf.name_scope('features'):
            return x

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

    def fit(self, x, y, x_test, y_test,
            theta=np.array([1., 1.]), zeta=0.01,
            learning_rate=0.01, grad_tol=0.01, max_iter=100, n_sgd_batch=None,
            sequential_batch=False,
            log_hypers=True, to_train=True,
            tensorboard_directory=None):
        """
        Fit the kernel embedding classifier.

        Parameters
        ----------
        x : numpy.ndarray
            The training inputs of size (n, d)
        y : numpy.ndarray
            The training outputs of size (n, 1)
        theta : numpy.ndarray
            The initial kernel hyperparameters (?,)
        zeta : float or numpy.ndarray
            The initial regularisation parameter (scalar or 1 element array)
        learning_rate : float
            The learning rate for the gradient descent optimiser
        grad_tol : float
            The gradient error tolerance for the stopping criteria
        n_sgd_batch : int or None
            The number of batches used for stochastic gradient descent.
        log_hypers : bool
            To train over the log-space of the hyperparameters instead

        Returns
        -------
        bake.Classifier
            The trained classifier
        """
        with tf.name_scope('train_data'):

            # Determine the classes
            classes = np.unique(y)
            class_indices = np.arange(classes.shape[0])
            self.classes = tf.cast(tf.constant(classes), tf_float_type, name='classes')
            self.class_indices = tf.cast(tf.constant(class_indices), tf_int_type, name='class_indices')
            self.n_classes = classes.shape[0]

            # Setup the data
            self.x = tf.placeholder(tf_float_type, shape=[None, x.shape[1]], name='x')
            self.y = tf.placeholder(tf_float_type, shape=[None, y.shape[1]], name='y')
            self.feed_dict = {self.x: x, self.y: y}
            self.n = tf.shape(self.x)[0]
            self.d = x.shape[1]
            n_all = x.shape[0]

            # Determine the one hot encoding and index form of the training labels
            self.y_one_hot = tf.equal(self.y, self.classes, name='y_one_hot')
            self.y_indices = _decode_one_hot(self.y_one_hot, name='y_indices')

        with tf.name_scope('test_data'):

            # Setup the data
            self.x_test = tf.placeholder(tf_float_type, shape=[None, x_test.shape[1]], name='x_test')
            self.y_test = tf.placeholder(tf_float_type, shape=[None, y_test.shape[1]], name='y_test')
            self.feed_dict.update({self.x_test: x_test, self.y_test: y_test})
            self.n_test = tf.shape(self.x_test)[0]

            # Determine the one hot encoding and index form of the training labels
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

            tf.summary.histogram('theta', self.theta)
            tf.summary.histogram('zeta', self.zeta[0])

            # Setup deep hyperparameters
            self.initialise_deep_parameters()
            var_list += self.deep_var_list

        with tf.name_scope('train_graph'):
            # Setup the training graph
            self._setup_train_graph()

        with tf.name_scope('test_graph'):
            # Setup the testing graph
            self._setup_test_graph()

        with tf.name_scope('optimisation'):

            # Setup the lagrangian objective
            self.lagrangian = self.complexity
            self.grad = tf.gradients(self.lagrangian, var_list)

            # Setup the training optimisation program
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train = opt.minimize(self.lagrangian, var_list=var_list)

            # Run the optimisation
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # Merge all the summaries
            if tensorboard_directory:
                merged_summary = tf.summary.merge_all()

                if n_sgd_batch:
                    writer = tf.summary.FileWriter(tensorboard_directory + 'batch/')
                    writer.add_graph(self.sess.graph)
                    writer_whole = tf.summary.FileWriter(tensorboard_directory + 'whole/')
                    writer_whole.add_graph(self.sess.graph)

                else:
                    writer = tf.summary.FileWriter(tensorboard_directory)
                    writer.add_graph(self.sess.graph)

            # Run the optimisation
            print('Starting Training')
            print('Batch size for stochastic gradient descent: %d' % n_sgd_batch) if n_sgd_batch else print('Using full dataset for gradient descent')
            feed_dict = self.feed_dict
            # _train_history = []
            step = 0
            grad_norm_check = grad_tol + 1 if to_train else 0
            np.set_printoptions(precision=2)
            while grad_norm_check > grad_tol and step < max_iter:

                # Sample the data batch for this training iteration
                if n_sgd_batch:
                    sgd_indices = np.arange(step * n_sgd_batch, (step + 1) * n_sgd_batch) % n_all if sequential_batch else np.random.choice(n_all, n_sgd_batch, replace=False)
                    feed_dict = {self.x: x[sgd_indices], self.y: y[sgd_indices], self.x_test: x_test, self.y_test: y_test}

                theta = self.sess.run(self.theta)
                zeta = self.sess.run(self.zeta)
                complexity = self.sess.run(self.complexity, feed_dict=feed_dict)
                cel = self.sess.run(self.cross_entropy_loss, feed_dict=feed_dict)
                cel_valid = self.sess.run(self.cross_entropy_loss_valid, feed_dict=feed_dict)
                acc = self.sess.run(self.train_accuracy, feed_dict=feed_dict)
                t_cel = self.sess.run(self.test_cross_entropy_loss, feed_dict=feed_dict)
                t_cel_valid = self.sess.run(self.test_cross_entropy_loss_valid, feed_dict=feed_dict)
                t_acc = self.sess.run(self.test_accuracy, feed_dict=feed_dict)
                grad = self.sess.run(self.grad, feed_dict=feed_dict)
                grad_norm = compute_grad_norm(grad)
                grad_norm_check = grad_norm

                if tensorboard_directory:
                    if n_sgd_batch and step % 1 == 0:
                        s = self.sess.run(merged_summary, feed_dict=self.feed_dict)
                        writer_whole.add_summary(s, step)
                    s = self.sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, step)

                self.sess.run(train, feed_dict=feed_dict)

                # _train_history.append([step, complexity, acc, cel, cel_valid, t_acc, t_cel, t_cel_valid, grad_norm] + list(np.append(theta, zeta)))
                step += 1
                print('Step %d' % step, '|| H: ', theta, zeta[0], '|| C: ', complexity, '|| ACC: ', acc, '|| CEL: ', cel, '|| CELV: ', cel_valid, '|| TACC: ', t_acc, '|| TCEL: ', t_cel, '|| TCELV: ', t_cel_valid, '|| Gradient Norm: ', grad_norm)
                print('Gradient Norms: ', np.array([np.max(np.abs(grad_i)) for grad_i in grad]))
            # Store train history
            # _train_history = np.array(_train_history)
            # self.train_history = {'iterations': _train_history[:, 0],
            #                       'complexity': _train_history[:, 1],
            #                       'accuracy': _train_history[:, 2],
            #                       'cross_entropy_loss': _train_history[:, 3],
            #                       'valid_cross_entropy_loss': _train_history[:, 4],
            #                       'test_accuracy': _train_history[:, 5],
            #                       'test_cross_entropy_loss': _train_history[:, 6],
            #                       'test_valid_cross_entropy_loss': _train_history[:, 7],
            #                       'gradient_norm': _train_history[:, 8],
            #                       'kernel_hypers': _train_history[:, 9:-1],
            #                       'regularisation': _train_history[:, -1]} if to_train else None

        # Store the optimal hyperparameters
        self.theta_train = self.sess.run(self.theta)
        self.zeta_train = self.sess.run(self.zeta)
        self.steps_train = step
        self.complexity_train = self.sess.run(self.complexity, feed_dict=self.feed_dict)
        self.acc_train = self.sess.run(self.train_accuracy, feed_dict=self.feed_dict)
        self.cel_train = self.sess.run(self.cross_entropy_loss, feed_dict=self.feed_dict)
        self.grad_norm_train = compute_grad_norm(self.sess.run(self.grad, feed_dict=self.feed_dict))
        self.msp_train = self.sess.run(self.msp, feed_dict=self.feed_dict)
        print('Training Finished')
        print('Learned Hyperparameters: ', self.theta_train, self.zeta_train)
        print('Learned Complexity: ', self.complexity_train)
        print('Achieved Training Accuracy: ', self.acc_train)
        print('Achieved Cross Entropy Loss: ', self.cel_train)
        print('Achieved Mean Sum of Probabilities: ', self.msp_train)
        # print(self.sess.run(var_list))

        self.sess.close()
        # with tf.name_scope('query_graph'):
        #     # Setup the query graph
        #     self._setup_query_graph()

        return self

    def _setup_train_graph(self):
        """Setup the training computational graph."""
        with tf.name_scope('regularisation_matrix'):
            # The regulariser matrix
            i = tf.cast(tf.eye(self.n), tf_float_type, name='i')
            reg = tf.multiply(tf.cast(self.n, tf_float_type), tf.multiply(self.zeta, i), name='reg')

        with tf.name_scope('output_gram_matrix'):
            # The gram matrix and regularised version thereof for the output space
            self.l = _kronecker_delta(self.y, self.y, name='l')
            self.l_reg = tf.add(self.l, reg, name='l_reg')
            self.chol_l_reg = tf.cholesky(self.l_reg, name='chol_l_reg')

        with tf.name_scope('input_gram_matrix'):
            # The gram matrix and regularised version thereof for the input space
            self.k = self.kernel(self.x, self.x, name='k')
            self.k_reg = tf.add(self.k, reg, name='k_reg')
            self.chol_k_reg = tf.cholesky(self.k_reg, name='chol_k_reg')

        with tf.name_scope('train_embedding_weights'):
            # The embedding weights on the training data
            self.w = tf.cholesky_solve(self.chol_k_reg, self.k, name='w')

        with tf.name_scope('train_decision_probabilities'):
            # The decision probabilities on the training data
            self.p_pred = _expectance(tf.cast(tf.equal(self.y, self.classes), tf_float_type), self.w, name='p_pred')

        with tf.name_scope('train_predictions'):
            # The predictions on the training data
            self.y_pred = _classify(self.p_pred, classes=self.classes, name='y_pred')

        with tf.name_scope('train_accuracy'):
            # The training accuracy
            self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, tf.reshape(self.y, [-1])), tf_float_type), name='train_accuracy')

        with tf.name_scope('train_cross_entropy_loss'):
            # The prediction probabilities on the actual label
            indices = tf.cast(tf.range(self.n), tf_int_type, name='indices')
            self.p_want = tf.gather_nd(self.p_pred, tf.stack([indices, self.y_indices], axis=1), name='p_want')

            # The cross entropy loss over the training data
            self.cross_entropy_loss = tf.reduce_mean(- tf.log(self.p_want), name='cross_entropy_loss')

            # The clip-normalised valid decision probabilities
            self.p_pred_valid = tf.transpose(_clip_normalize(tf.transpose(self.p_pred)), name='p_pred_valid')

            # The clip-normalised valid decision probabilities on the actual label
            self.p_want_valid = tf.gather_nd(self.p_pred_valid, tf.stack([indices, self.y_indices], axis=1), name='p_want_valid')

            # The valid cross entropy loss over the training data
            self.cross_entropy_loss_valid = tf.reduce_mean(- tf.log(self.p_want_valid), name='cross_entropy_loss_valid')

        with tf.name_scope('complexity'):
            # The model complexity of the classifier
            self.complexity = self._define_complexity(name='complexity')

        with tf.name_scope('other'):
            # Other interesting quantities
            self.pred_constraint = self.p_want - 1 / self.n_classes
            self.prob_constraint = tf.reduce_sum(self.w, axis=0) - 1
            self.msp = tf.reduce_mean(tf.reduce_sum(self.w, axis=0), name='msp')

        tf.summary.scalar('train_accuracy', self.train_accuracy)
        tf.summary.scalar('train_cross_entropy_loss', self.cross_entropy_loss)
        tf.summary.scalar('train_cross_entropy_loss_valid', self.cross_entropy_loss_valid)
        tf.summary.scalar('train_msp', self.msp)
        tf.summary.scalar('train_complexity', self.complexity)

    def _setup_test_graph(self):

        with tf.name_scope('inference_kernel_matrix'):
            # Inference Gram Matrix
            self.k_test = self.kernel(self.x, self.x_test, name='k_test')

        with tf.name_scope('test_embedding_weights'):
            # Conditional Embedding Weights
            self.w_test = tf.cholesky_solve(self.chol_k_reg, self.k_test, name='w_test')

        with tf.name_scope('test_decision_probabilities'):
            # Decision Probability
            self.p_test_pred = _expectance(tf.cast(tf.equal(self.y, self.classes), tf_float_type), self.w_test, name='p_test_pred')

        with tf.name_scope('test_predictions'):
            # Prediction
            self.y_test_pred = _classify(self.p_test_pred, classes=self.classes, name='y_test_pred')

        with tf.name_scope('test_accuracy'):
            # The training accuracy
            self.test_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_test_pred, tf.reshape(self.y_test, [-1])), tf_float_type), name='test_accuracy')

        with tf.name_scope('test_cross_entropy_loss'):
            # The prediction probabilities on the actual label
            test_indices = tf.cast(tf.range(self.n_test), tf_int_type, name='indices')
            self.p_test_want = tf.gather_nd(self.p_test_pred, tf.stack([test_indices, self.y_test_indices], axis=1), name='p_test_want')

            # The cross entropy loss over the training data
            self.test_cross_entropy_loss = tf.reduce_mean(- tf.log(tf.clip_by_value(self.p_test_want, 1e-15, np.inf)), name='test_cross_entropy_loss')

            # The clip-normalised valid decision probabilities
            self.p_test_pred_valid = tf.transpose(_clip_normalize(tf.transpose(self.p_test_pred)), name='p_test_pred_valid')

            # The clip-normalised valid decision probabilities on the actual label
            self.p_test_want_valid = tf.gather_nd(self.p_test_pred_valid, tf.stack([test_indices, self.y_test_indices], axis=1), name='p_test_want_valid')

            # The valid cross entropy loss over the training data
            self.test_cross_entropy_loss_valid = tf.reduce_mean(- tf.log(tf.clip_by_value(self.p_test_want_valid, 1e-15, np.inf)), name='test_cross_entropy_loss_valid')

        with tf.name_scope('others'):
            # Other interesting quantities
            self.msp_test = tf.reduce_mean(tf.reduce_sum(self.w_test, axis=0), name='msp_test')

        tf.summary.scalar('test_accuracy', self.test_accuracy)
        tf.summary.scalar('test_cross_entropy_loss', self.test_cross_entropy_loss)
        tf.summary.scalar('test_cross_entropy_loss_valid', self.test_cross_entropy_loss_valid)
        tf.summary.scalar('test_msp', self.msp_test)

    def _setup_query_graph(self):
        """Setup the query inference computational graph."""
        # Query on x
        self.x_query = tf.placeholder(tf_float_type, shape=[None, self.d], name='x_query')

        # Inference Gram Matrix
        self.k_query = self.kernel(self.x, self.x_query, name='k_query')

        # Conditional Embedding Weights
        self.w_query = tf.cholesky_solve(self.chol_k_reg, self.k_query, name='w_query')

        # Decision Probability
        self.p_query = _expectance(tf.cast(tf.equal(self.y, self.classes), tf_float_type), self.w_query, name='p_query')
        self.p_query_valid = tf.transpose(_clip_normalize(tf.transpose(self.p_query)), name='p_query_valid')

        # Prediction
        self.y_query_pred = _classify(self.p_query, classes=self.classes, name='y_query_pred')

        # Information Entropy
        ind = tf.reshape(self.y_indices, [-1])
        self.p_field = tf.transpose(tf.gather(tf.transpose(self.p_query), ind), name='p_field')
        self.p_field_adjust = _adjust_prob(self.p_field, name='p_field_adjust')  # (n_query, n_train)
        self.u_field = tf.subtract(0., tf.log(self.p_field_adjust), name='u_field')
        # TODO: Einstein Sum for diagonal of matrix product
        # self.h_query = tf.einsum('ij,ji->i', u, self.w_query)
        self.h_query = tf.diag_part(tf.matmul(self.u_field, self.w_query), name='h_query')
        self.h_query_valid = tf.clip_by_value(self.h_query, 0, np.inf, name='h_query_valid')

        # Query on y
        self.y_query = tf.placeholder(tf_float_type, shape=[None, 1], name='y_query')

        # Reverse Inference Gram Matrix
        self.l_query = _kronecker_delta(self.y, self.y_query, name='l_query')

        # Reverse Conditional Embedding Weights
        self.v_query = tf.cholesky_solve(self.chol_l_reg, self.l_query, name='v_query')

        # Reverse Embedding
        self.mu_xy = tf.matmul(self.kernel(self.x_query, self.x), self.v_query, name='mu_xy')

        # Input Expectance
        self.x_exp = _expectance(self.x, self.v_query, name='x_exp')

        # Input Variance
        self.x_var = _variance(self.x, self.v_query, name='x_var')

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
                return tf.trace(self.w, name=name)
            elif complexity == 'Global Rademacher Complexity':
                b = _kronecker_delta(self.y, self.classes[:, tf.newaxis])
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
            The query weights of size (n, n_query)
        """
        self.feed_dict.update({self.x_query: x_query})
        return self.sess.run(self.w_query, feed_dict=self.feed_dict)

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
        return self.sess.run(self.p_query_valid if normalize else self.p_query, feed_dict=self.feed_dict)

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
        return self.sess.run(self.y_query, feed_dict=self.feed_dict)

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
            return self.sess.run(self.h_query_valid if clip else self.h_query, feed_dict=self.feed_dict)
        else:
            u_field = self.sess.run(self.u_field, feed_dict=self.feed_dict)
            w_query = self.sess.run(self.w_query, feed_dict=self.feed_dict)
            h_query = np.einsum('ij,ji->i', u_field, w_query)
            return np.clip(h_query, 0, np.inf) if clip else h_query

    def reverse_weights(self, y_query):
        """
        Compute the weights for the reverse embedding.

        Parameters
        ----------
        y_query : numpy.ndarray
            The query outputs to condition the reverse embedding on

        Returns
        -------
        numpy.ndarray
            The weights for the reverse embedding (n, n_query)
        """
        self.feed_dict.update({self.y_query: y_query})
        return self.sess.run(self.v_query, feed_dict=self.feed_dict)

    def reverse_embedding(self, y_query, x_query):
        """
        Evaluate the reverse embedding.

        Parameters
        ----------
        y_query : numpy.ndarray
            The query outputs to condition the reverse embedding on
        x_query : numpy.ndarray
            The query inputs to evaluate the reverse embedding on

        Returns
        -------
        numpy.ndarray
            The evaluated reverse embedding of size (n_x_query, n_y_query)
        """
        self.feed_dict.update({self.x_query: x_query, self.y_query: y_query})
        return self.sess.run(self.mu_xy, feed_dict=self.feed_dict)

    def input_expectance(self):
        """
        Predict the expectance of the input conditioned on a class.

        Returns
        -------
        numpy.ndarray
            The expectance of each feature for each class (n_class, d)
        """
        return self.sess.run(self.x_exp, feed_dict=self.feed_dict)

    def input_variance(self):
        """
        Predict the variance of the input conditioned on a class.

        Returns
        -------
        numpy.ndarray
            The variance of each feature for each class (n_class, d)
        """
        return self.sess.run(self.x_var, feed_dict=self.feed_dict)

    def input_mode(self, learning_rate=0.01, grad_tol=0.01):
        """
        Predict the mode of the input conditioned on a class.

        Parameters
        ----------
        learning_rate : float
            The learning rate for the gradient descent optimiser
        grad_tol : float
            The gradient error tolerance for the stopping criteria

        Returns
        -------
        numpy.ndarray
            The mode of each feature for each class (n_class, d)
        """
        raise ValueError('Implementation not complete.')
        # TODO: This optimisation does not converge sometimes. Investigate.
        # # For each class, compute the mode
        # classes = self.sess.run(self.classes)
        # x_mode = np.zeros((self.n_classes, self.d))
        # for c in range(self.n_classes):
        #
        #     # Choose an initial starting point
        #     self.feed_dict.update(
        #         {self.y_query: self.sess.run(self.classes)[:, np.newaxis]})
        #     x_init = self.sess.run(self.x_exp, feed_dict=self.feed_dict)
        #
        #     # Define a candidate and initialise it to the starting point (d,)
        #     x_cand = tf.Variable(x_init[c])
        #
        #     # Inference Gram Matrix
        #     k_cand = self.kernel(self.x, x_cand[tf.newaxis, :], self.theta)
        #
        #     # Conditional Embedding Weights
        #     w_cand = tf.cholesky_solve(self.chol_k_reg, k_cand)
        #
        #     # Embedding
        #     mu_yx = tf.matmul(_kronecker_delta(self.classes[:, tf.newaxis],
        #                                        self.y), w_cand)
        #
        #     # Embedding Gradients
        #     grad = tf.gradients(mu_yx, [x_cand])
        #     grad_norm = tf.reduce_max(tf.abs(tf.concat(grad, axis=0)))
        #
        #     # Begin optimisation
        #     self.sess.run(tf.global_variables_initializer())
        #     print('Starting Mode Decoding for Class %d' % classes[c])
        #     tf.assign(x_cand, x_init[c])
        #     opt = tf.train.GradientDescentOptimizer(
        #         learning_rate=learning_rate)
        #     mu = mu_yx[c, 0]
        #     print(mu)
        #     train = opt.minimize(mu, var_list=[x_cand])
        #     grad_eps = grad_tol + 1
        #     i = 1
        #     while grad_eps > grad_tol:
        #         grad_eps = self.sess.run(grad_norm, feed_dict=self.feed_dict)
        #         print('Class %d || Step %d || Mode Candidate: '
        #               % (classes[c], i), self.sess.run(x_cand),
        #               '|| Gradient Norm: %f' % grad_eps)
        #         self.sess.run(train, feed_dict=self.feed_dict)
        #         i += 1
        #     x_mode[c] = self.sess.run(x_cand)
        #     print('Mode Decoded for Class %d:' % classes[c], x_mode[c])
        #     print('Embedding Value at Mode: %f'
        #           % self.sess.run(mu_yx[c, 0], feed_dict=self.feed_dict))
        # print('All Modes Decoded: \n', x_mode)
        # return x_mode


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