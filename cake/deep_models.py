"""
Models Module.
"""
import tensorflow as tf
import numpy as np
from .infer import expectance as _expectance
from .infer import variance as _variance
from .infer import clip_normalize as _clip_normalize
from .infer import adjust_prob as _adjust_prob
from .infer import classify as _classify
from .kernels import s_gaussian as _s_gaussian
from .kernels import kronecker_delta as _kronecker_delta
from sklearn.model_selection import KFold as _KFold

tf_float_type = tf.float32
tf_int_type = tf.int32
np_float_type = np.float32
np_int_type = np.int32


class DKEC():

    def __init__(self, kernel=_s_gaussian):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        self.out_kernel = kernel

    def kernel(self, x_p, x_q, *args):

        Xp = tf.reshape(x_p, shape=[-1, 28, 28, 1])
        Xq = tf.reshape(x_q, shape=[-1, 28, 28, 1])

        K = 4  # first convolutional layer output depth
        L = 8  # second convolutional layer output depth
        M = 12  # third convolutional layer

        W1 = tf.cast(self.w1, tf_float_type)
        B1 = tf.cast(self.b1, tf_float_type)
        W2 = tf.cast(self.w2, tf_float_type)
        B2 = tf.cast(self.b2, tf_float_type)
        W3 = tf.cast(self.w3, tf_float_type)
        B3 = tf.cast(self.b3, tf_float_type)

        # The model
        stride = 1  # output is 28x28
        Y1p = tf.nn.relu(tf.nn.conv2d(Xp, W1, strides=[1, stride, stride, 1],
                                     padding='SAME') + B1)
        stride = 2  # output is 14x14
        Y2p = tf.nn.relu(tf.nn.conv2d(Y1p, W2, strides=[1, stride, stride, 1],
                                     padding='SAME') + B2)
        stride = 2  # output is 7x7
        Y3p = tf.nn.relu(tf.nn.conv2d(Y2p, W3, strides=[1, stride, stride, 1],
                                     padding='SAME') + B3)

        # reshape the output from the third convolution
        # for the fully connected layer
        YYp = tf.reshape(Y3p, shape=[-1, 7 * 7 * M])

        # The model
        stride = 1  # output is 28x28
        Y1q = tf.nn.relu(tf.nn.conv2d(Xq, W1, strides=[1, stride, stride, 1],
                                      padding='SAME') + B1)
        stride = 2  # output is 14x14
        Y2q = tf.nn.relu(tf.nn.conv2d(Y1q, W2, strides=[1, stride, stride, 1],
                                      padding='SAME') + B2)
        stride = 2  # output is 7x7
        Y3q = tf.nn.relu(tf.nn.conv2d(Y2q, W3, strides=[1, stride, stride, 1],
                                      padding='SAME') + B3)

        # print(Y1p, Y2p, Y3p)
        # reshape the output from the third convolution
        # for the fully connected layer
        YYq = tf.reshape(Y3q, shape=[-1, 7 * 7 * M])

        return self.out_kernel(YYp, YYq, self.theta)

    def compute_features(self, x):

        x_input = tf.placeholder(tf_float_type, list(x.shape))

        Xp = tf.reshape(x_input, shape=[-1, 28, 28, 1])
        W1 = tf.cast(self.w1, tf_float_type)
        B1 = tf.cast(self.b1, tf_float_type)
        W2 = tf.cast(self.w2, tf_float_type)
        B2 = tf.cast(self.b2, tf_float_type)
        W3 = tf.cast(self.w3, tf_float_type)
        B3 = tf.cast(self.b3, tf_float_type)

        # The model
        stride = 1  # output is 28x28
        Y1p = tf.nn.relu(tf.nn.conv2d(Xp, W1, strides=[1, stride, stride, 1],
                                     padding='SAME') + B1)
        stride = 2  # output is 14x14
        Y2p = tf.nn.relu(tf.nn.conv2d(Y1p, W2, strides=[1, stride, stride, 1],
                                     padding='SAME') + B2)
        stride = 2  # output is 7x7
        Y3p = tf.nn.relu(tf.nn.conv2d(Y2p, W3, strides=[1, stride, stride, 1],
                                     padding='SAME') + B3)

        return self.sess.run([Y1p, Y2p, Y3p], feed_dict={x_input: x})

    def build_deep_variables(self):

        K = 4  # first convolutional layer output depth
        L = 8  # second convolutional layer output depth
        M = 12  # third convolutional layer

        # 5x5 patch, 1 input channel, K output channels
        self.w1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
        self.b1 = tf.Variable(tf.ones([K]) / 10)
        self.w2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
        self.b2 = tf.Variable(tf.ones([L]) / 10)
        self.w3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
        self.b3 = tf.Variable(tf.ones([M]) / 10)
        self.deep_var_list = \
            [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def fit(self, x, y,
            theta=np.array([1., 1.]), zeta=0.01,
            learning_rate=0.01, grad_tol=0.01, n_sgd_batch=None,
            sequential_batch=False,
            log_hypers=True, to_train=True):
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
        # Determine the classes
        classes = np.unique(y)
        class_indices = np.arange(classes.shape[0])
        self.classes = tf.cast(tf.constant(classes), tf_float_type)
        self.class_indices = tf.cast(tf.constant(class_indices), tf_int_type)
        self.n_classes = classes.shape[0]

        # Setup the data
        self.x = tf.placeholder(tf_float_type, shape=[None, x.shape[1]])
        self.y = tf.placeholder(tf_float_type, shape=[None, y.shape[1]])
        self.feed_dict = {self.x: x, self.y: y}
        self.n = tf.shape(self.x)[0]
        self.d = x.shape[1]
        n_all = x.shape[0]

        # Determine the one hot encoding and index form of the training labels
        self.y_one_hot = tf.equal(self.y, self.classes)
        self.y_indices = \
            tf.cast(tf.reduce_sum(
                tf.where(self.y_one_hot,
                         tf.ones(tf.shape(self.y_one_hot)) * class_indices,
                         tf.zeros(tf.shape(self.y_one_hot))), axis=1),
                    tf_int_type)

        # Setup the hyperparameters
        offset = False
        if log_hypers:
            self.log_theta = tf.Variable(np.log(theta).astype(np_float_type),
                                         name="log_theta")
            self.log_zeta = tf.Variable(
                np.log(np.atleast_1d(zeta)).astype(np_float_type),
                name="log_zeta")
            if offset:
                self.theta_offset = \
                    tf.Variable(np.atleast_1d(0.0).astype(np_float_type))
                d = tf.concat([np.array([0.0]),
                               tf.ones(theta.shape[0] - 1) *
                               self.theta_offset ** 2], axis=0)
                self.theta = tf.exp(self.log_theta) + d
                self.zeta = tf.exp(self.log_zeta)
                var_list = [self.log_theta, self.log_zeta, self.theta_offset]
            else:
                self.theta = tf.exp(self.log_theta, name="theta")
                self.zeta = tf.exp(self.log_zeta, name="zeta")
                var_list = [self.log_theta, self.log_zeta]
        else:
            self.theta = tf.Variable(theta.astype(np_float_type), name="theta")
            self.zeta = tf.Variable(np.atleast_1d(zeta).astype(np_float_type),
                                    name="zeta")
            var_list = [self.theta, self.zeta]

        self.build_deep_variables()
        var_list += self.deep_var_list

        if n_sgd_batch:
            print('Batch size for stochastic gradient descent: %d'
                  % n_sgd_batch)
        else:
            # n_sgd_batch = x.shape[0]
            print('Using full dataset for gradient descent')
        # complexity_ratio = np.sqrt(n_all / n_sgd_batch)

        # Setup the training graph
        self._setup_train_graph()

        # Setup the query graph
        self._setup_query_graph()

        # Setup the lagrangian objective
        # self.lagrangian = tf.log(complexity_ratio * self.complexity ** 2)
        self.lagrangian = self.complexity
        self.grad = tf.gradients(self.lagrangian, var_list)
        # self.grad_norm = tf.reduce_max(tf.abs(tf.concat(self.grad, axis=0)))
        # self.stop_criterion = tf.greater_equal(self.grad_norm, grad_tol)

        # Setup the training optimisation program
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = opt.minimize(self.lagrangian, var_list=var_list)

        # # Run the optimisation
        # print('Starting Training')
        # if n_sgd_batch > 1:
        #     k_fold = _KFold(n_splits=n_sgd_batch)
        # feed_dict = self.feed_dict
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        # step = 0
        # _train_history = []
        # grad_norm_check = grad_tol + 1
        # while grad_norm_check > grad_tol:
        #     if n_sgd_batch > 1:
        #         grad_norm_list = []
        #         for train_indices, test_indices in k_fold.split(x):
        #             feed_dict = {self.x: x[test_indices],
        #                          self.y: y[test_indices]}
        #             theta = self.sess.run(self.theta)
        #             zeta = self.sess.run(self.zeta)
        #             complexity = self.sess.run(self.complexity,
        #                                        feed_dict=feed_dict)
        #             cel = self.sess.run(self.cross_entropy_loss,
        #                                 feed_dict=feed_dict)
        #             acc = self.sess.run(self.train_accuracy,
        #                                 feed_dict=feed_dict)
        #             grad_norm = self.sess.run(self.grad_norm,
        #                                       feed_dict=feed_dict)
        #             grad_norm_list.append(grad_norm)
        #             self.sess.run(train, feed_dict=feed_dict)
        #             print('Step %d' % step,
        #                   '|| Hyperparameters: ', theta, zeta,
        #                   '|| Complexity: ', complexity,
        #                   '|| Train Accuracy: ', acc,
        #                   '|| Cross Entropy Loss: ', cel,
        #                   '|| Gradient Norm: ', grad_norm)
        #             step += 1
        #             _train_history.append(
        #                 [step, complexity, acc, cel, grad_norm]
        #                 + list(np.append(theta, zeta)))
        #         grad_norm_check = np.max(grad_norm_list)
        #     else:
        #         theta = self.sess.run(self.theta)
        #         zeta = self.sess.run(self.zeta)
        #         complexity = self.sess.run(self.complexity,
        #                                    feed_dict=feed_dict)
        #         cel = self.sess.run(self.cross_entropy_loss,
        #                             feed_dict=feed_dict)
        #         acc = self.sess.run(self.train_accuracy,
        #                             feed_dict=feed_dict)
        #         grad_norm = self.sess.run(self.grad_norm, feed_dict=feed_dict)
        #         grad_norm_check = grad_norm
        #         self.sess.run(train, feed_dict=feed_dict)
        #         print('Step %d' % step,
        #               '|| Hyperparameters: ', theta, zeta,
        #               '|| Complexity: ', complexity,
        #               '|| Train Accuracy: ', acc,
        #               '|| Cross Entropy Loss: ', cel,
        #               '|| Gradient Norm: ', grad_norm)
        #         step += 1
        #         _train_history.append([step, complexity, acc, cel, grad_norm]
        #                               + list(np.append(theta, zeta)))

        # Run the optimisation
        print('Starting Training')
        feed_dict = self.feed_dict
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        step = 0
        _train_history = []
        grad_norm_check = grad_tol + 1
        # np.set_printoptions(3)
        if to_train:
            while grad_norm_check > grad_tol:

                if n_sgd_batch:
                    sgd_indices = \
                        np.arange(step * n_sgd_batch,
                                  (step + 1) * n_sgd_batch) % n_all \
                            if sequential_batch \
                            else np.random.choice(n_all, n_sgd_batch,
                                                  replace=False)
                    feed_dict = {self.x: x[sgd_indices],
                                 self.y: y[sgd_indices]}

                theta = self.sess.run(self.theta)
                zeta = self.sess.run(self.zeta)
                complexity = self.sess.run(self.complexity,
                                           feed_dict=feed_dict)
                cel = self.sess.run(self.cross_entropy_loss,
                                    feed_dict=feed_dict)
                cel_valid = self.sess.run(self.cross_entropy_loss_valid,
                                          feed_dict=feed_dict)
                acc = self.sess.run(self.train_accuracy, feed_dict=feed_dict)
                grad = self.sess.run(self.grad, feed_dict=feed_dict)
                grad_norm = compute_grad_norm(grad)
                grad_norm_check = grad_norm
                self.sess.run(train, feed_dict=feed_dict)
                print('Step %d' % step,
                      '|| Hyperparameters: ', theta, zeta,
                      '|| Complexity: ', complexity,
                      '|| Train Accuracy: ', acc,
                      '|| Cross Entropy Loss: ', cel,
                      '|| Gradient Norm: ', grad_norm,
                      np.array([np.max(np.abs(grad_i)) for grad_i in grad]))
                step += 1
                _train_history.append([step, complexity, acc, cel, cel_valid,
                                       grad_norm]
                                      + list(np.append(theta, zeta)))

        # Store train history
        _train_history = np.array(_train_history)
        self.train_history = {'iterations': _train_history[:, 0],
                              'complexity': _train_history[:, 1],
                              'accuracy': _train_history[:, 2],
                              'cross_entropy_loss': _train_history[:, 3],
                              'valid_cross_entropy_loss': _train_history[:, 4],
                              'gradient_norm': _train_history[:, 5],
                              'kernel_hypers': _train_history[:, 6:-1],
                              'regularisation': _train_history[:, -1]} \
            if to_train else None

        # Store the optimal hyperparameters
        self.theta_train = self.sess.run(self.theta)
        self.zeta_train = self.sess.run(self.zeta)
        self.steps_train = step
        self.complexity_train = self.sess.run(self.complexity,
                                              feed_dict=self.feed_dict)
        self.acc_train = self.sess.run(self.train_accuracy,
                                       feed_dict=self.feed_dict)
        self.cel_train = self.sess.run(self.cross_entropy_loss,
                                       feed_dict=self.feed_dict)
        self.grad_norm_train = compute_grad_norm(self.sess.run(self.grad,
                                                 feed_dict=self.feed_dict))
        self.msp_train = self.sess.run(self.msp, feed_dict=self.feed_dict)
        print('Training Finished')
        print('Learned Hyperparameters: ', self.theta_train, self.zeta_train)
        print('Learned Complexity: ', self.complexity_train)
        print('Achieved Training Accuracy: ', self.acc_train)
        print('Achieved Cross Entropy Loss: ', self.cel_train)
        print('Achieved Mean Sum of Probabilities: ', self.msp_train)
        print(self.sess.run(var_list))
        return self

    def _setup_train_graph(self):
        """Setup the training computational graph."""
        # The regulariser matrix
        i = tf.cast(tf.eye(self.n), tf_float_type)
        reg = tf.cast(self.n, tf_float_type) * self.zeta * i

        # The gram matrix and regularised version thereof for the output space
        self.l = _kronecker_delta(self.y, self.y)
        self.l_reg = self.l + reg
        self.chol_l_reg = tf.cholesky(self.l_reg)

        # The gram matrix and regularised version thereof for the input space
        self.k = self.kernel(self.x, self.x, self.theta)
        self.k_reg = self.k + reg
        self.chol_k_reg = tf.cholesky(self.k_reg)

        # The embedding weights on the training data
        self.w = tf.cholesky_solve(self.chol_k_reg, self.k)

        # The decision probabilities on the training data
        self.p_pred = _expectance(tf.cast(tf.equal(self.y, self.classes),
                                          tf_float_type), self.w)

        # The predictions on the training data
        self.y_pred = _classify(self.p_pred, classes=self.classes)

        # The training accuracy
        self.train_accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.y_pred, tf.reshape(self.y, [-1])), tf_float_type))

        # The prediction probabilities on the actual label
        indices = tf.cast(tf.range(self.n), tf_int_type)
        self.p_want = tf.gather_nd(self.p_pred,
                                   tf.stack([indices, self.y_indices], axis=1))

        # The cross entropy loss over the training data
        self.cross_entropy_loss = - tf.reduce_mean(tf.log(self.p_want))

        # The clip-normalised valid decision probabilities
        self.p_pred_valid = \
            tf.transpose(_clip_normalize(tf.transpose(self.p_pred)))

        # The clip-normalised valid decision probabilities on the actual label
        self.p_want_valid = \
            tf.gather_nd(self.p_pred_valid, tf.stack([indices, self.y_indices],
                                                     axis=1))

        # The valid cross entropy loss over the training data
        self.cross_entropy_loss_valid = \
            - tf.reduce_mean(tf.log(self.p_want_valid))

        # The model complexity of the classifier
        self.complexity = self._define_complexity()

        # Other interesting quantities
        self.pred_constraint = self.p_want - 1 / self.n_classes
        self.prob_constraint = tf.reduce_sum(self.w, axis=0) - 1
        self.msp = tf.reduce_mean(self.prob_constraint + 1)

    def _setup_query_graph(self):
        """Setup the query inference computational graph."""
        # Query on x
        self.x_query = tf.placeholder(tf_float_type, shape=[None, self.d])

        # Inference Gram Matrix
        self.k_query = self.kernel(self.x, self.x_query, self.theta)

        # Conditional Embedding Weights
        self.w_query = tf.cholesky_solve(self.chol_k_reg, self.k_query)

        # Decision Probability
        self.p_query = _expectance(tf.cast(tf.equal(self.y, self.classes),
                                           tf_float_type), self.w_query)
        self.p_query_valid = \
            tf.transpose(_clip_normalize(tf.transpose(self.p_query)))

        # Prediction
        self.y_query = _classify(self.p_query, classes=self.classes)

        # Information Entropy
        ind = tf.reshape(self.y_indices, [-1])
        self.p_field = tf.transpose(tf.gather(tf.transpose(self.p_query), ind))
        self.p_field_adjust = _adjust_prob(self.p_field)  # (n_query, n_train)
        self.u_field = -tf.log(self.p_field_adjust)
        # TODO: Einstein Sum for diagonal of matrix product
        # self.h_query = tf.einsum('ij,ji->i', u, self.w_query)
        self.h_query = tf.diag_part(tf.matmul(self.u_field, self.w_query))
        self.h_query_valid = tf.clip_by_value(self.h_query, 0, np.inf)

        # Query on y
        self.y_query = tf.placeholder(tf_float_type, shape=[None, 1])

        # Reverse Inference Gram Matrix
        self.l_query = _kronecker_delta(self.y, self.y_query)

        # Reverse Conditional Embedding Weights
        self.v_query = tf.cholesky_solve(self.chol_l_reg, self.l_query)

        # Reverse Embedding
        self.mu_xy = tf.matmul(self.kernel(self.x_query, self.x, self.theta),
                               self.v_query)

        # Input Expectance
        self.x_exp = _expectance(self.x, self.v_query)

        # Input Variance
        self.x_var = _variance(self.x, self.v_query)

    def _define_complexity(self, complexity='Global Rademacher Complexity'):
        """
        Define the kernel embedding classifier model complexity.

        Returns
        -------
        float
            The model complexity
        """
        if complexity == 'Embedding Trace':
            return tf.trace(self.w)
        elif complexity == 'Global Rademacher Complexity':
            b = _kronecker_delta(self.y, self.classes[:, tf.newaxis])
            v = tf.cholesky_solve(self.chol_k_reg, b)
            wtw = tf.matmul(tf.transpose(v), tf.matmul(self.k, v))
            return tf.sqrt(tf.trace(wtw))
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
        return self.sess.run(self.p_query_valid if normalize else self.p_query,
                             feed_dict=self.feed_dict)

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
            return self.sess.run(self.h_query_valid if clip else self.h_query,
                                 feed_dict=self.feed_dict)
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