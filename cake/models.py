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
float_type = tf.float32


class Classifier():

    def __init__(self, kernel=_s_gaussian):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        self.kernel = kernel

    def fit(self, x, y,
            theta=np.array([1., 1.]), zeta=0.01,
            learning_rate=0.01, grad_tol=0.01, log_hypers=True):
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

        Returns
        -------
        bake.Classifier
            The trained classifier
        """
        # Setup the data
        # self.n = x.shape[0]
        # self.d = x.shape[1]
        classes = np.unique(y)
        class_indices = np.arange(classes.shape[0])
        # y_one_hot = y == classes
        # y_indices = np.where(y_one_hot)[1]
        self.classes = tf.cast(tf.constant(classes), float_type)
        self.class_indices = tf.cast(tf.constant(class_indices), tf.int32)
        self.n_classes = classes.shape[0]
        # self.y_one_hot = tf.cast(tf.constant(y_one_hot), float_type)
        # self.y_indices = tf.cast(tf.constant(y_indices), tf.int32)


        self.x = tf.placeholder(float_type, shape=[None, x.shape[1]])
        self.y = tf.placeholder(float_type, shape=[None, y.shape[1]])
        self.feed_dict = {self.x: x, self.y: y}

        self.n = tf.shape(self.x)[0]
        self.d = x.shape[1]
        # self.classes = tf.cast(tf.unique(self.y), float_type)
        # self.n_classes = tf.shape(self.classes)[0]
        # self.class_indices = tf.cast(tf.range(self.n_classes), tf.int32)
        self.y_one_hot = tf.equal(self.y, self.classes)
        self.y_indices = \
            tf.cast(tf.reduce_sum(tf.where(self.y_one_hot,
                             tf.ones(tf.shape(self.y_one_hot)) * class_indices,
                             tf.zeros(tf.shape(self.y_one_hot))), axis=1), tf.int32)

        # Setup the optimisation parameters
        if log_hypers:
            self.log_theta = tf.Variable(np.log(theta).astype(np.float32),
                                         name="log_theta")
            self.log_zeta = tf.Variable(
                np.log(np.atleast_1d(zeta)).astype(np.float32),
                name="log_zeta")
            self.theta = tf.exp(self.log_theta, name="theta")
            self.zeta = tf.exp(self.log_zeta, name="zeta")
            var_list = [self.log_theta, self.log_zeta]
        else:
            self.theta = tf.Variable(theta.astype(np.float32), name="theta")
            self.zeta = tf.Variable(np.atleast_1d(zeta).astype(np.float32),
                                    name="zeta")
            var_list = [self.theta, self.zeta]
        self.alpha = tf.Variable(np.zeros(x.shape[0]).astype(np.float32))
        self.beta = tf.Variable(np.zeros(x.shape[0]).astype(np.float32))
        self.gamma = tf.Variable(np.float32(0.))

        # Setup the training graph
        i = tf.cast(tf.eye(self.n), float_type)
        reg = tf.cast(self.n, float_type) * self.zeta * i
        self.l = _kronecker_delta(self.y, self.y)
        self.l_reg = self.l + reg
        self.chol_l_reg = tf.cholesky(self.l_reg)
        self.k = self.kernel(self.x, self.x, self.theta)
        self.k_reg = self.k + reg
        self.chol_k_reg = tf.cholesky(self.k_reg)
        self.w = tf.cholesky_solve(self.chol_k_reg, self.k)
        self.p_pred = _expectance(tf.cast(tf.equal(self.y, self.classes),
                                          float_type), self.w)
        self.y_pred = _classify(self.p_pred, classes=self.classes)
        self.train_accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.y_pred, tf.reshape(self.y, [-1])), float_type))
        self.complexity = self.compute_complexity()
        indices = tf.cast(tf.range(self.n), tf.int32)
        self.p_want = tf.gather_nd(self.p_pred,
                              tf.stack([indices, self.y_indices], axis=1))
        self.cross_entropy_loss = -tf.reduce_sum(tf.log(self.p_want)) / tf.cast(self.n, float_type)
        self.pred_constraint = self.p_want - 1 / self.n_classes
        self.prob_constraint = tf.reduce_sum(self.w, axis=0) - 1
        self.mean_sum_probability = \
            tf.reduce_mean(self.prob_constraint + 1)

        self.cross_entropy_loss = - tf.cast(self.y_one_hot, float_type) \
                                  * tf.log(self.p_pred) / tf.cast(self.n, float_type)

        # Setup the lagrangian objective
        hypers_list = [self.theta, self.zeta]
        self.lagrangian = self.complexity \
                          # - tf.reduce_sum(self.alpha * (self.p_want - 1))
                          # - tf.reduce_sum(self.beta * self.prob_constraint)
        self.l_grad = tf.gradients(self.lagrangian, var_list)
        self.l_grad_norm = \
            tf.reduce_max(tf.abs(tf.concat(self.l_grad, axis=0)))
        self.stop_criterion = tf.greater_equal(self.l_grad_norm, grad_tol)

        # Setup the query graph
        self.setup_query_graph()

        # Setup the training optimisation program
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = opt.minimize(self.lagrangian, var_list=var_list)

        # k_fold = _KFold(n_splits=10)
        feed_dict = self.feed_dict
        # Run the optimisation
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        step = 1
        while self.sess.run(self.stop_criterion, feed_dict=feed_dict):
            norm = self.sess.run(self.l_grad_norm, feed_dict=feed_dict)
            hypers = self.sess.run(hypers_list, feed_dict=feed_dict)
            acc = self.sess.run(self.train_accuracy, feed_dict=feed_dict)
            complexity = self.sess.run(self.complexity, feed_dict=feed_dict)

            self.sess.run(train, feed_dict=feed_dict)
            # for train_indices, test_indices in k_fold.split(x):
            #     feed_dict = {self.x: x[test_indices], self.y: y[test_indices]}
            #     self.sess.run(train, feed_dict=feed_dict)
            print('Step %d:' % step,
                  '|| Gradient Norm: ', norm,
                  '|| Complexity: ', complexity,
                  '|| Hyperparameters: ', *tuple(hypers),
                  '|| Train Accuracy: ', acc)
            step += 1

        # Store the optimal hyperparameters
        self.theta_opt = self.sess.run(self.theta)
        self.zeta_opt = self.sess.run(self.zeta)

        # Print out the results
        print('Hyperparameters: ', self.theta_opt, self.zeta_opt)
        final_gradients = self.sess.run(self.l_grad, feed_dict=self.feed_dict)
        print('Final Gradients: ', final_gradients)
        complexity_opt = self.sess.run(self.complexity,
                                       feed_dict=self.feed_dict)
        print('Complexity: ', complexity_opt)
        # print(self.sess.run(self.p_pred, feed_dict=self.feed_dict))
        train_accuracy_opt = \
            self.sess.run(self.train_accuracy, feed_dict=self.feed_dict)
        print('Training Accuracy: ', train_accuracy_opt)
        mean_sum_probability_opt = \
            self.sess.run(self.mean_sum_probability, feed_dict=self.feed_dict)
        print('Mean Sum Probability: ', mean_sum_probability_opt)
        return self

    def setup_query_graph(self):

        # Query on x
        self.x_query = tf.placeholder(float_type, shape=[None, self.d])

        # Inference Gram Matrix
        self.k_query = self.kernel(self.x, self.x_query, self.theta)

        # Conditional Embedding Weights
        self.w_query = tf.cholesky_solve(self.chol_k_reg, self.k_query)

        # Decision Probability
        self.p_query = _expectance(tf.cast(tf.equal(self.y, self.classes),
                                           float_type), self.w_query)
        self.p_query_valid = \
            tf.transpose(_clip_normalize(tf.transpose(self.p_query)))

        # Prediction
        self.y_query = _classify(self.p_query, classes=self.classes)

        # Information Entropy
        ind = tf.reshape(self.y_indices, [-1])
        self.p_field = tf.transpose(tf.gather(tf.transpose(self.p_query), ind))
        self.p_field_adjust = _adjust_prob(self.p_field)  # (n_query, n_train)
        u = -tf.log(self.p_field_adjust)
        # TODO: Einstein Sum for diagonal of matrix product
        # self.h_query = tf.einsum('ij,ji->i', u, self.w_query)
        self.h_query = tf.diag_part(tf.matmul(u, self.w_query))
        self.h_query_valid = tf.clip_by_value(self.h_query, 0, np.inf)

        # Query on y
        self.y_query = tf.placeholder(float_type, shape=[None, 1])

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

        # Input Mode
        ind = tf.cast(tf.argmax(self.mu_xy, axis=0), tf.int32)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def compute_complexity(self, complexity='Global Rademacher Complexity'):
        """
        Compute the model complexity of the current classifier.

        Returns
        -------
        float
            The log of the model complexity
        """
        if complexity == 'Embedding Trace':
            complexity = tf.trace(self.w)
            return complexity
        elif complexity == 'Global Rademacher Complexity':
            b = _kronecker_delta(self.y, self.classes[:, tf.newaxis])
            step_1 = tf.cholesky_solve(self.chol_k_reg, b)
            step_2 = tf.matmul(self.k, step_1)
            step_3 = tf.cholesky_solve(self.chol_k_reg, step_2)
            wtw = tf.matmul(tf.transpose(b), step_3)
            complexity = tf.sqrt(tf.trace(wtw))
            return complexity
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

    def predict_entropy(self, x_query, clip=True):
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
        return self.sess.run(self.h_query_valid if clip else self.h_query,
                             feed_dict=self.feed_dict)

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

    def input_mode_optimized(self, x_inits=None):
        """
        Predict the mode of the condition input using posterior mode decoding.

        Returns
        -------
        numpy.ndarray
            The mode of each feature for each class (n_class, d)
        """
        y_query = self.classes[:, np.newaxis]
        self.x_modes = np.zeros((self.n_classes, self.d))
        self.mu_modes = np.zeros(self.n_classes)
        x_inits = self.input_expectance() if x_inits is None else x_inits

        for i, y in enumerate(y_query):
            v_query = self.reverse_weights(y[:, np.newaxis])

            def embedding(x):
                x_2d = np.array([x])
                return np.dot(self.kernel(x_2d, self.x, self.theta), v_query)

            def objective(x):
                f = -embedding(x)[0][0]
                # print('\tObjective: %f' % -f)
                return f

            x_init = x_inits[i]
            print('Starting Mode Decoding for Class %d' % y)
            optimal_result = _minimize(objective, x_init)
            self.x_modes[i, :] = optimal_result.x
            self.mu_modes[i] = -optimal_result.fun
            print('Embedding Value at Mode: %f' % self.mu_modes[i])
        return self.x_modes

    def input_mode_enumerated(self, x_candidates):
        """
        Predict the mode of the conditioned input by picking from candidates.

        Parameters
        ----------
        x_candidates : numpy.ndarray
            The set of candidate points of size (n_cand, d)

        Returns
        -------
        numpy.ndarray
            The mode of each feature for each class (n_class, d)
        """
        y_query = self.classes[:, np.newaxis]
        reverse_embedding = self.reverse_embedding(y_query, x_candidates)
        ind = np.argmax(reverse_embedding, axis=0)
        return self.input_mode_optimized(x_inits=x_candidates[ind])

    def input_mode(self, x_candidates=None):
        """
        Predict the mode of the conditioned input.

        Parameters
        ----------
        x_candidates : numpy.ndarray, optional
            The set of candidate points of size (n_cand, d)

        Returns
        -------
        numpy.ndarray
            The mode of each feature for each class (n_class, d)
        """
        return self.input_mode_optimized() if x_candidates is None \
            else self.input_mode_enumerated(x_candidates)
