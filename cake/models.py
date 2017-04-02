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

float_type = tf.float32
_t = tf.transpose

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
            theta=np.array([10., 0.1]), zeta=1e-2,
            learning_rate=0.1):
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
        zeta : float
            The initial regularisation parameter

        Returns
        -------
        bake.Classifier
            The trained classifier
        """
        self.x = tf.placeholder(float_type, shape=list(x.shape))
        self.y = tf.placeholder(float_type, shape=list(y.shape))
        self.n = tf.shape(x)[0]
        self.d = tf.shape(x)[0]
        classes = np.unique(y)
        class_indices = np.arange(classes.shape[0])
        self.classes = tf.cast(tf.constant(classes), float_type)
        self.class_indices = tf.cast(tf.constant(class_indices), tf.int32)
        self.y_one_hot = tf.equal(self.y, self.classes)
        self.y_indices = tf.where(self.y_one_hot)[1]
        self.n_classes = tf.shape(self.classes)
        self.log_theta = tf.Variable(np.log(theta).astype(np.float32),
                                     name="Log_Kernel_Hyperparameters")
        self.log_zeta = tf.Variable(np.float32(np.log(zeta)),
                                    name="Log_Regularisation_Parameter")
        self.theta = tf.exp(self.log_theta, name="Kernel_Hyperparameters")
        self.zeta = tf.exp(self.log_zeta, name="Regularisation_Parameter")
        self.alpha = tf.Variable(1.0, name="Train_Accuracy_Multiplier")
        self.beta = tf.Variable(1.0, name="Mean_Sum_Probablity_Multiplier")
        self.update(self.theta, self.zeta)

        self.lagrangian = self.complexity - \
                          tf.multiply(self.alpha, self.train_accuracy - 1) - \
                          tf.multiply(self.beta, self.mean_sum_probability - 1)
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optim = opt.minimize(self.lagrangian,
                             var_list=[self.log_theta,
                                       self.log_zeta,
                                       self.alpha,
                                       self.beta])

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run(optim, feed_dict={self.x: x, self.y: y})

        self.theta_opt = sess.run(self.theta)
        self.zeta_opt = sess.run(self.zeta)
        print('Hyperparameters: ', self.theta_opt, self.zeta_opt)
        alpha_opt = sess.run(self.alpha)
        beta_opt = sess.run(self.beta)
        print('Lagrange Multipliers: ', alpha_opt, beta_opt)

        complexity_opt = sess.run(self.complexity,
                                  feed_dict={self.x: x, self.y: y})
        print('Complexity: ', complexity_opt)
        train_accuracy_opt = sess.run(self.train_accuracy,
                                      feed_dict={self.x: x, self.y: y})
        print('Training Accuracy: ', train_accuracy_opt)
        mean_sum_probability_opt = sess.run(self.mean_sum_probability,
                                            feed_dict={self.x: x, self.y: y})
        print('MSP: ', mean_sum_probability_opt)
        return self

    def compute_complexity(self):
        """
        Compute the model complexity of the current classifier.

        Returns
        -------
        float
            The log of the model complexity
        """
        ## FIRST METHOD
        complexity = tf.trace(self.w)
        return tf.log(complexity)

        ## THIRD METHOD (GLOBAL RADEMACHER COMPLEXITY)
        # b = _kronecker_delta(self.y, self.classes[:, np.newaxis])
        # a = tf.matmul(self.k, tf.cholesky_solve(self.chol_k_reg, b))
        # wtw = tf.matmul(_t(b), tf.cholesky_solve(self.chol_k_reg, a))
        # complexity = tf.sqrt(tf.trace(wtw))
        # return tf.log(complexity)

    def update(self, theta, zeta):
        """
        Update the hyperparameters of the classifier.

        Parameters
        ----------
        theta : tensorflow.Variable
            The hyperparameters of the kernel
        zeta : tensorflow.Variable
            The regularisation parameter of the conditional embedding

        Returns
        -------
        bake.Classifier
            The updated classifier
        """
        i = tf.cast(tf.eye(self.n), float_type)
        reg = tf.multiply(tf.cast(self.n, float_type),
                          tf.multiply(self.zeta, i))
        self.l = _kronecker_delta(self.y, self.y)
        self.l_reg = self.l + reg
        self.theta = theta
        self.zeta = zeta
        self.k = self.kernel(self.x, self.x, self.theta)
        self.k_reg = self.k + reg
        self.chol_k_reg = tf.cholesky(self.k_reg)
        self.w = self.predict_weights(self.x)
        self.mean_sum_probability = tf.reduce_mean(tf.reduce_sum(self.w,
                                                                 axis=0))
        self.p_pred = self.predict_proba(self.x, normalize=False)
        self.y_pred = _classify(self.p_pred, classes=self.classes)
        self.train_accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.y_pred, tf.reshape(self.y, [-1])), float_type))
        self.complexity = self.compute_complexity()
        return self

    def predict_weights(self, x_query):
        """
        Predict the query weights for classification.

        Parameters
        ----------
        x_query : tensorflow.Tensor
            The query points of size (n_query, d)

        Returns
        -------
        tensorflow.Tensor
            The query weights of size (n, n_query)
        """
        k_query = self.kernel(self.x, x_query, self.theta)
        w_query = tf.cholesky_solve(self.chol_k_reg, k_query)
        return w_query

    def predict_proba(self, x_query, normalize=True):
        """
        Predict the probabilities for classification.

        Parameters
        ----------
        x_query : tensorflow.Tensor
            The query points of size (n_query, d)
        normalize : bool
            Normalize the probabilities properly

        Returns
        -------
        tensorflow.Tensor
            The query probability estimates of size (n_query, n_class)
        """
        w_query = self.predict_weights(x_query)
        p_query = _expectance(tf.cast(tf.equal(self.y, self.classes),
                                      float_type), w_query)
        return _t(_clip_normalize(_t(p_query))) if normalize else p_query

    def predict(self, x_query):
        """
        Predict the targets for classification.

        Parameters
        ----------
        x_query : tensorflow.Tensor
            The query points of size (n_query, d)

        Returns
        -------
        tensorflow.Tensor
            The query targets of size (n_query,)
        """
        p_query = self.predict_proba(x_query, normalize=False)
        y_query = _classify(p_query, classes=self.classes)
        return y_query

    def predict_entropy(self, x_query, clip=True):
        """
        Predict the entropy of the predictions.

        Parameters
        ----------
        x_query : tensorflow.Tensor
            The query points of size (n_query, d)
        clip : bool
            Make sure the entropy estimate is non-negative

        Returns
        -------
        tensorflow.Tensor
            The query entropy of size (n_query,)
        """
        w_query = self.predict_weights(x_query)
        p_query = _expectance(tf.cast(tf.equal(self.y, self.classes),
                                      float_type), w_query)
        p_field = p_query[:, tf.reshape(self.y_indices, [-1])]
        p_field_adjust = _adjust_prob(p_field)  # (n_query, n_train)
        h_query = tf.einsum('ij,ji->i', -tf.log(p_field_adjust), w_query)
        return tf.clip_by_value(h_query, 0, np.inf) if clip else h_query

    # def reverse_weights(self, y_query):
    #     """
    #     Compute the weights for the reverse embedding.
    #
    #     Parameters
    #     ----------
    #     y_query : tensorflow.Tensor
    #         The query outputs to condition the reverse embedding on
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The weights for the reverse embedding (n, n_query)
    #     """
    #     l_query = _kronecker_delta(self.y, y_query)
    #     v_query = _solve_posdef(self.l_reg, l_query)[0]
    #     return v_query
    #
    # def reverse_embedding(self, y_query, x_query):
    #     """
    #     Evaluate the reverse embedding.
    #
    #     Parameters
    #     ----------
    #     y_query : tensorflow.Tensor
    #         The query outputs to condition the reverse embedding on
    #     x_query : tensorflow.Tensor
    #         The query inputs to evaluate the reverse embedding on
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The evaluated reverse embedding of size (n_x_query, n_y_query)
    #     """
    #     v_query = self.reverse_weights(y_query)
    #     return np.dot(self.kernel(x_query, self.x, self.theta), v_query)
    #
    # def input_expectance(self):
    #     """
    #     Predict the expectance of the input conditioned on a class.
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The expectance of each feature for each class (n_class, d)
    #     """
    #     y_query = self.classes[:, np.newaxis]
    #     v_query = self.reverse_weights(y_query)
    #     return _expectance(self.x, v_query)
    #
    # def input_variance(self):
    #     """
    #     Predict the variance of the input conditioned on a class.
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The variance of each feature for each class (n_class, d)
    #     """
    #     y_query = self.classes[:, np.newaxis]
    #     v_query = self.reverse_weights(y_query)
    #     return _variance(self.x, v_query)
    #
    # def input_mode_optimized(self, x_inits=None):
    #     """
    #     Predict the mode of the condition input using posterior mode decoding.
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The mode of each feature for each class (n_class, d)
    #     """
    #     y_query = self.classes[:, np.newaxis]
    #     self.x_modes = np.zeros((self.n_classes, self.d))
    #     self.mu_modes = np.zeros(self.n_classes)
    #     x_inits = self.input_expectance() if x_inits is None else x_inits
    #
    #     for i, y in enumerate(y_query):
    #         v_query = self.reverse_weights(y[:, np.newaxis])
    #
    #         def embedding(x):
    #             x_2d = np.array([x])
    #             return np.dot(self.kernel(x_2d, self.x, self.theta), v_query)
    #
    #         def objective(x):
    #             f = -embedding(x)[0][0]
    #             # print('\tObjective: %f' % -f)
    #             return f
    #
    #         x_init = x_inits[i]
    #         # x_class = self.x[self.y.ravel() == i]
    #         # x_min = np.min(x_class, axis=0)
    #         # x_max = np.max(x_class, axis=0)
    #         # bounds = [(x_min[i], x_max[i]) for i in range(len(x_init))]
    #         print('Starting Mode Decoding for Class %d' % y)
    #         optimal_result = _minimize(objective, x_init)
    #         self.x_modes[i, :] = optimal_result.x
    #         self.mu_modes[i] = -optimal_result.fun
    #         print('Embedding Value at Mode: %f' % self.mu_modes[i])
    #     return self.x_modes
    #
    # def input_mode_enumerated(self, x_candidates):
    #     """
    #     Predict the mode of the conditioned input by picking from candidates.
    #
    #     Parameters
    #     ----------
    #     x_candidates : tensorflow.Tensor
    #         The set of candidate points of size (n_cand, d)
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The mode of each feature for each class (n_class, d)
    #     """
    #     y_query = self.classes[:, np.newaxis]
    #     reverse_embedding = self.reverse_embedding(y_query, x_candidates)
    #     ind = np.argmax(reverse_embedding, axis=0)
    #     return self.input_mode_optimized(x_inits=x_candidates[ind])
    #
    # def input_mode(self, x_candidates=None):
    #     """
    #     Predict the mode of the conditioned input.
    #
    #     Parameters
    #     ----------
    #     x_candidates : tensorflow.Tensor, optional
    #         The set of candidate points of size (n_cand, d)
    #
    #     Returns
    #     -------
    #     tensorflow.Tensor
    #         The mode of each feature for each class (n_class, d)
    #     """
    #     return self.input_mode_optimized() if x_candidates is None \
    #         else self.input_mode_enumerated(x_candidates)
