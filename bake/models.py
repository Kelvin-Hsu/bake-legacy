"""
Models Module.
"""
import numpy as np
from .infer import expectance as _expectance
from .infer import variance as _variance
from .infer import clip_normalize as _clip_normalize
from .infer import classify as _classify
from .kernels import s_gaussian as _s_gaussian
from .kernels import kronecker_delta as _kronecker_delta
from .kernels import general_kronecker_delta as _general_delta
from .linalg import solve_posdef as _solve_posdef
from scipy.optimize import minimize as _minimize
from scipy.linalg import svd as _svd
import datetime


class Classifier():

    def __init__(self, kernel=_s_gaussian, output_kernel=_general_delta):
        """
        Initialize the classifier.

        Parameters
        ----------
        kernel : callable, optional
            A kernel function
        """
        self.kernel = kernel
        self.output_kernel = output_kernel
        self.psi = 0

    def fit(self, x, y,
            h=None,
            h_min=np.array([1e-8, 1e-8, 1e-8]),
            h_max=np.array([np.inf, np.inf, np.inf]),
            h_init=np.array([1.0, 1.0, 1e-6]),
            verbose=True,
            save_history=True,
            directory=None):
        """
        Fit the kernel embedding classifier.

        Parameters
        ----------
        x : numpy.ndarray
            The training inputs of size (n, d)
        y : numpy.ndarray
            The training outputs of size (n, 1)
        h : numpy.ndarray, optional
            The hyperparameters to be set (training will be skipped)
        h_min : numpy.ndarray, optional
            The lower bound of the hyperparameters
        h_max : numpy.ndarray, optional
            The upper bound of the hyperparameters
        h_init : numpy.ndarray, optional
            The initial values of the hyperparameters
        verbose : bool, optional
            Display the training process

        Returns
        -------
        bake.Classifier
            The trained classifier
        """
        self.x = x.copy()
        self.y = y.copy()
        self.n, self.d = self.x.shape
        self.classes = np.unique(self.y)
        self.class_indices = np.arange(self.classes.shape[0])
        self.y_one_hot = self.y == self.classes
        self.y_indices = np.where(self.y_one_hot)[1]
        self.n_classes = self.classes.shape[0]
        self.theta = 0 * h_init[:-1]
        self.zeta = 0 * h_init[-1]
        self.update(self.theta, self.zeta)

        self._f_train = []
        self._a_train = []
        self._p_train = []
        self._h_train = []

        self.__minute_output__ = True if directory is not None else False

        if self.__minute_output__:
            self.__minute__ = -1
            verbose = True

        def constraint_pred(hypers):
            self.update(hypers[:-1], hypers[-1], training=True)
            return self.train_accuracy - 1  # log instead of -1

        def constraint_prob(hypers):
            self.update(hypers[:-1], hypers[-1], training=True)
            return self.mean_sum_probability - 1  # log instead of -1

        def objective(hypers):
            self.update(hypers[:-1], hypers[-1], training=True)
            if verbose:
                s = 'Training Accuracy: %f || ' \
                    'Mean Sum of Probabilities: %f || ' \
                    'Complexity: %f' \
                    % (self.train_accuracy,
                       self.mean_sum_probability,
                       self.complexity)
                string = 'Hyperparameters: %s %s' % (np.array_str(hypers), s)
                print(string)
            if save_history:
                self._f_train.append(self.complexity)
                self._a_train.append(self.train_accuracy)
                self._p_train.append(self.mean_sum_probability)
                self._h_train.append(hypers)
                if self.__minute_output__:
                    now = datetime.datetime.now()
                    if now.minute != self.__minute__:
                        self.__minute__ = now.minute
                        now_string = '%s_%s_%s_%s_%s_%s' % (
                            now.year, now.month, now.day,
                            now.hour, now.minute, now.second)
                        filename = '%s%s_minute_output.txt' % \
                                   (directory, now_string)
                        f = open(filename, 'w')
                        f.write('Time: %s\n' % now_string)
                        f.write('Iterations: %d\n' % len(self._f_train))
                        f.write('%s\n' % string)
                        f.close()
                        print('Minute output saved in "%s"' % filename)

            return self.complexity

        def get_ith_constraint(i):
            def ith_constraint(hypers):
                self.update(hypers[:-1], hypers[-1], training=True)
                return self.constraint_vector[i]
            return ith_constraint

        if h is None:

            bounds = [(h_min[i], h_max[i]) for i in range(len(h_init))]
            c_1 = {'type': 'ineq', 'fun': constraint_pred}
            c_2 = {'type': 'ineq', 'fun': constraint_prob}
            constraints = (c_1, c_2)
            constraints = tuple([{'type': 'ineq', 'fun': get_ith_constraint(i)}
                                 for i in range(self.n)])
            options = {'maxiter': 5000,
                       'disp': True}
            optimal_result = _minimize(objective, h_init,
                                       bounds=bounds,
                                       constraints=constraints,
                                       options=options)
            h = optimal_result.x
            if verbose:
                print('Training Completed')
            if save_history:
                self._f_train = np.array(self._f_train)
                self._a_train = np.array(self._a_train)
                self._p_train = np.array(self._p_train)
                self._h_train = np.array(self._h_train)
                self._optimal_result = optimal_result

        if verbose:
            s = 'Training Accuracy: %f || ' \
                'Mean Sum of Probabilities: %f || ' \
                'Complexity: %f' \
                % (self.train_accuracy,
                   self.mean_sum_probability,
                   self.complexity)
            print('Hyperparameters: ', np.append(self.theta, self.zeta), s)
        self.update(h[:-1], h[-1])

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
        # complexity = np.trace(self.w)
        # return np.log(complexity)

        ## SECOND METHOD
        # identity = np.eye(self.n)
        # k_reg_inv = _solve_posdef(self.k_reg, identity)[0]
        # # k_reg_inv = _pinv(self.k_reg)
        # # print(np.linalg.det(self.k_reg), np.linalg.det(k_reg_inv), np.linalg.det(self.w))
        # w = np.dot(k_reg_inv, self.k)
        # a = np.dot(w, k_reg_inv)
        # # print(np.linalg.det(a))
        # b = _kronecker_delta(self.y, self.classes[:, np.newaxis], self.psi)
        # complexity_terms = np.array([np.dot(b[:, c], np.dot(a, b[:, c]))
        #                             for c in self.class_indices])
        # print('Complexity Terms: ', complexity_terms)
        # return np.log(np.sum(complexity_terms))

        ## THIRD METHOD (GLOBAL RADEMACHER COMPLEXITY)
        b = _kronecker_delta(self.y, self.classes[:, np.newaxis])
        a = np.dot(self.k, _solve_posdef(self.k_reg, b)[0])
        wtw = np.dot(b.T, _solve_posdef(self.k_reg, a)[0])
        complexity = np.trace(wtw)
        return np.log(complexity)

        ## FOURTH METHOD (GLOBAL RADEMACHER COMPLEXITY NOT LEGIT: NOT GOOD)
        # b = _kronecker_delta(self.y, self.classes[:, np.newaxis])
        # a = _solve_posdef(self.k_reg, b)[0]
        # wtw = np.dot(b.T, _solve_posdef(self.k_reg, a)[0])
        # complexity = np.trace(wtw)
        # return np.log(complexity)

        ## FIFTH METHOD (LOCAL RADEMACHER COMPLEXITY NOT LEGIT)
        # b = _kronecker_delta(self.y, self.classes[:, np.newaxis])
        # a = _solve_posdef(self.k_reg, b)[0]
        # u, s, v = _svd(a)
        # # t = np.floor(self.n_classes/2)
        # t = 0
        # print(s[t:])
        # complexity = np.sum(s[t:])
        # return np.log(complexity)


    def update(self, theta, zeta, training=False):
        """
        Update the hyperparameters of the classifier.

        Parameters
        ----------
        theta : numpy.ndarray
            The hyperparameters of the kernel
        zeta : float
            The regularisation parameter of the conditional embedding

        Returns
        -------
        bake.Classifier
            The updated classifier
        """
        if not training:
            self.l = self.output_kernel(self.y, self.y, self.psi)
            self.l_reg = self.l + self.n * (self.zeta ** 2) * np.eye(self.n)
        if np.allclose(theta, self.theta, 1e-300) \
                and np.allclose(zeta, self.zeta, 1e-300):
            return self
        self.theta = theta
        self.zeta = zeta
        self.k = self.kernel(self.x, self.x, self.theta)
        self.k_reg = self.k + self.n * (self.zeta ** 2) * np.eye(self.n)
        self.w = self.predict_weights(self.x)
        self.mean_sum_probability = self.w.sum(axis=0).mean()  # prod()
        self.p_pred = self.predict_proba(self.x, normalize=False)
        self.y_pred = _classify(self.p_pred, classes=self.classes)

        # prediction constraint
        self.constraint_matrix = self.n_classes * self.p_pred.T - 1
        self.constraint_vector = self.constraint_matrix[self.y_indices,
                                                        np.arange(self.n)]
        self.train_accuracy = np.mean(self.y_pred == self.y.ravel())
        self.complexity = self.compute_complexity()
        return self

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
        k_query = self.kernel(self.x, x_query, self.theta)
        w_query = _solve_posdef(self.k_reg, k_query)[0]
        return w_query

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
        w_query = self.predict_weights(x_query)
        p_query = _expectance(self.y == self.classes, w_query)
        return _clip_normalize(p_query.T).T if normalize else p_query

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
        p_query = self.predict_proba(x_query, normalize=False)
        y_query = _classify(p_query, classes=self.classes)
        return y_query

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
        w_query = self.predict_weights(x_query)
        p_query = _expectance(self.y == self.classes, w_query)
        p_field = p_query[:, self.y_indices.ravel()]  # (n_query, n_train)
        p_field[p_field <= 0] = 1
        h_query = np.einsum('ij,ji->i', -np.log(p_field), w_query)
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
        l_query = self.output_kernel(self.y, y_query, self.psi)
        v_query = _solve_posdef(self.l_reg, l_query)[0]
        return v_query

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
        v_query = self.reverse_weights(y_query)
        return np.dot(self.kernel(x_query, self.x, self.theta), v_query)

    def input_expectance(self):
        """
        Predict the expectance of the input conditioned on a class.

        Returns
        -------
        numpy.ndarray
            The expectance of each feature for each class (n_class, d)
        """
        y_query = self.classes[:, np.newaxis]
        v_query = self.reverse_weights(y_query)
        return _expectance(self.x, v_query)

    def input_variance(self):
        """
        Predict the variance of the input conditioned on a class.

        Returns
        -------
        numpy.ndarray
            The variance of each feature for each class (n_class, d)
        """
        y_query = self.classes[:, np.newaxis]
        v_query = self.reverse_weights(y_query)
        return _variance(self.x, v_query)

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
            # x_class = self.x[self.y.ravel() == i]
            # x_min = np.min(x_class, axis=0)
            # x_max = np.max(x_class, axis=0)
            # bounds = [(x_min[i], x_max[i]) for i in range(len(x_init))]
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
