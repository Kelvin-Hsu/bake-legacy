"""
Models Module.
"""
import autograd.numpy as np
from .infer import expectance as _expectance
from .infer import clip_normalize as _clip_normalize
from .infer import classify as _classify
from .kernels import s_gaussian as _s_gaussian
from .optimize import explore_optimization as _explore_optimization
from .linalg import solve_posdef as _solve_posdef
from sklearn.model_selection import KFold as _KFold
from scipy.linalg import cholesky as _cholesky
from scipy.optimize import minimize as _minimize


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

    # def fit_cross_val(self, x, y,
    #                   h=None,
    #                   h_min=np.array([0.00, 0.00, 0.001]),
    #                   h_max=np.array([10.0, 10.0, 1.000]),
    #                   h_init=np.array([1.0, 1.00, 0.010]),
    #                   n_splits=10,
    #                   verbose=True):
    #
    #     self.x = x.copy()
    #     self.y = y.copy()
    #     self.classes = np.unique(self.y)
    #     self.class_indices = np.arange(self.classes.shape[0])
    #     self.n_classes = self.classes.shape[0]
    #
    #     if h is None:
    #
    #         kf = _KFold(n_splits=n_splits)
    #
    #         self.best_so_far = np.inf
    #
    #         def objective(hyperparam):
    #
    #             theta, zeta = hyperparam[:-1], hyperparam[-1]
    #
    #             total_score = 0
    #             for train, test in kf.split(self.x):
    #                 X_train, X_test, y_train, y_test = self.x[train], \
    #                                                    self.x[test], \
    #                                                    self.y[train], \
    #                                                    self.y[test]
    #                 w_query = _conditional_weights(X_train, theta, X_test,
    #                                                zeta=zeta, k_x=self.kernel)
    #                 p_query = _expectance(y_train == self.classes, w_query)
    #                 y_query = self.classes[np.argmax(p_query, axis=0)]
    #                 accuracy = np.mean(y_query == y_test)
    #                 total_score += accuracy
    #             score = -total_score / n_splits
    #             if verbose:
    #                 if score < self.best_so_far:
    #                     print('Hyperparam: ', hyperparam, '|| Score: ', score, '  *IMPROVEMENT*')
    #                     self.best_so_far = score
    #                 else:
    #                     print('Hyperparam: ', hyperparam, '|| Score: ', score)
    #             return score
    #
    #         h_opt, score = _explore_optimization(objective, h_min, h_max, n_samples=2000)
    #
    #         self.theta, self.zeta = h_opt[:-1], h_opt[-1]
    #         if verbose:
    #             print('The final hyperparameters are: ', self.theta, self.zeta)
    #
    #     else:
    #         self.theta, self.zeta = h[:-1], h[-1]
    #
    #     self.k = self.kernel(self.x, self.x, self.theta)
    #     self.k_reg = self.k + self.n * (self.zeta ** 2) * np.eye(self.n)
    #     return self

    def fit(self, x, y,
            h=None,
            h_min=np.array([0.20, 0.20, 0.001]),
            h_max=np.array([10.0, 10.0, 1.000]),
            h_init=np.array([1.0, 1.00, 0.010]),
            verbose=True):
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
        self.n_classes = self.classes.shape[0]

        def model_complexity(w):
            # f = np.sum(w.diagonal())
            f = np.sum(np.dot(w.T, w).diagonal())
            return np.log(f)

        def constraint(hypers):
            self.update(hypers[:-1], hypers[-1])
            w = self.predict_weights(x)
            y_pred = self.predict(x)
            c = np.mean(y_pred == self.y.ravel()) - 1
            if verbose:
                f = model_complexity(w)
                s = 'Training Accuracy: %f || Objective: %f' % (1 + c, f)
                print('Hyperparameters: ', hypers, s)
            return c

        def objective(hypers):
            self.update(hypers[:-1], hypers[-1])
            w = self.predict_weights(x)
            f = model_complexity(w)
            return f

        if h is None:

            # options = {'maxiter': 50}
            bounds = [(h_min[i], h_max[i]) for i in range(len(h_init))]
            constraints_ineq = {'type': 'ineq', 'fun': constraint}
            constraints = constraints_ineq
            optimal_result = _minimize(objective, h_init,
                                       bounds=bounds,
                                       constraints=constraints)
            h = optimal_result.x
            if verbose:
                print('Training Completed.')

        c = constraint(h)
        f = objective(h)
        s = 'Training Accuracy: %f || Objective: %f' % (1 + c, f)
        print('Hyperparameters: ', h, s)
        self.update(h[:-1], h[-1])

        return self

    def update(self, theta, zeta):
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
        self.theta = theta
        self.zeta = zeta
        self.k = self.kernel(self.x, self.x, self.theta)
        self.k_reg = self.k + self.n * (self.zeta ** 2) * np.eye(self.n)
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
        p_field = p_query[:, self.y.ravel()]  # (n_query, n_train)
        p_field[p_field <= 0] = 1
        h_query = np.einsum('ij,ji->i', -np.log(p_field), w_query)
        return np.clip(h_query, 0, np.inf) if clip else h_query