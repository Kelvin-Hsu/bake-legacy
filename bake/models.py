"""
Models Module.
"""
import autograd.numpy as np
from .infer import conditional_weights as _conditional_weights
from .infer import expectance as _expectance
from .kernels import s_gaussian as _gaussian
from sklearn.model_selection import KFold as _KFold
from .optimize import explore_optimization as _explore_optimization
from scipy.optimize import minimize as _minimize


class Classifier():

    def __init__(self, kernel=_gaussian):
        self.kernel = kernel

    # def fit(self, X, y, hyperparam=None, h_min=None, h_max=None, h_init=None, n_splits=10, verbose=True):
    #
    #     self.X_train_ = X.copy()
    #     self.y_train_ = y.copy()
    #     self.classes = np.unique(self.y_train_)
    #     self.class_indices = np.arange(self.classes.shape[0])
    #     self.n_classes = self.classes.shape[0]
    #
    #     if hyperparam is None:
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
    #             for train, test in kf.split(self.X_train_):
    #                 X_train, X_test, y_train, y_test = self.X_train_[train], \
    #                                                    self.X_train_[test], \
    #                                                    self.y_train_[train], \
    #                                                    self.y_train_[test]
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
    #         self.theta, self.zeta = hyperparam[:-1], hyperparam[-1]
    #
    #     # w = _conditional_weights(self.X_train_, self.theta, self.X_train_,
    #     #                      zeta=self.zeta, k_x=self.kernel)
    #     # print(np.sum(w**2))
    #     return self

    def fit(self, X, y, hyperparam=None, h_min=None, h_max=None, h_init=None, verbose=True):

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes = np.unique(self.y_train_)
        self.class_indices = np.arange(self.classes.shape[0])
        self.n_classes = self.classes.shape[0]

        if hyperparam is None:

            def constraint(hypers):
                theta, zeta = hypers[:-1], hypers[-1]
                w = _conditional_weights(self.X_train_, theta,
                                         self.X_train_, zeta=zeta,
                                         k_x=self.kernel)
                p = _expectance(self.y_train_ == self.classes, w)
                y_pred = self.classes[np.argmax(p, axis=0)]
                c = np.mean(y_pred == self.y_train_.ravel()) - 1
                print(c)
                return c

            def objective(hypers):
                theta, zeta = hypers[:-1], hypers[-1]
                w = _conditional_weights(self.X_train_, theta,
                                         self.X_train_, zeta=zeta,
                                         k_x=self.kernel)
                f = np.sum(w**2)
                # print(f)
                return f

            method = 'COBYLA'
            options = {'disp': True, 'maxls': 20, 'gtol': 1e-10,
                       'eps': 1e-12, 'maxiter': 15000, 'ftol': 1e-9,
                       'maxcor': 20, 'maxfun': 15000}
            bounds = [(h_min[i], h_max[i]) for i in range(len(h_init))]
            constraints = {'type': 'ineq',
                           'fun': constraint}
            optimal_result = _minimize(objective, h_init,
                                       method=method, bounds=bounds,
                                       constraints=constraints,
                                       options=options)
            h_opt, f_opt = optimal_result.x, optimal_result.fun
            print(h_opt, objective(h_opt), constraint(h_opt))

            self.theta, self.zeta = h_opt[:-1], h_opt[-1]
            if verbose:
                print('The final hyperparameters are: ', self.theta, self.zeta)

        else:
            self.theta, self.zeta = hyperparam[:-1], hyperparam[-1]

        # w = _conditional_weights(self.X_train_, self.theta, self.X_train_,
        #                      zeta=self.zeta, k_x=self.kernel)
        # print(np.sum(w**2))
        return self

    def probability(self, X_query):

        w_query = _conditional_weights(self.X_train_, self.theta, X_query,
                                  zeta=self.zeta, k_x=self.kernel)
        return _expectance(self.y_train_ == self.classes, w_query)

    def predict(self, X_query):

        w_query = _conditional_weights(self.X_train_, self.theta, X_query,
                                  zeta=self.zeta, k_x=self.kernel)
        p_query = _expectance(self.y_train_ == self.classes, w_query)
        return self.classes[np.argmax(p_query, axis=0)]

    def entropy(self, X_query):

        w_query = _conditional_weights(self.X_train_, self.theta, X_query,
                                  zeta=self.zeta, k_x=self.kernel)
        p_query = _expectance(self.y_train_ == self.classes, w_query)
        p_field = p_query[self.y_train_.ravel(), :] # (n_train, n_query)

        def log(x):
            answer = np.log(x)
            answer[x <= 0] = 0
            return answer

        return np.einsum('ij,ji->i', -log(p_field).T, w_query)

    def infer(self, X_query):

        w_query = _conditional_weights(self.X_train_, self.theta, X_query,
                                  zeta=self.zeta, k_x=self.kernel)
        p_query = _expectance(self.y_train_ == self.classes, w_query)

        y_query = self.classes[np.argmax(p_query, axis=0)]

        # BE CAREFUL: Cannot always assume label and index are the same
        p_field = p_query[self.y_train_.ravel(), :] # (n_train, n_query)

        def log(x):
            x[x <= 0] = 1
            return np.log(x)

        h_query = np.einsum('ij,ji->i', -log(p_field).T, w_query)

        return y_query, p_query, h_query


class Regressor():

    def __init__(self, kernel=_gaussian):
        self.kernel = kernel

    def fit(self, X, y, hyperparam=None):

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        if hyperparam is None:
            # Do Learning
            self.theta = 1.0
            self.zeta = 0.1
        else:
            self.theta, self.zeta = hyperparam

        self.k_xx_ = self.kernel(self.X_train_, self.X_train_, self.theta)

    def predict(self, X_query):

        w_q = _conditional_weights(self.X_train_, self.theta, X_query,
                                   zeta=self.zeta, k_x=self.kernel,
                                   k_xx=self.k_xx_)
        return _expectance(self.y_train_, w_q)