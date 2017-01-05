"""
Models Module.
"""
import autograd.numpy as np
from .infer import conditional_weights as _conditional_weights
from .infer import expectance as _expectance
from .kernels import gaussian as _gaussian


class Classifier():

    def __init__(self, kernel=_gaussian):
        self.kernel = kernel

    def fit(self, X, y, hyperparam=None):

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes = np.unique(self.y_train_)
        self.class_indices = np.arange(self.classes.shape[0])
        self.n_classes = self.classes.shape[0]

        if hyperparam is None:
            # Do Learning
            self.theta = 1.0
            self.zeta = 0.1
        else:
            self.theta, self.zeta = hyperparam

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