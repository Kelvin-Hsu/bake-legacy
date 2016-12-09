"""
Gaussian process based Bayesian optimisers.
"""
import autograd.numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.stats import norm
from sklearn.utils.validation import check_X_y, check_is_fitted

class GPREI(GaussianProcessRegressor):
    """
    Template Only. Not computationally efficient AT ALL.
    """
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=20,
                 normalize_y=False, copy_X_train=True, random_state=None,
                 n_stop_train=np.inf):
        self.n_stop_train=n_stop_train
        super().__init__(kernel=kernel,
                         alpha=alpha,
                         optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y,
                         copy_X_train=copy_X_train,
                         random_state=random_state)

    def fit(self, X, y):
        return super().fit(X, y)

    def update(self, X_new, y_new):

        if self.X_train_.shape[0] > self.n_stop_train:
            print('Just Adding')
            return self.add(X_new, y_new)
        else:
            print('Refitting')
            self.X_train_ = np.concatenate((self.X_train_, X_new), axis=0)
            self.y_train_ = np.concatenate((self.y_train_, y_new), axis=0)
            return self.fit(self.X_train_, self.y_train_)

    def add(self, X_new, y_new):
        """
        Add data to the model and re-learn the parameters.

        Parameters
        ----------
        X_new : ndarray (n,m)
                The new features vectors to be included in the model
        y_new : ndarray (n,)
                The new targets to be included in the model
        """
        check_is_fitted(self, ['L_', 'alpha_'])

        X_new, y_new = check_X_y(X_new, y_new)

        self.X_train_ = np.concatenate((self.X_train_, X_new), axis=0)
        self.y_train_ = np.concatenate((self.y_train_, y_new), axis=0)
        # TODO: decide what to do about self.y_mean_
        self.L_ = np.linalg.cholesky(
            self.kernel_(self.X_train_) +
            self.alpha * np.eye(self.X_train_.shape[0])
        )
        self.alpha_ = np.linalg.solve(self.L_.T,
                                      np.linalg.solve(self.L_, self.y_train_))
        self.log_marginal_likelihood_value_ = \
            -0.5 * self.y_train_[np.newaxis, :].dot(
                self.alpha_[:, np.newaxis])[0, 0] - \
            np.sum(np.log(np.diag(self.L_))) - \
            self.X_train_.shape[0] / 2 * np.log(2 * np.pi)
        return self

    def acquisition(self, x_query):
        mu, std = self.predict(x_query, return_std=True)
        return gaussian_expected_improvement(mu, std, np.max(self.y_train_))

    def pick(self, x_query):
        acq = self.acquisition(x_query)
        return x_query[[np.argmax(acq)]]


def gaussian_expected_improvement(mu, std, best):
    """
    Expected Improvement Acquisition Function for a Gaussian process.

    Parameters
    ----------
    mu : numpy.ndarray
        The mean of the predictions (n_q,)
    std : numpy.ndarray
        The standard deviations of the predictions (n_q,)
    best : float
        The maximum observed value so far

    Returns
    -------
    numpy.ndarray
        The acquisition function evaluated at the corresponding points (n_q,)
    """
    diff = mu - best
    abs_diff = np.abs(diff)
    clip_diff = np.clip(diff, 0, np.inf)
    return clip_diff + std*norm.pdf(diff/std) - abs_diff*norm.cdf(-abs_diff/std)