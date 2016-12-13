"""
Active Sampler Module.

Active sampler classes for Bayesian optimization.

For utility and benchmarking purposes, each active sampler class must
follow the form specified by the class '__EMPTY_TEMPLATE__' no matter what.
"""
import autograd.numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_X_y, check_is_fitted
from bayesian_optimization.acquisition_functions import \
    gaussian_expected_improvement
import logging

# TODO: USE KERNEL HERDING TO JUMP START ACTIVE SAMPLER


class __ANY_SUPER_CLASS__():
    """
    An empty template for some superclass.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialise some class.

        Parameters
        ----------
        args
            Any arguments
        kwargs
            Any keyword arguments

        Returns
        -------
        Anything
        """
        return None


class __EXAMPLE_SAMPLER__(__ANY_SUPER_CLASS__):
    """
    An empty template to demonstrate the basic building blocks of an active
    sampler.

    There are no restrictions to the type of superclass the active sampler
    inherits from.

    The active sampler must have the following methods. It is allowed to have
    more methods, but they will not be explicitly called during the testing
    stage.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the active sampler.

        No specific formats required. This part is kept general so that the
        model can be specified as specifically as required when it is
        initialized, because for testing purposes no intervention can be made
        after this stage by an expert.

        The test script will call a function or module that sets up each
        active sampler for comparison in which we would write ourselves. This is
        the only part we can modify. We may even create several instances of
        the same samplers, but with different initial parameters, and compare
        them against each other.

        For example, the exploration-exploitation trade-off ratio can be set
        here. For kernel-based active samplers, the kernel chosen can be
        set here. If these choices are to be changed during the actual run,
        then it should be inbuilt into the algorithm using one of the standard
        methods below. An example would be to increase the rate at which the
        sampler exploits during the 'update' stage, perhaps by some flag which
        is set when the sampler has explored enough.

        It is suggested to keep an internal memory of the current state of the
        active sampling procedure. For example, keep a copy of the training data
        collected so far. There will be a copy of that in the testing script,
        but it will not be fed into the active sampler explicitly.

        Parameters
        ----------
        args
            Any arguments
        kwargs
            Any keyword arguments

        Returns
        -------
        Anything
        """
        pass

    def fit(self, X, y, **kwargs):
        """
        Fit a model to the training data.

        For model based active samplers, the sampler would learn a model from
        the training data X and y. For model free active samplers, this function
        still has to exist, but it can do nothing within this function.

        During the testing stage, this function is only called once in the
        beginning where it would be initialised with some training dataset.
        If the model is to be refitted, it should do this during the 'update'
        stage. This is why it is important to keep a copy of the training data
        as an attribute of the sampler, since the 'update' method only takes
        new data points as the input and often the model would need both
        the old training data and new data points to be refitted.

        Keyword arguments are allowed, but they are not used during the testing
        stage, meaning that only their default values will be used.

        Parameters
        ----------
        X : numpy.ndarray
            A set of training inputs of size (n, n_dim)
        y : numpy.ndarray
            A set of training outputs of size (n,)
        kwargs
            Any keyword arguments

        Returns
        -------
        Anything
        """
        return None

    def predict(self, X_query, **kwargs):
        """
        Predict the outputs at the given query points.

        For model based active samplers, the sampler would predict the target
        outputs at the given query points 'X_query' using its model.
        For model free active samplers, this function still has the exist,
        but it can just output all zeros or garbage, as long as an array of
        prediction corresponding to the size of 'X_query' is returned.

        This function is really for visualisation purposes. It is not
        called for numerical benchmarking.

        Keyword arguments are allowed, but they are not used during the testing
        stage, meaning that only their default values will be used.

        Parameters
        ----------
        X_query : numpy.ndarray
            A set of query inputs of size (n_query, n_dim)
        kwargs
            Any keyword arguments

        Returns
        -------
        numpy.ndarray
            A set of query outputs of size (n_query,)
        """
        n_query, n_dim = X_query.shape
        y_query_expected = np.random.normal(loc=0, scale=1, size=n_query)
        return y_query_expected

    def acquisition(self, X_query, **kwargs):
        """
        Compute the acquisition function used for this active sampler.

        Output an acquisition value at each query location.
        For model based active samplers, the sampler would compute the
        acquisition value at the given query points 'X_query' using its model.
        For model free active samplers, this function still has the exist,
        but it can just output all zeros or garbage, as long as an array of
        prediction corresponding to the size of 'X_query' is returned.

        For example, a Delaunay active sampler would not need an acquisition
        function, but an active sampling decision can still be encoded in the
        pick function described in the next block.

        This function is really for visualisation purposes. It is not
        called for numerical benchmarking.

        Keyword arguments are allowed, but they are not used during the testing
        stage, meaning that only their default values will be used.

        Parameters
        ----------
        X_query : numpy.ndarray
            A set of query inputs of size (n_query, n_dim)
        kwargs
            Any keyword arguments

        Returns
        -------
        numpy.ndarray
            An array of acquisition values of size (n_query,)
        """
        n_query, n_dim = X_query.shape
        acquisition_query = np.random.normal(loc=0, scale=1, size=n_query)
        return acquisition_query

    def pick(self, X_query, **kwargs):
        """
        Pick a query location to sample from a set of given query locations.

        This is an essential function of the active sampler.

        If a valid acquisition method is written, then a typical pick method
        would be written as follows.
        >>> def pick(self, X_query):
        >>>     # Pick the query location with the highest acquisition value
        >>>     acquisition_query = self.acquisition(X_query)
        >>>     return X_query[[np.argmax(acquisition_query)]]

        During the testing stage, this function would be repeated called in
        each iteration.

        Warning: Make sure that the returned location is still a
        two dimensional array, but with one row, and not a one dimensional array

        Keyword arguments are allowed, but they are not used during the testing
        stage, meaning that only their default values will be used.

        Parameters
        ----------
        X_query : numpy.ndarray
            A set of query inputs of size (n_query, n_dim)
        kwargs
            Any keyword arguments

        Returns
        -------
        numpy.ndarray
            One single query point of size (1, n_dim), NOT of size (n_dim,)
        """
        acquisition_query = self.acquisition(X_query)
        return X_query[[np.argmax(acquisition_query)]]

    def update(self, X_new, y_new, **kwargs):
        """
        Update the active sampler with a new observation.

        For model based active samplers, the sampler would re-learn a model from
        the training data X and y. For model free active samplers, this function
        still has to exist, but it can do nothing within this function.

        During the testing phase, this function would be repeated called in
        each iteration.

        Usually only one new data point is available, but some benchmark may
        input more, so the function must be written for the general case.
        That is 'n_new' is usually 1, but the method would be written to take
        inputs with arbitrary 'n_new'.

        Keyword arguments are allowed, but they are not used during the testing
        stage, meaning that only their default values will be used.

        Parameters
        ----------
        X_new : numpy.ndarray
            A set of new inputs of size (n_new, n_dim)
        y_new : numpy.ndarray
            A set of new outputs of size (n_new,)
        kwargs
            Any keyword arguments
        Returns
        -------
        Anything
        """
        return None

# The example sampler is actually just a random sampler
RandomSampler = __EXAMPLE_SAMPLER__

ACQ_EI = ['Expected Improvement', 'Improvement', 'EI',]
ACQ_EXP = ['Expectance', 'expectance', 'Expected Value', 'expected value',
           'Mean Prediction', 'Mean prediction', 'mean prediction',
           'Prediction', 'prediction',
           'Mean', 'mean', 'MU', 'Mu', 'mu']
ACQ_STD = ['Standard Deviation', 'SD', 'STD', 'Sd', 'Std', 'sd', 'std']
ACQ_VAR = ['Variance', 'VAR', 'Var', 'var']
ACQ_UCB = ['Upper Confidence Bound', 'upper confidence bound', 'UCB', 'ucb']


class GaussianProcessSampler(GaussianProcessRegressor):
    """
    Gaussian process regressor based Bayesian optimiser using
    Expected Improvement as the acquisition criterion.

    Notice that this follows the active sampler template described at the top
    of this page.

    Since this is inherited from scikit-learn's 'GaussianProcessRegressor',
    we will not repeat documentations for shared methods and attributes.

    Parameters
    ----------
    acquisition : str
        A string indicating the type of acquisition function used.
    n_start : int
        The number of random samples to collect before actual active sampling.
    n_stop_train : int
        When the model has collected more than 'n_stop_train' data points, the
        hyperparameters will be kept fixed and no longer trained.

    Attributes
    ----------
    acquisition : str
        A string indicating the type of acquisition function used.
    n_stop_train : int
        When the model has collected more than 'n_stop_train' data points, the
        hyperparameters will be kept fixed and no longer trained.
    """
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=20,
                 normalize_y=False, copy_X_train=True, random_state=None,
                 acquisition_method='EI', n_start=10, n_stop_train=np.inf):
        self.acquisition_method = acquisition_method
        self.n_start = n_start
        self.n_stop_train = n_stop_train
        super().__init__(kernel=kernel,
                         alpha=alpha,
                         optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y,
                         copy_X_train=copy_X_train,
                         random_state=random_state)

    def fit(self, X, y):
        """
        See scikit-learn's 'GaussianProcessRegressor'
        """
        return super().fit(X, y)

    def predict(self, X_query, return_std=False, return_cov=False):
        """
        See scikit-learn's 'GaussianProcessRegressor'
        """
        return super().predict(X_query,
                               return_std=return_std, return_cov=return_cov)

    def acquisition(self, X_query):
        """
        Compute the acquisition function according to the method chosen.

        The acquisition functions implemented so far are:
            - Expected Improvement
            - Mean Prediction
            - Standard Deviation
            - Variance

        Parameters
        X_query : numpy.ndarray
            A set of query inputs of size (n_query, n_dim)

        Returns
        -------
        numpy.ndarray
            An array of acquisition values of size (n_query,)
        """
        mu, std = self.predict(X_query, return_std=True)
        if self.acquisition_method in ACQ_EI:
            return gaussian_expected_improvement(mu, std, np.max(self.y_train_))
        elif self.acquisition_method in ACQ_EXP:
            return mu
        elif self.acquisition_method in ACQ_STD:
            return std
        elif self.acquisition_method in ACQ_VAR:
            return std**2
        else:
            return ValueError('No acquisition method named %s' %
                              self.acquisition_method)

    def pick(self, X_query):
        """
        Pick a query location to sample from a set of given query locations.

        Parameters
        ----------
        X_query : numpy.ndarray
            A set of query inputs of size (n_query, n_dim)

        Returns
        -------
        numpy.ndarray
            One single query point of size (1, n_dim), NOT of size (n_dim,)
        """
        if not hasattr(self, "X_train_") \
                or self.X_train_.shape[0] < self.n_start:
            logging.debug('picking new points randomly')
            n_query, n_dim = X_query.shape
            random_index = np.random.randint(n_query)
            return X_query[[random_index]]
        else:
            acquisition_query = self.acquisition(X_query)
            return X_query[[np.argmax(acquisition_query)]]

    def update(self, X_new, y_new):
        """
        Update the active sampler with a new observation.

        Parameters
        ----------
        X_new : numpy.ndarray
            A set of new inputs of size (n_new, n_dim)
        y_new : numpy.ndarray
            A set of new outputs of size (n_new,)

        Returns
        -------
        self
            The current instance of the sampler.
        """
        # If no training data has been collected before, just fit to the
        # initial data
        if not hasattr(self, "X_train_"):
            logging.debug('"update" called before "fit"; '
                          'fitting to incoming data')
            return self.fit(X_new, y_new)

        # Determine if we should train or not
        logging.debug('sampler now has %d data points' % self.X_train_.shape[0])
        if self.X_train_.shape[0] > self.n_stop_train:
            # If not, then just add the data points with a Cholesky update
            logging.debug('adding data without re-training')
            return self.add(X_new, y_new)
        else:
            # Otherwise, add the data points and refit the model.
            logging.debug('retraining...')
            X_new, y_new = check_X_y(X_new, y_new)
            self.X_train_ = np.concatenate((self.X_train_, X_new), axis=0)
            self.y_train_ = np.concatenate((self.y_train_, y_new), axis=0)
            return self.fit(self.X_train_, self.y_train_)

    def add(self, X_new, y_new):
        """
        Add data to the model without updating the hyperparameters.

        Instead, the Cholesky matrix and related quantities are updated.

        Parameters
        ----------
        X_new : numpy.ndarray
            A set of new inputs of size (n_new, n_dim)
        y_new : numpy.ndarray
            A set of new outputs of size (n_new,)

        Returns
        -------
        self
            The current instance of the sampler.
        """
        # Check that relevant quantities are fitted and the input data is
        # of the correct size, and add the data points to the training data
        check_is_fitted(self, ['L_', 'alpha_'])
        X_new, y_new = check_X_y(X_new, y_new)
        self.X_train_ = np.concatenate((self.X_train_, X_new), axis=0)
        self.y_train_ = np.concatenate((self.y_train_, y_new), axis=0)

        # TODO: Decide what to do about self.y_mean_

        # Recompute relevant quantities (copied from scikit-learn)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        self.L_ = cholesky(K, lower=True)  # Line 2
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

        # TODO: Not sure which way is correct

        # self.log_marginal_likelihood_value_ = self.log_marginal_likelihood()
        # print(self.log_marginal_likelihood_value_)
        # self.log_marginal_likelihood_value_ = \
        #     -0.5 * self.y_train_[np.newaxis, :].dot(
        #         self.alpha_[:, np.newaxis])[0, 0] - \
        #     np.sum(np.log(np.diag(self.L_))) - \
        #     self.X_train_.shape[0] / 2 * np.log(2 * np.pi)
        # print(self.log_marginal_likelihood_value_)
        return self




