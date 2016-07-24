"""
Bayesian Inference for Kernel Embeddings Module.

In the inference module, kernel functions are by default the Gaussian kernel
unless otherwise specified through a keyword argument, since the learning that
is implemented in the learning module is done with the Gaussian kernel only for
the time being.
"""
import numpy as np
from .kernels import gaussian as _gaussian
from .linalg import solve_posdef as _solve_posdef
from .kbr import posterior_fields as _posterior_fields
from .optimize import local_optimization as _local_optimization
from scipy.signal import argrelextrema as _argrelextrema


def uniform_weights(n):
    """
    Obtain uniform weights on a given dataset.

    Parameters
    ----------
    n : int
        The number of data points to place the uniform weights on

    Returns
    -------
    numpy.ndarray
        Uniform weights on each data point
    """
    return np.ones((n, 1)) / n


def embedding(x, theta, w=None, k=_gaussian):
    """
    Obtain empirical embedding on a given dataset.

    Parameters
    ----------
    x : numpy.ndarray
        The dataset to be used for generating the embedding (n, d)
    theta : numpy.ndarray
        Hyperparameters of the kernel (d,)
    w : numpy.ndarray, optional
        Weights on each data point. Default weights are uniform weights (n,)
    k : callable, optional
        Kernel functions from the kernels module. Default is a Gaussian kernel

    Returns
    -------
    callable
        The empirical embedding with feature maps on each data point
    """
    w = uniform_weights(x.shape[0]) if w is None else w

    def embedding_function(x_q):
        """
        The empirical embedding with feature maps on each data point.

        Parameters
        ----------
        x_q : numpy.ndarray
            The query points where the embedding is to be evaluated at (n_q, d)

        Returns
        -------
        numpy.ndarray
            The embedding evaluated at the query points (n_q, 1)
        """
        return np.dot(k(x_q, x, theta), w)
    return embedding_function


def conditional_weights(x, theta_x, x_q,
                        zeta=0, k_x=_gaussian, k_xx=None):
    """
    Obtain the empirical conditional embedding on a given dataset.

    Here y is the target output to be modelled and x is the input covariate.

    Parameters
    ----------
    x : numpy.ndarray
        The collected dataset of the input covariates (n, d_x)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (d_x,)
    x_q : numpy.ndarray
        The query inputs (n_q, d_x)
    zeta : float
        The regularisation parameter, interpreted as the likelihood noise level
    k_x : callable, optional
        The kernel on x [Default is the Gaussian kernel]
    k_xx : numpy.ndarray, optional
        The gramix on x cached beforehand to avoid recomputation

    Returns
    -------
    numpy.ndarray
        The conditional weights for the embedding(s)
    """
    k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
    k_xx_reg = k_xx + zeta ** 2 * np.eye(x.shape[0])
    return _solve_posdef(k_xx_reg, k_x(x, x_q, theta_x))[0]


def conditional_embedding(x, y, theta_x, theta_y,
                          zeta=0, k_x=_gaussian, k_y=_gaussian, k_xx=None):
    """
    Obtain the empirical conditional embedding on a given dataset.

    Here y is the target output to be modelled and x is the input covariate.

    Parameters
    ----------
    x : numpy.ndarray
        The collected dataset of the input covariates (n, d_x)
    y : numpy.ndarray
        The collected dataset of the output targets (n, d_y)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (d_x,)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)
    zeta : float
        The regularisation parameter, interpreted as the likelihood noise level
    k_x : callable, optional
        The kernel on x [Default is the Gaussian kernel]
    k_y : callable, optional
        The kernel on y [Default is the Gaussian kernel]
    k_xx : numpy.ndarray, optional
        The gramix on x cached beforehand to avoid recomputation

    Returns
    -------
    callable
        The conditional embedding of Y | X, to be evaluated at (y_q, x_q)
    """
    def conditional_embedding_function(y_q, x_q):
        """
        The conditional embedding of Y | X, to be evaluated at (y_q, x_q).

        Parameters
        ----------
        y_q : numpy.ndarray
            The query points in y (n_y_q, d_y)
        x_q : numpy.ndarray
            The query points in x (n_x_q, d_x)

        Returns
        -------
        numpy.ndarray
            The evaluated conditional embedding at query points (n_y_q, n_x_q)
        """
        w = conditional_weights(x, theta_x, x_q, zeta=zeta, k_x=k_x, k_xx=k_xx)
        return np.dot(k_y(y_q, y, theta_y), w)
    return conditional_embedding_function


def posterior_weights(prior_embedding, x, y, theta_x, theta_y, x_q,
                      epsil=0, delta=0,
                      kbr='tikhonov',
                      k_x=_gaussian, k_y=_gaussian,
                      k_xx=None, k_yy=None,
                      v=None):
    """
    Obtain the empirical posterior embedding using KBR on a given dataset.

    Parameters
    ----------
    prior_embedding : callable
        The prior embedding on y
    x : numpy.ndarray
        The collected dataset of the input covariates (n,)
    y : numpy.ndarray
        The collected dataset of the output targets (n,)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (d_x,)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)
    x_q : numpy.ndarray
        The query locations of the input covariates (n_q,)
    epsil : float, optional
        The regularisation parameter for the prior
        This is not needed if v is already supplied
    delta : float, optional
        The regularisation parameter for the likelihood
        This is not needed if v is already supplied
    kbr : str
        A string denoting the type of Kernels Bayes Rule to be applied
        This is not needed if v is already supplied
    k_x : callable, optional
        The kernel on x [Default is the Gaussian kernel]
    k_y : callable, optional
        The kernel on y [Default is the Gaussian kernel]
    k_xx : numpy.ndarray, optional
        The gramix on x if cached beforehand to avoid recomputation
        This is not needed if v is already supplied
    k_yy : numpy.ndarray, optional
        The gramix on y if cached beforehand to avoid recomputation
        This is not needed if v is already supplied
    v : numpy.ndarray, optional
        The posterior weight matrix if cached before to avoid recomputation

    Returns
    -------
    callable
        The posterior embedding of Y | X, to be evaluated at (y, x)
    """
    if v is None:
        k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
        k_yy = k_y(y, y, theta_y) if not k_yy else k_yy
        v = _posterior_fields[kbr](prior_embedding(y), k_xx, k_yy, epsil, delta)
    return np.dot(v, k_x(x, x_q, theta_x))


def posterior_embedding(prior_embedding, x, y, theta_x, theta_y,
                        epsil=0, delta=0,
                        kbr='tikhonov',
                        k_x=_gaussian, k_y=_gaussian,
                        k_xx=None, k_yy=None,
                        v=None):
    """
    Obtain the empirical posterior embedding using KBR on a given dataset.

    Parameters
    ----------
    prior_embedding : callable
        The prior embedding on y
    x : numpy.ndarray
        The collected dataset of the input covariates (n,)
    y : numpy.ndarray
        The collected dataset of the output targets (n,)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (d_x,)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)
    epsil : float, optional
        The regularisation parameter for the prior
        This is not needed if v is already supplied
    delta : float, optional
        The regularisation parameter for the likelihood
        This is not needed if v is already supplied
    kbr : str
        A string denoting the type of Kernels Bayes Rule to be applied
        This is not needed if v is already supplied
    k_x : callable, optional
        The kernel on x [Default is the Gaussian kernel]
    k_y : callable, optional
        The kernel on y [Default is the Gaussian kernel]
    k_xx : numpy.ndarray, optional
        The gramix on x if cached beforehand to avoid recomputation
        This is not needed if v is already supplied
    k_yy : numpy.ndarray, optional
        The gramix on y if cached beforehand to avoid recomputation
        This is not needed if v is already supplied
    v : numpy.ndarray, optional
        The posterior weight matrix if cached before to avoid recomputation

    Returns
    -------
    callable
        The posterior embedding of Y | X, to be evaluated at (y_q, x_q)
    """
    if v is None:
        k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
        k_yy = k_y(y, y, theta_y) if not k_yy else k_yy
        v = _posterior_fields[kbr](prior_embedding(y), k_xx, k_yy, epsil, delta)

    def posterior_embedding_function(y_q, x_q):
        """
        The posterior embedding of Y | X, to be evaluated at (y_q, x_q).

        Parameters
        ----------
        y_q : numpy.ndarray
            The query points in y (n_y_q, d_y)
        x_q : numpy.ndarray
            The query points in x (n_x_q, d_x)

        Returns
        -------
        numpy.ndarray
            The evaluated posterior embedding at the query points (n_y_q, n_x_q)
        """
        w = np.dot(v, k_x(x, x_q, theta_x))
        return np.dot(k_y(y_q, y, theta_y), w)
    return posterior_embedding_function


def mode(mu, xv_start, xv_min, xv_max):
    """
    Determine a density mode (peak), given its kernel embedding representation.

    Note that for stationary kernels, the peak of the density is located at the
    same places as the peak of the kernel embedding.

    Parameters
    ----------
    mu : callable
        The kernel embedding
    xv_start : numpy.ndarray
        The starting location for which mode searching begins (d_x,)
    xv_min : numpy.ndarray
        The lower bound of the rectangular search region (d_x,)
    xv_max : numpy.ndarray
        The upper bound of the rectangular search region (d_x,)

    Returns
    -------
    numpy.ndarray
        The mode location (d_x,)
    """
    # Define the objective to be minimised
    # For stationary kernels, the objective to optimise can be reduced to simply
    # the embedding
    def objective(xvq):
        return -mu(np.array(xvq, ndmin=2))[0, 0]

    # Find the mode
    x_mode, f_mode = _local_optimization(objective, xv_min, xv_max, xv_start)

    # Size: (d_x,)
    return x_mode


def multiple_modes(mu, xv_min, xv_max, n_modes=10):
    """
    Determine density modes (peaks), given its kernel embedding representation.

    Note that for stationary kernels, the peak of the density is located at the
    same places as the peak of the kernel embedding.

    Parameters
    ----------
    mu : callable
        The kernel embedding
    xv_min : numpy.ndarray
        The lower bound of the rectangular search region (d_x,)
    xv_max : numpy.ndarray
        The upper bound of the rectangular search region (d_x,)
    n_modes : int, optional
        The number of modes to search for (some of them will converge together)

    Returns
    -------
    numpy.ndarray
        The mode locations (n_modes, d_x)
    """
    # Make sure these are arrays
    xv_min = np.array(xv_min)
    xv_max = np.array(xv_max)

    # Generate a list of starting points
    n_dims, = xv_min.shape
    standard_range = np.random.rand(n_modes, n_dims)
    xv_start_list = (xv_max - xv_min) * standard_range + xv_min

    # Compute the modes
    # Size: (n_modes, d_x)
    return np.array([mode(mu, xv_start, xv_min, xv_max)
                     for xv_start in xv_start_list])


def conditional_modes(mu_yx, x_q, yv_min, yv_max, n_modes=10):
    """
    Determine a conditional density mode (peak), given its kernel embedding.

    Note that for stationary kernels, the peak of the density is located at the
    same places as the peak of the kernel embedding.

    Parameters
    ----------
    mu_yx : callable
        The conditional kernel embedding
    x_q : numpy.ndarray
        The query points (n_q, d_x)
    yv_min : numpy.ndarray
        The lower bound of the rectangular search region (d_y,)
    yv_max : numpy.ndarray
        The upper bound of the rectangular search region (d_y,)
    n_modes : int, optional
        The number of modes to search for at each query point

    Returns
    -------
    numpy.ndarray
        The x coordinates of the mode locations (n_query, n_modes, n_x_dims)
    numpy.ndarray
        The y coordinates of the mode locations (n_query, n_modes, n_y_dims)
    """
    def modes_of(mu):
        """
        Find the modes of a given embedding.

        Parameters
        ----------
        mu : callable
            A given joint embedding.

        Returns
        -------
        numpy.ndarray
            The mode locations (n_modes, d_y)
        """
        return multiple_modes(mu, yv_min, yv_max, n_modes=n_modes)

    # This finds the modes at the embedding conditioned at x
    def modes_at(x_query):
        """
        Find the modes at the embedding conditioned at x.

        Parameters
        ----------
        x_query : numpy.ndarray
            A query point (1, d_x)

        Returns
        -------
        numpy.ndarray
            The mode locations (n_modes, d_y)
        """
        return modes_of(lambda y_q: mu_yx(y_q, x_query))

    # This computes the modes at all query points
    # Size: (n_query, n_modes, n_dims)
    y_modes = np.array([modes_at(x) for x in x_q[:, np.newaxis]])

    # This returns the corresponding input coordinates for each mode
    # Size: (n_query, n_modes, n_dims)
    x_modes = np.repeat(x_q[:, np.newaxis], n_modes, axis=1)

    # Return the modes
    # Size: (n_query, n_modes, n_x_dims)
    # Size: (n_query, n_modes, n_y_dims)
    return x_modes, y_modes


def search_modes_from_conditional_embedding(conditional_embedding_q, x_q, y_q):
    """
    Search for modes in the conditional density given a fully and densely
    queried conditional embedding.

    Parameters
    ----------
    conditional_embedding_q : numpy.ndarray
        The fully and densely queried conditional embedding (n_y_q, n_x_q)
    x_q : numpy.ndarray
        The query points in x (n_x_q, d_x)
    y_q : numpy.ndarray
        The query points in y (n_y_q, d_y)

    Returns
    -------
    list
        The x locations of the modes in the conditional embedding
        The y locations of the modes in the conditional embedding
    """
    n_y_q, n_x_q = conditional_embedding_q.shape
    assert n_x_q == x_q.shape[0]
    assert n_y_q == y_q.shape[0]
    x_peaks = np.array([])
    y_peaks = np.array([])

    for i in range(n_x_q):
        ind = _argrelextrema(conditional_embedding_q[:, i], np.greater)
        for j in ind[0]:
            x_peaks = np.append(x_peaks, x_q[i])
            y_peaks = np.append(y_peaks, y_q[j])

    return x_peaks, y_peaks


def kernels_bayes_average(g_y, w):
    """
    Obtain the conditional kernels Bayes average of a function g.

    Parameters
    ----------
    g_y : numpy.ndarray
        The callable realisations at training outputs (n,)
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The average of the callable at the query points (n_q,)
    """
    return np.dot(g_y, w)


def expectance(y, w):
    """
    Obtain the expectance from an empirical embedding.

    Parameters
    ----------
    y : numpy.ndarray
        The training outputs (n, d_y)
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q,)

    Returns
    -------
    numpy.ndarray
        The conditional expected value of the output (d_y, n_q,)
    """
    return np.dot(y.T, w)


def variance(y, w):
    """
    Obtain the variance from an empirical embedding.

    Parameters
    ----------
    y : numpy.ndarray
        The training outputs (n, d_y)
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The conditional covariance value of the output (d_y, n_q)
    """
    # Compute the expectance (d_y, n_q)
    y_q_exp = np.dot(y.T, w)

    # Compute the expectance of squares (d_y, n_q)
    y_q_exp_sq = np.dot((y ** 2).T, w)

    # Compute the variance (d_y, n_q)
    return y_q_exp_sq - (y_q_exp ** 2)


def clip_normalise(w):
    """
    Use the clipping method to normalize weights.

    Parameters
    ----------
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The clip-normalized conditional or posterior weight matrix (n, n_q)
    """
    w_clip = np.clip(w, 0, np.inf)
    return w_clip / np.sum(w_clip, axis=0)
