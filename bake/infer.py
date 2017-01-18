"""
Bayesian Inference for Kernel Embeddings Module.

In the inference module, kernel functions are by default the Gaussian kernel
unless otherwise specified through a keyword argument, since the learning that
is implemented in the learning module is done with the Gaussian kernel only for
the time being.
"""
import autograd.numpy as np
from scipy.signal import argrelextrema as _argrelextrema
from scipy.special import erf as _erf
from scipy.optimize import root as _root
from .kernels import gaussian as _gaussian
from .linalg import solve_posdef as _solve_posdef
from .linalg import dist as _dist
from .linalg import gaussian_norm as _gaussian_norm
from .kbr import posterior_fields as _posterior_fields
from .optimize import local_optimization as _local_optimization
from .optimize import solve_normalized_unit_constrained_quadratic as _snucq


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
    n = x.shape[0]
    k_xx_reg = k_xx + n * zeta ** 2 * np.eye(n)
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


def embedding_to_weights(mu, x, k_xx):
    """
    Recover the weights in an embedding.

    Parameters
    ----------
    mu : callable
        The embedding of interest
    x : numpy.ndarray
        The observed data (n, d_x)
    k_xx : numpy.ndarray
        The gram matrix on x (n, n)

    Returns
    -------
    numpy.ndarray
        The recovered weights (n,)
    """
    return _solve_posdef(k_xx, mu(x))[0]


def conditional_embedding_to_weights(mu_y_x, y, k_yy, x_q):
    """
    Recover the weights in a conditional or posterior embedding at query points.

    Parameters
    ----------
    mu_y_x : callable
        The conditional or posterior embedding of interest
    y : numpy.ndarray
        The observed output data (n, d_x)
    k_yy : numpy.ndarray
        The gram matrix on x (n, n)
    x_q : numpy.ndarray
        The input query points (n_q,)

    Returns
    -------
    numpy.ndarray

    """
    return _solve_posdef(k_yy, mu_y_x(y, x_q))[0]


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


def clean_multiple_modes(mu, modes, ratio=0.25):
    """
    Clean away the local minimum that arises from the local optimizer.

    Parameters
    ----------
    mu : callable
        The kernel embedding
    modes : numpy.ndarray
        The modes that were found (n_modes, d_x)
    ratio : float, optional
        The ratio for the cut off

    Returns
    -------
    numpy.ndarray
        The cleaned modes (?, d_x)
    """
    mu_modes = mu(modes)
    cut_off = ratio * np.max(mu_modes)
    i_modes_clean = mu_modes > cut_off
    return modes[i_modes_clean.ravel(), :]


def cleaned_multiple_modes(mu, xv_min, xv_max, n_modes=10, ratio=0.25):
    """
    A wrapper for returning cleaned modes straight away.

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
    ratio : float, optional
        The ratio for the cut off

    Returns
    -------
    numpy.ndarray
        The cleaned modes (?, d_x)
    """
    modes = multiple_modes(mu, xv_min, xv_max, n_modes=n_modes)
    return clean_multiple_modes(mu, modes, ratio=ratio)


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
        The x coordinates of the mode locations (n_q, n_modes, d_x)
    numpy.ndarray
        The y coordinates of the mode locations (n_q, n_modes, d_y)
    """
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
        return multiple_modes(lambda y_q: mu_yx(y_q, x_query), yv_min, yv_max,
                              n_modes=n_modes)

    # This computes the modes at all query points
    # Size: (n_q, n_modes, n_dims)
    y_modes = np.array([modes_at(x) for x in x_q[:, np.newaxis]])

    # This returns the corresponding input coordinates for each mode
    # Size: (n_q, n_modes, n_dims)
    x_modes = np.repeat(x_q[:, np.newaxis], n_modes, axis=1)

    # Return the modes
    # Size: (n_q, n_modes, d_x)
    # Size: (n_q, n_modes, d_y)
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
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The conditional expected value of the output (d_y, n_q)
    """
    return np.dot(y.T, w).T


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
    # w = clip_normalize(w)
    # Compute the expectance (d_y, n_q)
    y_q_exp = np.dot(y.T, w)

    # Compute the expectance of squares (d_y, n_q)
    y_q_exp_sq = np.dot((y ** 2).T, w)

    # Compute the variance (d_y, n_q)
    return y_q_exp_sq - (y_q_exp ** 2)


def normalize(w):
    """
    Normalize weights.

    Parameters
    ----------
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The normalized conditional or posterior weight matrix (n, n_q)
    """
    return w / np.sum(w, axis=0)


def softmax_normalize(w):
    """
    Normalize weights.

    Parameters
    ----------
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The normalized conditional or posterior weight matrix (n, n_q)
    """
    w_softmax = np.exp(w)
    return w_softmax / np.sum(w_softmax, axis=0)


def softplus_normalize(w):
    """
    Use the softplus method to normalize weights.

    Parameters
    ----------
    w : numpy.ndarray
        The conditional or posterior weight matrix (n, n_q)

    Returns
    -------
    numpy.ndarray
        The softplus-normalized conditional or posterior weight matrix (n, n_q)
    """
    w_softplus = np.log(1 + np.exp(w))
    return w_softplus / np.sum(w_softplus, axis=0)


def clip_normalize(w):
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


def density_weights(w_q, y, theta_y):
    """
    Recover the weights on the probability density function from the embedding.

    .. note:: This is not working as well as expected for some reason.

    Parameters
    ----------
    w_q : numpy.ndarray
        The weight matrix or weight vector [(n, n_q), (n,)]
    y : numpy.ndarray
        The training outputs (n, d_y)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)

    Returns
    -------
    numpy.ndarray
        The normalized weight matrix or weight vector [(n, n_q), (n,)]
    """
    # Setup the optimization problem
    up_scale = _gaussian_norm(theta_y)
    a_down_scale = _gaussian_norm(np.sqrt(3) * theta_y)
    a_body = _gaussian(y, y, np.sqrt(3) * theta_y)
    b_down_scale = _gaussian_norm(np.sqrt(2) * theta_y)
    b_body = _gaussian(y, y, np.sqrt(2) * theta_y)
    a_matrix = (up_scale / a_down_scale) * a_body
    b_matrix = (up_scale / b_down_scale) * b_body
    a = a_matrix # (n, n)
    b = -np.dot(b_matrix, w_q) # [(n, n_q), (n,)]

    # Initialize the normalized weights
    w_q_pdf_init = clip_normalize(w_q)
    w_q_pdf = np.zeros(w_q_pdf_init.shape)

    # Find the normalized weights using snucq
    if w_q.ndim == 2:
        n, n_q = w_q.shape
        for i in np.arange(n_q):
            w_q_pdf[:, i] = _snucq(a, b[:, i], w_q_pdf_init[:, i])
    elif w_q.ndim == 1:
        w_q_pdf = _snucq(a, b, w_q_pdf_init)
    else:
        raise ValueError('Weights have the wrong dimensions')

    # Return the normalized weights
    return w_q_pdf


def _distribution_singular(y_eval, w_q_pdf, y, theta_y, clip=True):
    """
    Obtain the distribution function from normalized weights (non-vectorized).

    .. note:: This will be deleted once the vectorized version has been tested.

    Parameters
    ----------
    y_eval : numpy.ndarray
        The point to evaluate distribution (d_y,) [Not Vectorized]
    w_q_pdf : numpy.ndarray
        The normalized weight matrix or weight vector [(n, n_q), (n,)]
    y : numpy.ndarray
        The training outputs (n, d_y)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)

    Returns
    -------
    numpy.ndarray
        The distribution evaluated at y for each query point [(n_q,), 1]
    """
    # (n, d_y)
    all_cdf = 0.5 * (1 + _erf((y_eval - y) / (np.sqrt(2) * theta_y)))

    # (n,)
    each_cdf = np.prod(all_cdf, axis=1)

    # (n_q,) or constant
    cdf =  np.dot(each_cdf, w_q_pdf)

    # Clip the distribution if required
    return np.clip(cdf, 0, 1) if clip else cdf


def _distribution_vector(y_eval, w_q_pdf, y, theta_y, clip=True):
    """
    Obtain the distribution function from normalized weights (vectorized).

    Parameters
    ----------
    y_eval : numpy.ndarray
        The point to evaluate distribution (n_eval, d_y) [Vectorized]
    w_q_pdf : numpy.ndarray
        The normalized weight matrix or weight vector [(n, n_q), (n,)]
    y : numpy.ndarray
        The training outputs (n, d_y)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)

    Returns
    -------
    numpy.ndarray
        The distribution evaluated at y for each query point
        [(n_eval, n_q), (n_eval,)]
    """
    # (n_eval, n, d_y)
    y_dist = _dist(y_eval, y)

    # (n_eval, n, d_y)
    z = y_dist / (np.sqrt(2) * theta_y)

    # (n_eval, n, d_y)
    all_cdf = 0.5 * (1 + _erf(z))

    # (n_eval, n)
    each_cdf = np.prod(all_cdf, axis=-1)

    # (n_eval, n_q) or (n_eval,)
    cdf = np.dot(each_cdf, w_q_pdf)

    # Clip the distribution if required
    return np.clip(cdf, 0, 1) if clip else cdf


def distribution(y_eval, w_q_pdf, y, theta_y, clip=True):
    """
    Obtain the distribution function from normalized weights.

    Parameters
    ----------
    y_eval : numpy.ndarray
        The point to evaluate distribution [(n_eval, d_y), (d_y,)]
    w_q_pdf : numpy.ndarray
        The normalized weight matrix or weight vector [(n, n_q), (n,)]
    y : numpy.ndarray
        The training outputs (n, d_y)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)

    Returns
    -------
    numpy.ndarray
        The distribution evaluated at y for each query point
        [(n_q,), 1] or [(n_eval, n_q), (n_eval,)]
    """
    if y_eval.ndim == 1:
        return _distribution_singular(y_eval, w_q_pdf, y, theta_y, clip=clip)
    elif y_eval.ndim == 2:
        return _distribution_vector(y_eval, w_q_pdf, y, theta_y, clip=clip)
    else:
        raise ValueError('y_eval not in the right dimensions')


def quantile_regression(p, y_q_init, w_q_pdf, y, theta_y):
    """
    Perform quantile regression.

    Parameters
    ----------
    p : float
        The probability level of the quantile
    y_q_init : numpy.ndarray
        The initial quantile estimate for the first query point (d_y,)
    w_q_pdf : numpy.ndarray
        The conditional weights (n, n_q)
    y : numpy.ndarray
        The training outputs (n, d_y)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)

    Returns
    -------
    The quantiles at the query points (n_q, d_y)
    """
    # Initialise the quantile root finding routine
    n, n_q = w_q_pdf.shape
    y_q_opt = y_q_init.copy()
    y_quantiles = []

    # Go through each query point
    for j in np.arange(n_q):

        # Find the quantile for this query point
        def function(y_q):
            return _distribution_singular(y_q,
                                          w_q_pdf[:, j],y, theta_y) - p
        y_q_opt = _root(function, y_q_opt).x
        y_quantiles.append(y_q_opt)

    # (n_q, d_y)
    return np.array(y_quantiles)


def multiple_quantile_regression(probabilities, w_q_pdf, y, theta_y):
    """
    Perform multiple quantile regressions.

    Parameters
    ----------
    probabilities : array_like
        A list or array of probabilities to find the quantiles for
    w_q_pdf : numpy.ndarray
        The conditional weights (n, n_q)
    y : numpy.ndarray
        The training outputs (n, d_y)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (d_y,)

    Returns
    -------
    list
        A list of quantile curves, each of which are of size (n_q, d_y)
    """
    # Initialise the quantile at the mean
    y_q_init = y.mean()

    # Go through each probability
    y_quantiles = []
    for p in probabilities:

        # Find the quantile for this probability
        y_quantile = quantile_regression(p, y_q_init, w_q_pdf, y, theta_y)
        y_quantiles.append(y_quantile)

        # Start the next quantile search with the result of the current one
        y_q_init = y_quantile[0]

    # Return the quantiles
    return y_quantiles


def classify(p, classes=None):
    """
    Classify or predict based on a discrete probability distribution.

    Parameters
    ----------
    p : numpy.ndarray
        Discrete probability distribution of size (n, m)
    classes : numpy.ndarray
        The unique class labels of size (m,) where the default is [0, 1, ..., m]

    Returns
    -------
    numpy.ndarray
        The classification predictions of size (n,)
    """
    if classes is None:
        classes = np.arange(p.shape[1])
    return classes[np.argmax(p, axis=1)]


def entropy(p):
    """
    Compute the entropy of a discrete probability distribution.

    Parameters
    ----------
    p : numpy.ndarray
        Discrete probability distribution of size (n, m)

    Returns
    -------
    numpy.ndarray
        The entropy of size (n,)
    """
    p_copy = p.copy()
    p_copy[p_copy <= 0] = 1
    return -np.sum(p_copy * np.log(p_copy), axis=1)

# def expected_improvement(x_q, ):