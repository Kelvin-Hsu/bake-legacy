"""
Bayesian Inference for Kernel Embeddings Module.

In the inference module, kernel functions are by default the Gaussian kernel
unless otherwise specified through a keyword argument, since the learning that
is implemented in the learning module is done with the Gaussian kernel only for
the time being.
"""
import numpy as np
from .kernels import gaussian
from .linalg import solve_posdef
from .kbr import posterior_fields
from .optimize import local_optimisation


def uniform_weights(x):
    """
    Obtain uniform weights on a given dataset.

    Parameters
    ----------
    x : numpy.ndarray
        The dataset to place the uniform weights on (n x m)

    Returns
    -------
    numpy.ndarray
        Weights on each data point. Default weights are uniform weights (n)
    """
    return np.ones((x.shape[0], 1)) / x.shape[0]


def embedding(x, theta, w=None, k=gaussian):
    """
    Obtain empirical embedding on a given dataset.

    Parameters
    ----------
    x : numpy.ndarray
        The dataset to be used for generating the embedding (n x m)
    theta : numpy.ndarray
        Hyperparameters of the kernel (m)
    w : numpy.ndarray, optional
        Weights on each data point. Default weights are uniform weights (n)
    k : function
        Kernel functions from the kernels module. Default is a Gaussian kernel

    Returns
    -------
    function
        The empirical embedding with feature maps on each data point.
    """
    w = uniform_weights(x) if w is None else w
    return lambda xq: np.dot(k(xq, x, theta), w)


def conditional_weights(x, y, theta_x, theta_y, xq,
                        zeta=0, k_x=gaussian, k_xx=None):
    """
    Obtain the empirical conditional embedding on a given dataset.

    Here y is the target output to be modelled and x is the input covariate.

    Parameters
    ----------
    x : numpy.ndarray
        The collected dataset of the input covariates (n x n_x_dims)
    y : numpy.ndarray
        The collected dataset of the output targets (n x n_y_dims)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (n_x_dims)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (n_y_dims)
    xq : numpy.ndarray
        The query inputs (n_query x n_x_dims)
    zeta : float
        The regularisation parameter, interpreted as the likelihood noise level
    k_x : function, optional
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
    return solve_posdef(k_xx_reg, k_x(x, xq, theta_x))[0]


def conditional_embedding(x, y, theta_x, theta_y,
                          zeta=0, k_x=gaussian, k_y=gaussian, k_xx=None):
    """
    Obtain the empirical conditional embedding on a given dataset.

    Here y is the target output to be modelled and x is the input covariate.

    Parameters
    ----------
    x : numpy.ndarray
        The collected dataset of the input covariates (n x n_x_dims)
    y : numpy.ndarray
        The collected dataset of the output targets (n x n_y_dims)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (n_x_dims)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (n_y_dims)
    zeta : float
        The regularisation parameter, interpreted as the likelihood noise level
    k_x : function, optional
        The kernel on x [Default is the Gaussian kernel]
    k_y : function, optional
        The kernel on y [Default is the Gaussian kernel]
    k_xx : numpy.ndarray, optional
        The gramix on x cached beforehand to avoid recomputation

    Returns
    -------
    function
        The conditional embedding of Y | X, to be evaluated at (y, x)
    """
    w_func = lambda xq: conditional_weights(x, y, theta_x, theta_y, xq,
                                            zeta=zeta, k_x=k_x, k_xx=k_xx)
    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), w_func(xq))


def posterior_weights(prior_embedding, x, y, theta_x, theta_y, xq,
                      epsil=0, delta=0,
                      kbr='tikhonov',
                      k_x=gaussian, k_y=gaussian,
                      k_xx=None, k_yy=None,
                      v=None):
    """
    Obtain the empirical posterior embedding using KBR on a given dataset.

    Parameters
    ----------
    prior_embedding : function
        The prior embedding on y
    x : numpy.ndarray
        The collected dataset of the input covariates (n)
    y : numpy.ndarray
        The collected dataset of the output targets (n)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (m)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (m)
    epsil : float, optional
        The regularisation parameter for the prior
        This is not needed if v is already supplied
    delta : float, optional
        The regularisation parameter for the likelihood
        This is not needed if v is already supplied
    kbr : str
        A string denoting the type of Kernels Bayes Rule to be applied
        This is not needed if v is already supplied
    k_x : function, optional
        The kernel on x [Default is the Gaussian kernel]
    k_y : function, optional
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
    function
        The posterior embedding of Y | X, to be evaluated at (y, x)
    """
    if v is None:
        k_xx = k_x(x, x, theta_x) if not k_xx else k_xx
        k_yy = k_y(y, y, theta_y) if not k_yy else k_yy
        v = posterior_fields[kbr](prior_embedding(y), k_xx, k_yy, epsil, delta)
    return np.dot(v, k_x(x, xq, theta_x))


def posterior_embedding(prior_embedding, x, y, theta_x, theta_y,
                        epsil=0, delta=0,
                        kbr='tikhonov',
                        k_x=gaussian, k_y=gaussian,
                        k_xx=None, k_yy=None,
                        v=None):
    """
    Obtain the empirical posterior embedding using KBR on a given dataset.

    Parameters
    ----------
    prior_embedding : function
        The prior embedding on y
    x : numpy.ndarray
        The collected dataset of the input covariates (n)
    y : numpy.ndarray
        The collected dataset of the output targets (n)
    theta_x : numpy.ndarray
        Hyperparameters that parametrises the kernel on x (m)
    theta_y : numpy.ndarray
        Hyperparameters that parametrises the kernel on y (m)
    epsil : float, optional
        The regularisation parameter for the prior
        This is not needed if v is already supplied
    delta : float, optional
        The regularisation parameter for the likelihood
        This is not needed if v is already supplied
    kbr : str
        A string denoting the type of Kernels Bayes Rule to be applied
        This is not needed if v is already supplied
    k_x : function, optional
        The kernel on x [Default is the Gaussian kernel]
    k_y : function, optional
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
    function
        The posterior embedding of Y | X, to be evaluated at (y, x)
    """
    w_func = lambda xq: posterior_weights(prior_embedding, x, y,
                                          theta_x, theta_y, xq,
                                          epsil=epsil, delta=delta,
                                          kbr=kbr,
                                          k_x=k_x, k_y=k_y,
                                          k_xx=k_xx, k_yy=k_yy,
                                          v=v)
    return lambda yq, xq: np.dot(k_y(yq, y, theta_y), w_func(xq))


def mode(mu, xv_start, xv_min, xv_max):
    """
    Determine a density mode (peak), given its kernel embedding representation.

    Note that for stationary kernels, the peak of the density is located at the
    same places as the peak of the kernel embedding.

    Parameters
    ----------
    mu : function
        The kernel embedding
    xv_start : numpy.ndarray
        The starting location for which mode searching begins (n_dims)
    xv_min : numpy.ndarray
        The lower bound of the rectangular search region (n_dims)
    xv_max : numpy.ndarray
        The upper bound of the rectangular search region (n_dims)

    Returns
    -------
    numpy.ndarray
        The mode location (n_dims)
    """
    # Define the objective to be minimised
    # For stationary kernels, the objective to optimise can be reduced to simply
    # the embedding
    def objective(xvq):
        return -mu(np.array([xvq]))[0][0]

    # Find the mode
    x_mode, f_mode = local_optimisation(objective, xv_min, xv_max, xv_start)

    # Size: (n_dims)
    return x_mode


def multiple_modes(mu, xv_min, xv_max, n_modes = 10):
    """
    Determine density modes (peaks), given its kernel embedding representation.

    Note that for stationary kernels, the peak of the density is located at the
    same places as the peak of the kernel embedding.

    Parameters
    ----------
    mu : function
        The kernel embedding
    xv_min : numpy.ndarray
        The lower bound of the rectangular search region (n_dims)
    xv_max : numpy.ndarray
        The upper bound of the rectangular search region (n_dims)
    n_modes : int, optional
        The number of modes to search for (some of them will converge together)

    Returns
    -------
    numpy.ndarray
        The mode locations (n_modes x n_dims)
    """
    # Make sure these are arrays
    xv_min = np.array(xv_min)
    xv_max = np.array(xv_max)

    # Generate a list of starting points
    n_dims, = xv_min.shape
    standard_range = np.random.rand(n_modes, n_dims)
    xv_start_list = (xv_max - xv_min) * standard_range + xv_min

    # Compute the modes
    # Size: (n_modes x n_dims)
    return np.array([mode(mu, xv_start, xv_min, xv_max)
                     for xv_start in xv_start_list])


def conditional_modes(mu_yx, xq, yv_min, yv_max, n_modes = 10):
    """
    Determine a conditional density mode (peak), given its kernel embedding.

    Note that for stationary kernels, the peak of the density is located at the
    same places as the peak of the kernel embedding.

    Parameters
    ----------
    mu : function
        The kernel embedding
    xq : numpy.ndarray
        The query points (n_query, n_x_dims)
    yv_min : numpy.ndarray
        The lower bound of the rectangular search region (n_y_dims)
    yv_max : numpy.ndarray
        The upper bound of the rectangular search region (n_y_dims)
    n_modes : int, optional
        The number of modes to search for at each query point

    Returns
    -------
    numpy.ndarray
        The x coordinates of the mode locations (n_query x n_modes x n_x_dims)
    numpy.ndarray
        The y coordinates of the mode locations (n_query x n_modes x n_y_dims)
    """
    # This finds the modes of a given embedding
    modes_of = lambda mu: multiple_modes(mu, yv_min, yv_max, n_modes = n_modes)

    # This finds the modes at the embedding conditioned at x
    modes_at = lambda x: modes_of(lambda yq: mu_yx(yq, x))

    # This computes the modes at all query points
    # Size: (n_query x n_modes x n_dims)
    y_modes = np.array([modes_at(x) for x in xq[:, np.newaxis]])

    # This returns the corresponding input coordinates for each mode
    # Size: (n_query x n_modes x n_dims)
    x_modes = np.repeat(xq[:, np.newaxis], n_modes, axis = 1)

    # Return the modes
    # Size: (n_query x n_modes x n_x_dims)
    # Size: (n_query x n_modes x n_y_dims)
    return x_modes, y_modes

# def kernels_bayes_average():
#
