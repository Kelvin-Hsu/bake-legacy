"""
Kernel Function Module.

These are the definitions of commonly used characteristic kernels.
"""
import autograd.numpy as np
from scipy.spatial.distance import cdist

# TO DO: Add the following kernels
#   Rational Quadratic Kernel
#   Linear Kernel
#   Periodic Kernel
#   Locally Periodic Kernel
#   Matern Kernels
#   Chi-Squared Kernel


def gaussian(x_p, x_q, theta):
    """
    Defines the Gaussian or squared exponential kernel.

    The hyperparameters are the length scales of the kernel. There can either be
    m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    return np.exp(-0.5 * cdist(x_p/theta, x_q/theta, 'sqeuclidean'))


def laplace(x_p, x_q, theta):
    """
    Defines the Laplace or exponential kernel.

    The hyperparameters are the length scales of the kernel. There can either be
    m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    return np.exp(-cdist(x_p/theta, x_q/theta, 'euclidean'))


def matern3on2(x_p, x_q, theta):
    """
    Defines the Matern 3/2 kernel.

    The hyperparameters are the length scales of the kernel. There can either be
    m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    r = cdist(x_p/theta, x_q/theta, 'euclidean')
    return (1.0 + r) * np.exp(-r)


def kronecker_delta(x_p, x_q, *args):
    """
    Defines the Kronecker delta kernel.

    The Kronecker delta does not need any hyperparameters. Passing
    hyperparameter arguments do not change the kernel behaviour.

    Parameters
    ----------
    x_p : numpy.ndarray
        A collection of data (n_p x m) [2D Array]
    x_q : numpy.ndarray
        A collection of data (n_q x m) [2D Array]
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]

    Returns
    -------
    numpy.ndarray
        The corresponding gram matrix if "x_q" is given (n_p x n_q)
        The diagonal of the gram matrix if "x_q" is given as "None" (n_p)
    """
    if x_q is None:
        return np.ones(x_p.shape[0])
    return (cdist(x_q, x_q, 'sqeuclidean') == 0).astype(float)