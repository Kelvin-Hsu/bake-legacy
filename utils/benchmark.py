"""
Benchmark Module.

Utility tools that deal with benchmarking or testing algorithmic performance.
"""
import autograd.numpy as np
from scipy.spatial.distance import cdist


def loss_opt_loc(x=None, function=None, dist_ratio=0.01):

    dists = cdist(x, function.x_opt, 'euclidean')
    success_radius = dist_ratio * np.min(function.x_max - function.x_min)
    return np.min(dists/success_radius)


def success_opt_loc(loss_opt_loc_value):
    return loss_opt_loc_value < 1


def loss_func_approx(x_q, f_q, function=None):
    f_q_true = function(x_q)
    return np.sqrt(np.mean((f_q - f_q_true)**2))


def success_func_approx(loss_func_approx_value, rmse_bound=None):
    return loss_func_approx_value < rmse_bound


def branin_hoo(x):
    """
    Branin-Hoo Function.

    Parameters
    ----------
    x : numpy.ndarray
        The input to the function of size (n, 2)

    Returns
    -------
    numpy.ndarray
        The output from the function of size (n,)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return -f

branin_hoo.n_dim = 2
branin_hoo.x_min = np.array([-5., 0.])
branin_hoo.x_max = np.array([10., 15.])
branin_hoo.x_opt = np.array([[-np.pi, 12.275],
                             [np.pi, 2.275],
                             [9.42478, 2.475]])
branin_hoo.f_opt = branin_hoo(branin_hoo.x_opt)
branin_hoo.name = 'Branin Hoo'


def griewank(x):
    """
    Griewank Function.

    Parameters
    ----------
    x : numpy.ndarray
        The input to the function of size (n, d)

    Returns
    -------
    numpy.ndarray
        The output from the function of size (n,)
    """
    r = 4000
    s = np.sqrt(1 + np.arange(x.shape[1]))
    f = np.sum(x**2/r, axis=1) - np.prod(np.cos(x/s), axis=1) + 1
    return -f

griewank.n_dim = 2
griewank.x_min = np.array([-5., -5.])
griewank.x_max = np.array([5., 5.])
griewank.x_opt = np.array([[0., 0.]])
griewank.f_opt = griewank(griewank.x_opt)
griewank.name = 'Griewank'


def levy(x):
    """
    Levy Function.

    Parameters
    ----------
    x : numpy.ndarray
        The input to the function of size (n, d)

    Returns
    -------
    numpy.ndarray
        The output from the function of size (n,)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = np.sin(3*np.pi*x1)**2
    b = (x1 - 1)**2 * (1 + np.sin(3*np.pi*x2)**2)
    c = (x2 - 1)**2 * (1 + np.sin(2*np.pi*x2)**2)
    f = a + b + c
    return -f

levy.n_dim = 2
levy.x_min = np.array([-5., -5.])
levy.x_max = np.array([5., 5.])
levy.x_opt = np.array([[1., 1.]])
levy.f_opt = levy(levy.x_opt)
levy.name = 'Levy'


def schaffer(x):

    x1 = x[:, 0]
    x2 = x[:, 1]

    a = np.sin(x1**2 - x2**2)**2 - 0.5
    b = 1 + 0.001*(x1**2 + x2**2)
    f = 0.5 + a/(b**2)
    return -f

schaffer.n_dim = 2
schaffer.x_min = np.array([-2., -2.])
schaffer.x_max = np.array([2., 2.])
schaffer.x_opt = np.array([[0., 0.]])
schaffer.f_opt = schaffer(schaffer.x_opt)
schaffer.name = 'Schaffer'