"""
Benchmark Module.

Utility tools that deal with benchmarking or testing algorithmic performance.
"""
import numpy as np


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
        The output from the function of size (n, )
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

branin_hoo.x_opt = np.array([[-np.pi, 12.275],
                             [np.pi, 2.275],
                             [9.42478, 2.475]])
branin_hoo.f_opt = branin_hoo(branin_hoo.x_opt)

