"""
Linear Algebra Utility Module.

.. note:: Initially modified from revrand [http://github.com/nicta/revrand]
"""

import autograd.numpy as np
from scipy.linalg import cholesky, cho_solve, svd, LinAlgError


def dist(x_1, x_2):
    """
    Computes all the difference vectors between each pair of the given data.

    Parameters
    ----------
    x_1 : numpy.ndarray
        A collection of data points (n_1 x m) [2D Array]
    x_2 : numpy.ndarray
        A collection of data points (n_2 x m) [2D Array]
    Returns
    -------
    numpy.ndarray
        A collection of difference vectors (n_1 x n_2 x p) [3D Array]
    """
    # (n_1 x 1 x m) - (1 x n_2, m)
    return x_1[:, np.newaxis] - x_2[np.newaxis, :]


def gaussian_norm(theta, m = 1):
    """
    Obtains the normalising constant for the gaussian kernel.

    The hyperparameters are the length scales of the kernel. There can either be
    m of them for an anisotropic kernel or just 1 of them for an isotropic
    kernel. In the case of an isotropic kernel, the number of dimensions of the
    input variable must be given.

    Parameters
    ----------
    theta : numpy.ndarray
        Hyperparameters that parametrises the kernel [1D Array]
    m : int
        Dimensionality of the input
    Returns
    -------
    float
        The normalising constant
    """
    return np.prod(np.sqrt(2 * np.pi) * np.ones(m) * theta)


def log_gaussian_density(x, mu, sigma):

    b, log_det_sigma = solve_posdef(sigma, x - mu)
    const = x.shape[0] * np.log(2 * np.pi) # Can remove if needed
    return -0.5 * (np.dot(x - mu, b) + log_det_sigma + const)


def cho_log_det(L):
    """
    Compute the log of the determinant of :math:`A`, given its (upper or lower)
    Cholesky factorization :math:`LL^T`.
    Parameters
    ----------
    L: numpy.ndarray
        an upper or lower Cholesky factor
    Examples
    --------
    >>> A = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])
    >>> Lt = cholesky(A)
    >>> np.isclose(cho_log_det(Lt), np.log(np.linalg.det(A)))
    True
    >>> L = cholesky(A, lower=True)
    >>> np.isclose(cho_log_det(L), np.log(np.linalg.det(A)))
    True
    """
    return 2 * np.sum(np.log(L.diagonal()))


def svd_log_det(s):
    """
    Compute the log of the determinant of :math:`A`, given its singular values
    from an SVD factorisation (i.e. :code:`s` from :code:`U, s, Ut = svd(A)`).
    Parameters
    ----------
    s: numpy.ndarray
        the singular values from an SVD decomposition
    Examples
    --------
    >>> A = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])
    >>> _, s, _ = np.linalg.svd(A)
    >>> np.isclose(svd_log_det(s), np.log(np.linalg.det(A)))
    True
    """
    return np.sum(np.log(s))


def solve_posdef(A, b, cholthresh=1e-5):
    """
    Solve the system :math:`A X = b` for :math:`X` where :math:`A` is a
    positive semi-definite matrix.
    This first tries cholesky, and if numerically unstable with solve using a
    truncated SVD (see solve_posdef_svd).
    The log-determinant of :math:`A` is also returned since it requires minimal
    overhead.
    Parameters
    ----------
    A: numpy.ndarray
        A positive semi-definite matrix.
    b: numpy.ndarray
        An array or matrix
    Returns
    -------
    X: numpy.ndarray
        The result of :math:`X = A^-1 b`
    logdet: float
        The log-determinant of :math:`A`
    """

    # Try cholesky for speed
    try:
        lower = True
        L = cholesky(A, lower=lower, check_finite=False)
        if np.any(L.diagonal() < cholthresh):
            raise LinAlgError("Unstable cholesky factor detected")
        X = cho_solve((L, lower), b, check_finite=False)
        logdet = cho_log_det(L)

    # Failed cholesky, use svd to do the inverse
    except LinAlgError:

        U, s, V = svd(A)
        X = svd_solve(U, s, V, b)
        logdet = svd_log_det(s)

    return X, logdet


def svd_solve(U, s, V, b, s_tol=1e-15):
    """
    Solve the system :math:`A X = b` for :math:`X` where :math:`A` is a
    positive semi-definite matrix using the singular value decomposition.
    This truncates the SVD so only dimensions corresponding to non-negative and
    sufficiently large singular values are used.
    Parameters
    ----------
    U: numpy.ndarray
        The :code:`U` factor of :code:`U, s, V = svd(A)` positive
        semi-definite matrix.
    s: numpy.ndarray
        The :code:`s` factor of :code:`U, s, V = svd(A)` positive
        semi-definite matrix.
    V: numpy.ndarray
        The :code:`V` factor of :code:`U, s, V = svd(A)` positive
        semi-definite matrix.
    b: numpy.ndarray
        An array or matrix
    s_tol: float
        Cutoff for small singular values. Singular values smaller than
        :code:`s_tol` are clamped to :code:`s_tol`.
    Returns
    -------
    X: numpy.ndarray
        The result of :math:`X = A^-1 b`
    okind: numpy.ndarray
        The indices of :code:`s` that are kept in the factorisation
    """

    # Test shapes for efficient computations
    n = U.shape[0]
    assert(b.shape[0] == n)
    m = b.shape[1] if np.ndim(b) > 1 else 1

    # Auto clamp SVD based on threshold
    sclamp = np.maximum(s, s_tol)

    # Inversion factors
    ss = 1. / np.sqrt(sclamp)
    U2 = U * ss[np.newaxis, :]
    V2 = ss[:, np.newaxis] * V

    if m < n:
        # Few queries
        X = U2.dot(V2.dot(b))  # O(n^2 (2m))
    else:
        X = U2.dot(V2).dot(b)  # O(n^2 (m + n))

    return X
