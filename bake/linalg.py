"""
Linear Algebra Utility Module.

.. note:: This is modified from revrand [http://github.com/nicta/revrand]
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve, svd, LinAlgError

# Numerical constants / thresholds
singvalthresh = 1. / np.finfo(float).eps
cholthresh = 1e-5

def cho_log_det(L):
    """
    Compute the log of the determinant of :math:`A`, given its (upper or lower)
    Cholesky factorization :math:`LL^T`.
    Parameters
    ----------
    L: ndarray
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
    s: ndarray
        The singular values from an SVD decomposition
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


def solve_posdef(A, b):
    """
    Solve the system :math:`A X = b` for :math:`X` where :math:`A` is a
    positive semi-definite matrix.
    This first tries cholesky, and if numerically unstable with solve using a
    truncated SVD (see solve_posdef_svd).
    The log-determinant of :math:`A` is also returned since it requires minimal
    overhead.
    Parameters
    ----------
    A: ndarray
        A positive semi-definite matrix.
    b: ndarray
        An array or matrix
    Returns
    -------
    X: ndarray
        The result of :math:`X = A^-1 b`
    logdet: float
        The log-determinant of :math:`A`
    """

    # Try cholesky for speed
    try:
        lower = False
        L = cholesky(A, lower=lower)
        if any(L.diagonal() < cholthresh):
            raise LinAlgError("Unstable cholesky factor detected")
        X = cho_solve((L, lower), b)
        logdet = cho_log_det(L)

    # Failed cholesky, use (truncated) svd to do the inverse
    except LinAlgError:
        U, s, _ = svd(A)
        X, okind = svd_solve(U, s, b)
        logdet = svd_log_det(s[okind])

    return X, logdet


def svd_solve(U, s, b):
    """
    Solve the system :math:`A X = b` for :math:`X` where :math:`A` is a
    positive semi-definite matrix using the singular value decomposition.
    This truncates the SVD so only dimensions corresponding to non-negative and
    sufficiently large singular values are used.
    Parameters
    ----------
    U: ndarray
        The :code:`U` factor of :code:`U, s, Ut = svd(A)` positive
        semi-definite matrix.
    s: ndarray
        The :code:`s` factor of :code:`U, s, Ut = svd(A)` positive
        semi-definite matrix.
    b: ndarray
        An array or matrix
    Returns
    -------
    X: ndarray
        The result of :math:`X = A^-1 b`
    okind: ndarray
        The indices of :code:`s` that are kept in the factorisation
    """

    # Test shapes for efficient computations
    n = U.shape[0]
    assert(b.shape[0] == n)
    m = b.shape[1] if np.ndim(b) > 1 else 1

    # Auto truncate SVD based on negative or zero singular values
    okind = np.where(s > 0)[0]
    if len(okind) == 0:
        raise LinAlgError("No positive singular values!")

    # Auto truncate the svd based on condition number and machine precision
    okind = okind[np.abs(s[okind].max() / s[okind]) < singvalthresh]

    # Inversion factors
    ss = 1. / np.sqrt(s[okind])
    U2 = U[:, okind] * ss[np.newaxis, :]

    if m < n:
        # Few queries
        X = U2.dot(U2.T.dot(b))  # O(n^2 (2m))
    else:
        X = U2.dot(U2.T).dot(b)  # O(n^2 (m + n))

    return X, okind