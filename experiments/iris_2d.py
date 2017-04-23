import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from experiments import run_experiment


def load_iris_data(a, b, normalize_features=True):
    """
    Load the iris dataset with two feature dimensions.

    Parameters
    ----------
    a : int
        First feature dimension
    b : int
        Second feature dimension
    normalize_features : bool, optional
        To normalize the features or not

    Returns
    -------
    numpy.ndarray
        The features (n, 2)
    numpy.ndarray
        The target labels (n, 1)

    """
    iris = datasets.load_iris()
    x = iris.data[:, [a, b]]
    if normalize_features:
        x -= np.min(x, axis=0)
        x /= np.max(x, axis=0)
    y = iris.target
    return x, y[:, np.newaxis]

n_attr = 4
test_size = 0.1

for a in np.arange(n_attr):
    for b in np.arange(a + 1, n_attr):

        x, y = load_iris_data(a, b)
        np.random.seed(0)
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=test_size, random_state=0)

        run_experiment(x_train, y_train, x_test, y_test,
                       name='iris%d%d_fgd' % (a, b),
                       s_init=1.,
                       l_init=np.array([1.]),
                       zeta_init=1e-4,
                       learning_rate=0.01,
                       grad_tol=0.1,
                       max_iter=1000,
                       n_sgd_batch=None)

        run_experiment(x_train, y_train, x_test, y_test,
                       name='iris%d%d_sgd_25' % (a, b),
                       s_init=1.,
                       l_init=np.array([1.]),
                       zeta_init=1e-4,
                       learning_rate=0.01,
                       grad_tol=0.1,
                       max_iter=1000,
                       n_sgd_batch=50)