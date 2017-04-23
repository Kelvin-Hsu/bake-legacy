import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from experiments import run_experiment


def load_iris_data(normalize_features=True):
    """
    Load the iris dataset with two feature dimensions.

    Parameters
    ----------
    normalize_features : bool, optional
        To normalize the features or not

    Returns
    -------
    numpy.ndarray
        The features (n, d)
    numpy.ndarray
        The target labels (n, 1)

    """
    iris = datasets.load_iris()
    x = iris.data
    if normalize_features:
        x -= np.min(x, axis=0)
        x /= np.max(x, axis=0)
    y = iris.target
    return x, y[:, np.newaxis]

test_size = 0.1

x, y = load_iris_data()
np.random.seed(0)
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=test_size, random_state=0)

run_experiment(x_train, y_train, x_test, y_test,
               name='iris_isotropic_fgd',
               s_init=1.,
               l_init=np.array([1.]),
               zeta_init=1e-4,
               learning_rate=0.01,
               grad_tol=0.1,
               max_iter=1000,
               n_sgd_batch=None)

run_experiment(x_train, y_train, x_test, y_test,
               name='iris_isotropic_sgd_75',
               s_init=1.,
               l_init=np.array([1.]),
               zeta_init=1e-4,
               learning_rate=0.01,
               grad_tol=0.1,
               max_iter=1000,
               n_sgd_batch=75)

run_experiment(x_train, y_train, x_test, y_test,
               name='iris_anisotropic_fgd',
               s_init=1.,
               l_init=np.ones(x_train.shape[1]),
               zeta_init=1e-4,
               learning_rate=0.01,
               grad_tol=0.1,
               max_iter=1000,
               n_sgd_batch=None)

run_experiment(x_train, y_train, x_test, y_test,
               name='iris_anistropic_sgd_75',
               s_init=1.,
               l_init=np.ones(x_train.shape[1]),
               zeta_init=1e-4,
               learning_rate=0.01,
               grad_tol=0.1,
               max_iter=1000,
               n_sgd_batch=75)
