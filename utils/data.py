import numpy as np


def generate_two_waves(n=80, noise_level=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    x = 10 * np.random.rand(n, 1) - 5
    y1 = np.sin(x/3) + 1.0
    y2 = 1.2 * np.cos(x/3) - 2.0
    y = 0 * x
    ind = np.random.choice(np.arange(x.shape[0]), size=(2, int(n / 2)), replace=False)
    y[ind[0]] = y1[ind[0]]
    y[ind[1]] = y2[ind[1]]
    y += + noise_level * np.random.randn(*y.shape)
    return x, y


def generate_one_wave(n=80, noise_level=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    # Generate some data
    x = 10 * np.random.rand(n, 1) - 5
    y = np.sin(x) + 1.0
    y = y + noise_level * np.random.randn(*y.shape)
    return x, y


def joint_data(x, y):
    return np.vstack((x.ravel(), y.ravel())).T


def generate_gaussian_mixture(n=10, d=1, locs=[], scales=[], seed=None):
    # Set seed
    if seed:
        np.random.seed(seed)

    m = len(locs)
    assert m == len(scales)

    samples = []

    for i in range(n):
        samples.append(np.array(scales[i % m]) * np.random.randn(d) + np.array(locs[i % m]))

    return np.array(samples)


def regressor_mode(mu_yqxq, xq_array, yq_array):
    from scipy.signal import argrelextrema

    # Assume xq_array and yq_array are just 1D arrays for now

    n_y, n_x = mu_yqxq.shape

    assert n_x == xq_array.shape[0]
    assert n_y == yq_array.shape[0]

    x_peaks = np.array([])
    y_peaks = np.array([])

    for i in range(n_x):
        ind = argrelextrema(mu_yqxq[:, i], np.greater)
        for j in ind[0]:
            x_peaks = np.append(x_peaks, xq_array[i])
            y_peaks = np.append(y_peaks, yq_array[j])

    return x_peaks, y_peaks
