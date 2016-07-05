import numpy as np

def generate_two_waves(n = 80, noise_level = 0.2, seed = 100):

    n = 80
    x = 10 * np.random.rand(n, 1) - 5
    y1 = np.sin(x) + 1.0
    y2 = 1.2 * np.cos(x) - 2.0
    y = 0 * x
    ind = np.random.choice(np.arange(x.shape[0]), size = (2, int(n/2)), replace = False)
    y[ind[0]] = y1[ind[0]]
    y[ind[1]] = y2[ind[1]]
    y = y + noise_level * np.random.randn(*y.shape)
    return x, y

def joint_data(x, y):

	return np.vstack((x.ravel(), y.ravel())).T

def generate_gaussian_mixture(n = 10, d = 1, locs = [], scales = [], seed = None):

    # Set seed
    if seed:
        np.random.seed(seed)

    m = len(locs)
    assert m == len(scales)

    samples = []

    for i in range(n):
        samples.append(np.array(scales[i % m]) * np.random.randn(d) + np.array(locs[i % m]))

    return np.array(samples)