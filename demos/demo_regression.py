"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.learn, bake.kernels
import matplotlib.pyplot as plt
import time

def main(place_prior = False):

    np.random.seed(100)

    # Generate some data
    # x = generate_data(n = 20, d = 1, loc = 4, scale = 2, seed = 200)
    n = 80
    x = 10 * np.random.rand(n, 1) - 5

    y1 = np.sin(x) + 1.0
    y2 = 1.2 * np.cos(x) - 2.0
    y = 0 * x
    ind = np.random.choice(np.arange(x.shape[0]), size = (2, 40), replace = False)
    y[ind[0]] = y1[ind[0]]
    y[ind[1]] = y2[ind[1]]
    # y = y + 0.2 * np.random.randn(*y.shape)
    y = y1
    z = np.vstack((x.ravel(), y.ravel())).T

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    x0 = np.linspace(-1, 1, 100)[:, np.newaxis]
    y0 = np.linspace(-1, 1, 100)[:, np.newaxis]
    w0 = bake.infer.uniform_weights(x0)


    theta_x_0 = np.array([1.0])
    mu_x_0 = bake.infer.embedding(w0, x0, bake.kernels.gaussian, theta_x_0)
    theta_y_0 = np.array([1.0])
    mu_y_0 = bake.infer.embedding(w0, y0, bake.kernels.gaussian, theta_y_0)

    if place_prior:

        t_min_tuple = tuple(list(np.ones((6, 1)) * 1e-1 * np.array([[0.5, 0.5, 0.5, 0.5, 0.00001, 0.00001]]).T))
        t_max_tuple = tuple(list(np.ones((6, 1)) * 1e1 * np.array([[5.0, 2.0, 10.0, 10.0, 1.0, 1.0]]).T))
        t_init_tuple = tuple(list(np.ones((6, 1)) * np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).T))
        theta_x_opt, theta_y_opt, psi_opt, sigma_opt, epsil_opt, delta_opt = bake.learn.posterior_embedding(mu_y_0, x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 5)

        # Infer conditional posterior embedding
        mu_y_optimal = bake.infer.embedding(w0, y0, bake.kernels.gaussian, theta_y_opt)
        mu_yx_optimal = bake.infer.posterior_embedding(mu_y_optimal, x, y, bake.kernels.gaussian, bake.kernels.gaussian, theta_x_opt, theta_y_opt, epsil_opt, delta_opt)

    else:

        t_min_tuple = tuple(list(np.ones((5, 1)) * 1e-1 * np.array([[0.5, 0.5, 0.5, 0.5, 0.00001]]).T))
        t_max_tuple = tuple(list(np.ones((5, 1)) * 1e1 * np.array([[5.0, 2.0, 10.0, 10.0, 1.0]]).T))
        t_init_tuple = tuple(list(np.ones((5, 1)) * np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T))
        theta_x_opt, theta_y_opt, psi_opt, sigma_opt, zeta_opt = bake.learn.conditional_embedding(x, y, t_min_tuple, t_max_tuple, t_init_tuple = None, n = 1000, repeat = 5)

        # Infer conditional posterior embedding
        mu_y_optimal = bake.infer.embedding(w0, y0, bake.kernels.gaussian, theta_y_opt)
        mu_yx_optimal = bake.infer.conditional_embedding(mu_y_optimal, x, y, bake.kernels.gaussian, bake.kernels.gaussian, theta_x_opt, theta_y_opt, zeta_opt)

    x_lim = np.max(np.abs(x)) + 1.0
    y_lim = np.max(np.abs(y)) + 1.0
    xq_array = np.linspace(-x_lim, x_lim, 150)
    yq_array = np.linspace(-y_lim, y_lim, 150)
    xv, yv = np.meshgrid(xq_array, yq_array)

    xq = xq_array[:, np.newaxis]
    yq = yq_array[:, np.newaxis]


    mu_yqxq_optimal = mu_yx_optimal(yq, xq)

    x_peaks, y_peaks = bake.infer.regressor_mode(mu_yqxq_optimal, xq_array, yq_array)

    plt.figure()
    plt.pcolormesh(xv, yv, mu_yqxq_optimal, label = 'Optimal Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.scatter(x_peaks, y_peaks, s = 1, edgecolor = 'face', c = 'w')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    if place_prior:
        plt.title('Optimal Posterior Embedding')
    else:
        plt.title('Optimal Conditional Embedding')

def generate_data_random(n = 10, d = 1, loc = 1, scale = 1, seed = None):

    # Set seed
    if seed:
        np.random.seed(seed)

    # Generate some data
    # Note that data must come in 2D Arrays
    return scale * np.exp(np.random.rand(n, d))* np.random.randn(n, d) - \
             np.sin(loc * np.random.rand(n, d))

def generate_data_gaussian_mixture(n = 10, d = 1, locs = [], scales = [], seed = None):

    # Set seed
    if seed:
        np.random.seed(seed)

    m = len(locs)
    assert m == len(scales)

    samples = []

    for i in range(n):
        samples.append(np.array(scales[i % m]) * np.random.randn(d) + np.array(locs[i % m]))
    return np.array(samples)

def time_module(module, *args, **kwargs):

    t_start = time.clock()
    output = module(*args, **kwargs)
    t_finish = time.clock()
    print('Module finished in: %f seconds' % (t_finish - t_start))
    return output

if __name__ == "__main__":
    time_module(main, place_prior = True)
    time_module(main, place_prior = False)
    plt.show()