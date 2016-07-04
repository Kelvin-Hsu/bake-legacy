"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.learn, bake.kernels
import matplotlib.pyplot as plt
import time

def main():

    np.random.seed(100)

    # Generate some data
    n = 80
    x = 10 * np.random.rand(n, 1) - 5
    y1 = np.sin(x) + 1.0
    y2 = 1.2 * np.cos(x) - 2.0
    y = 0 * x
    ind = np.random.choice(np.arange(x.shape[0]), size = (2, 40), replace = False)
    y[ind[0]] = y1[ind[0]]
    y[ind[1]] = y2[ind[1]]
    y = y + 0.2 * np.random.randn(*y.shape)
    z = np.vstack((x.ravel(), y.ravel())).T

    w = bake.infer.uniform_weights(z)

    theta_x, theta_y, psi, sigma, zeta = bake.learn.conditional_embedding(x, y, ([1.0], [0.1], [0.1], [1e-8], [1e-10]), ([10.], [2.], [5.], [1.], [1e-5]), t_init_tuple = None, n = 2000)
    theta = np.append(theta_x, theta_y)

    mu_z_optimal = bake.infer.embedding(w, z, bake.kernels.gaussian, theta)
    mu_yx_optimal = bake.infer.conditional_embedding(x, y, bake.kernels.gaussian, bake.kernels.gaussian, theta_x, theta_y, zeta)

    x_lim = np.max(np.abs(x))
    y_lim = np.max(np.abs(y)) + 1.0
    xq_array = np.linspace(-x_lim, x_lim, 150)
    yq_array = np.linspace(-y_lim, y_lim, 150)
    xv, yv = np.meshgrid(xq_array, yq_array)

    xq = xq_array[:, np.newaxis]
    yq = yq_array[:, np.newaxis]

    zq = np.vstack((xv.ravel(), yv.ravel())).T

    mu_zq_optimal = mu_z_optimal(zq)
    mu_yqxq_optimal = mu_yx_optimal(yq, xq)

    yv_min = np.array([-y_lim])
    yv_max = np.array([+y_lim])

    n_modes = 10
    y_modes = bake.infer.conditional_modes(mu_yx_optimal, xq, yv_min, yv_max, bake.kernels.gaussian, theta_y, n_modes = n_modes)
    x_peaks, y_peaks = bake.infer.regressor_mode(mu_yqxq_optimal, xq_array, yq_array)

    plt.figure(1)
    plt.scatter(xv.ravel(), yv.ravel(), s = 20, c = mu_zq_optimal, linewidths = 0, label = 'Learned Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Optimal Joint Embedding')

    plt.figure(2)
    plt.pcolormesh(xv, yv, mu_yqxq_optimal, label = 'Optimal Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.scatter(x_peaks, y_peaks, s = 1, edgecolor = 'face', c = 'w')
    plt.scatter(xv[:n_modes, :].ravel(), y_modes.ravel(), s = 20, c = 'w')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Optimal Conditional Embedding')

def time_module(module, *args, **kwargs):

    t_start = time.clock()
    output = module(*args, **kwargs)
    t_finish = time.clock()
    print('Module finished in: %f seconds' % (t_finish - t_start))
    return output

if __name__ == "__main__":
    time_module(main)
    plt.show()