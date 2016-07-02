"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.learn, bake.rlearn, bake.kernels
import matplotlib.pyplot as plt
import time

def main():

    np.random.seed(200)

    # Generate some data
    # x = generate_data(n = 20, d = 1, loc = 4, scale = 2, seed = 200)
    n = 100
    x = 10 * np.random.rand(n, 1) - 5

    y1 = np.sin(x) + 1.0
    y2 = 2 * np.cos(x) - 1.0
    # y3 = 5 * np.sin(x + 1)
    # y4 = 0.25 * x ** 2 - 3.0
    # y5 = -0.4 * (x + 1) ** 2 + 2.0

    y = 0 * x

    ind = np.random.choice(np.arange(x.shape[0]), size = (2, 50), replace = False)

    y[ind[0]] = y1[ind[0]]
    y[ind[1]] = y2[ind[1]]

    y = y + 0.2 * np.random.randn(*y.shape)

    # plt.scatter(x.flatten(), y.flatten())
    # plt.show()
    # return

    z = np.vstack((x.ravel(), y.ravel())).T

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    # Initialise the hyperparameters of the kernel
    theta_x_0 = np.array([0.4])
    mu_x_0 = bake.infer.embedding(w, x, bake.kernels.gaussian, theta_x_0)

    # Initialise the hyperparameters of the kernel
    theta_y_0 = np.array([0.1])
    mu_y_0 = bake.infer.embedding(w, y, bake.kernels.gaussian, theta_y_0)


    theta_z_0 = np.append(theta_x_0, theta_y_0)
    mu_z_0 = bake.infer.embedding(w, z, bake.kernels.gaussian, theta_z_0)

    # Learn the hyperparameters of the kernel
    theta_z, var_z, alpha_z = bake.learn.hyperparameters(z, theta_z_0, var_init = None, alpha_init = None)
    mu_z = bake.infer.embedding(w, z, bake.kernels.gaussian, theta_z)

    print('z: The learned length scale is: ', theta_z)
    print('z: The learned standard deviation is: ', np.sqrt(var_z))
    print('z: The learned measure length scale is: ', alpha_z)

    log_theta_z, log_var_z, log_alpha_z = bake.learn.log_hyperparameters(z, np.log(theta_z_0), log_var_init = None, log_alpha_init = None)
    mu_z_log = bake.infer.embedding(w, z, bake.kernels.gaussian, np.exp(log_theta_z))

    print('z: The log-learned length scale is: ', np.exp(log_theta_z))
    print('z: The log-learned standard deviation is: ', np.sqrt(np.exp(log_var_z)))
    print('z: The log-learned measure length scale is: ', np.exp(log_alpha_z))

    theta_x = theta_z[[0]]
    theta_y = theta_z[[1]]

    mu_x = bake.infer.embedding(w, x, bake.kernels.gaussian, theta_x)
    mu_y = bake.infer.embedding(w, y, bake.kernels.gaussian, theta_y)

    log_theta_x = log_theta_z[[0]]
    log_theta_y = log_theta_z[[1]]    

    mu_x_log = bake.infer.embedding(w, x, bake.kernels.gaussian, np.exp(log_theta_x))
    mu_y_log = bake.infer.embedding(w, y, bake.kernels.gaussian, np.exp(log_theta_y))

    epsil = 1e-10
    delta = 1e-8
    mu_yx = bake.infer.posterior_embedding(mu_y, x, y, bake.kernels.gaussian, bake.kernels.gaussian, theta_x, theta_y, epsil, delta)
    mu_yx_log = bake.infer.posterior_embedding(mu_y_log, x, y, bake.kernels.gaussian, bake.kernels.gaussian, np.exp(log_theta_x), np.exp(log_theta_y), epsil, delta)






    # Generate some query points and evaluate the embedding at those points
    xq = np.linspace(x.min() - 2.0, x.max() + 2.0, 1000)[:, np.newaxis]
    mu_xq_0 = mu_x_0(xq)
    mu_xq = mu_x(xq)
    mu_xq_log = mu_x_log(xq)

    # Plot the query points
    plt.figure(1)
    plt_initial_embedding = plt.plot(xq.flatten(), mu_xq_0, 'r', label = 'Initial Embedding')
    plt_learned_embedding = plt.plot(xq.flatten(), mu_xq, 'g', label = 'Learned Embedding')
    plt_log_learned_embedding = plt.plot(xq.flatten(), mu_xq_log, 'c', label = 'Log-Learned Embedding')
    plt_training_data = plt.scatter(x.flatten(), np.zeros(x.shape[0]), label = 'Training Data')
    plt.xlim((x.min() - 2.0, x.max() + 2.0))
    plt.xlabel('$x$')
    plt.ylabel('$\mu_{\mathbb{P}}(x)$')
    plt.title('Bayesian Learning of Kernel Embedding (x)')
    plt.legend()

    # Generate some query points and evaluate the embedding at those points
    yq = np.linspace(y.min() - 1.0, y.max() + 1.0, 1000)[:, np.newaxis]
    mu_yq_0 = mu_y_0(yq)
    mu_yq = mu_y(yq)
    mu_yq_log = mu_y_log(yq)

    # Plot the query points
    plt.figure(2)
    plt_initial_embedding = plt.plot(yq.flatten(), mu_yq_0, 'r', label = 'Initial Embedding')
    plt_learned_embedding = plt.plot(yq.flatten(), mu_yq, 'g', label = 'Learned Embedding')
    plt_log_learned_embedding = plt.plot(yq.flatten(), mu_yq_log, 'c', label = 'Log-Learned Embedding')
    plt_training_data = plt.scatter(y.flatten(), np.zeros(y.shape[0]), label = 'Training Data')
    plt.xlim((y.min() - 1.0, y.max() + 1.0))
    plt.xlabel('$y$')
    plt.ylabel('$\mu_{\mathbb{P}}(y)$')
    plt.title('Bayesian Learning of Kernel Embedding (y)')
    plt.legend()






    # Generate some query points and evaluate the embedding at those points
    x_lim = np.max(np.abs(x)) + 1.0
    y_lim = np.max(np.abs(y)) + 1.0
    xq_array = np.linspace(-x_lim, x_lim, 150)
    yq_array = np.linspace(-y_lim, y_lim, 150)
    xv, yv = np.meshgrid(xq_array, yq_array)

    xq = xq_array[:, np.newaxis]
    yq = yq_array[:, np.newaxis]

    zq = np.vstack((xv.ravel(), yv.ravel())).T

    mu_yqxq = mu_yx(yq, xq)
    mu_yqxq_log = mu_yx_log(yq, xq)

    mu_zq = mu_z(zq)
    mu_zq_log = mu_z_log(zq)

    plt.figure(3)

    plt.subplot(221)
    x_peaks, y_peaks = bake.infer.regressor_mode(mu_yqxq, xq_array, yq_array)
    plt.pcolormesh(xv, yv, mu_yqxq, label = 'Learned Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.scatter(x_peaks, y_peaks, s = 1, edgecolor = 'face', c = 'w')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Learned Conditional Embedding')

    plt.subplot(222)
    x_peaks, y_peaks = bake.infer.regressor_mode(mu_yqxq_log, xq_array, yq_array)
    plt.pcolormesh(xv, yv, mu_yqxq_log, label = 'Log-Learned Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.scatter(x_peaks, y_peaks, s = 1, edgecolor = 'face', c = 'w')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Log-Learned Conditional Embedding')

    plt.subplot(223)
    plt.scatter(zq[:, 0], zq[:, 1], s = 20, c = mu_zq, linewidths = 0, label = 'Learned Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Learned Joint Embedding')

    plt.subplot(224)
    plt.scatter(zq[:, 0], zq[:, 1], s = 20, c = mu_zq_log, linewidths = 0, label = 'Log-Learned Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Log-Learned Joint Embedding')


    ########

    theta_x_opt, var_opt, alpha_opt = bake.learn.optimal_hyperparameters(x, n = 25000)

    print('Optimal x theta: ', theta_x_opt)
    print('Optimal var: ', var_opt)
    print('Optimal alpha: ', alpha_opt)

    theta_y_opt, var_opt, alpha_opt = bake.learn.optimal_hyperparameters(y, n = 25000)

    print('Optimal y theta: ', theta_y_opt)
    print('Optimal var: ', var_opt)
    print('Optimal alpha: ', alpha_opt)

    # Infer conditional posterior embedding
    mu_y_optimal = bake.infer.embedding(w, y, bake.kernels.gaussian, theta_y_opt)
    mu_yx_optimal = bake.infer.posterior_embedding(mu_y_optimal, x, y, bake.kernels.gaussian, bake.kernels.gaussian, theta_x_opt, theta_y_opt, epsil, delta)
    mu_yqxq_optimal = mu_yx_optimal(yq, xq)

    x_peaks, y_peaks = bake.infer.regressor_mode(mu_yqxq_optimal, xq_array, yq_array)

    plt.figure(5)
    plt.subplot(211)
    plt.pcolormesh(xv, yv, mu_yqxq_optimal, label = 'Optimal Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.scatter(x_peaks, y_peaks, s = 1, edgecolor = 'face', c = 'w')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Optimal Conditional Embedding (Original Method)')



    theta_x_opt, theta_y_opt, epsil_opt, delta_opt, var_opt, alpha_opt = bake.rlearn.optimal_hyperparameters(mu_y_0, x, y, n = 100000)

    print('Optimal x theta: ', theta_x_opt)
    print('Optimal y theta: ', theta_y_opt)
    print('Optimal epsil: ', epsil_opt)
    print('Optimal delta: ', delta_opt)
    print('Optimal var: ', var_opt)
    print('Optimal alpha: ', alpha_opt)

    # Infer conditional posterior embedding
    mu_y_optimal = bake.infer.embedding(w, y, bake.kernels.gaussian, theta_y_opt)
    mu_yx_optimal = bake.infer.posterior_embedding(mu_y_optimal, x, y, bake.kernels.gaussian, bake.kernels.gaussian, theta_x_opt, theta_y_opt, epsil_opt, delta_opt)
    mu_yqxq_optimal = mu_yx_optimal(yq, xq)

    x_peaks, y_peaks = bake.infer.regressor_mode(mu_yqxq_optimal, xq_array, yq_array)

    plt.subplot(212)
    plt.pcolormesh(xv, yv, mu_yqxq_optimal, label = 'Optimal Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.scatter(x_peaks, y_peaks, s = 1, edgecolor = 'face', c = 'w')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Optimal Conditional Embedding (My Method)')

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

def time_module(module, args = ()):

    t_start = time.clock()
    output = module(*args)
    t_finish = time.clock()
    print(t_finish - t_start)
    return output

if __name__ == "__main__":
    time_module(main)
    plt.show()