"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.learn, bake.kernels
import matplotlib.pyplot as plt

def main():

    np.random.seed(200)

    # Generate some data
    # x = generate_data(n = 20, d = 1, loc = 4, scale = 2, seed = 200)
    n = 100
    x = 10 * np.random.rand(n, 1) - 5

    y1 = np.sin(x)
    y2 = 2 * np.cos(x)

    ind = np.random.choice(np.arange(x.shape[0]), size = int(n/2))
    print(ind)

    y = y1.copy()
    y[ind] = y2[ind]

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    # Initialise the hyperparameters of the kernel
    theta_x_0 = np.array([1.0])
    mu_x_0 = bake.infer.embedding(w, x, bake.kernels.gaussian, theta_x_0)

    # Learn the hyperparameters of the kernel
    theta_x, var_x, alpha_x = bake.learn.hyperparameters(x, theta_x_0, var_init = None, alpha_init = None)
    mu_x = bake.infer.embedding(w, x, bake.kernels.gaussian, theta_x)

    print('x: The learned length scale is: ', theta_x)
    print('x: The learned standard deviation is: ', np.sqrt(var_x))
    print('x: The learned measure length scale is: ', alpha_x)

    log_theta_x, log_var_x, log_alpha_x = bake.learn.log_hyperparameters(x, np.log(theta_x_0), log_var_init = None, log_alpha_init = None)
    mu_x_log = bake.infer.embedding(w, x, bake.kernels.gaussian, np.exp(log_theta_x))

    print('x: The log-learned length scale is: ', np.exp(log_theta_x))
    print('x: The log-learned standard deviation is: ', np.sqrt(np.exp(log_var_x)))
    print('x: The log-learned measure length scale is: ', np.exp(log_alpha_x))





    # Initialise the hyperparameters of the kernel
    theta_y_0 = np.array([0.4])
    mu_y_0 = bake.infer.embedding(w, y, bake.kernels.gaussian, theta_y_0)

    # Learn the hyperparameters of the kernel
    theta_y, var_y, alpha_y = bake.learn.hyperparameters(y, theta_y_0, var_init = None, alpha_init = None)
    mu_y = bake.infer.embedding(w, y, bake.kernels.gaussian, theta_y)

    print('y: The learned length scale is: ', theta_y)
    print('y: The learned standard deviation is: ', np.sqrt(var_y))
    print('y: The learned measure length scale is: ', alpha_y)

    log_theta_y, log_var_y, log_alpha_y = bake.learn.log_hyperparameters(y, np.log(theta_y_0), log_var_init = None, log_alpha_init = None)
    mu_y_log = bake.infer.embedding(w, y, bake.kernels.gaussian, np.exp(log_theta_y))

    print('y: The log-learned length scale is: ', np.exp(log_theta_y))
    print('y: The log-learned standard deviation is: ', np.sqrt(np.exp(log_var_y)))
    print('y: The log-learned measure length scale is: ', np.exp(log_alpha_y))

    epsil = 1e-10
    delta = 1e-8
    mu_yx = bake.infer.posterior_embedding(mu_y_log, x, y, bake.kernels.gaussian, bake.kernels.gaussian, np.exp(log_theta_x), np.exp(log_theta_y), epsil, delta)







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
    # Z = np.vstack((xv.ravel(), yv.ravel())).T
    # xq = Z[:, [0]]
    # yq = Z[:, [1]]
    xq = xq_array[:, np.newaxis]
    yq = yq_array[:, np.newaxis]
    mu_yqxq = mu_yx(yq, xq)

    print(xv.shape)
    plt.figure(3)
    plt.pcolormesh(xv, yv, mu_yqxq, label = 'Log-Learned Embedding')
    plt.scatter(x.flatten(), y.flatten(), c = 'k', label = 'Training Data')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Bayesian Kernel Embedding Regression')
    plt.show()


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


if __name__ == "__main__":
    main()