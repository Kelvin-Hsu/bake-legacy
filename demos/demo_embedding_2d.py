"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.learn, bake.kernels
import matplotlib.pyplot as plt

def main():

    # Generate some data
    # x = generate_data(n = 20, d = 1, loc = 4, scale = 2, seed = 200)
    x = generate_data_gaussian_mixture(n = 50, d = 2, 
        locs = [[-6.0, 6.0], [1.0, 3.0], [4.0, -4.0], [-9.0, 2.0], [-5.0, -10.0]], 
        scales = [0.5, 3.0, 0.5, 1.5, 1.0])

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    # Initialise the hyperparameters of the kernel
    theta0 = np.array([0.5])
    mu0 = bake.infer.embedding(w, x, bake.kernels.gaussian, theta0)

    var_init = None
    alpha_init = None
    log_var_init = None if var_init is None else np.log(var_init)
    log_alpha_init = None if alpha_init is None else np.log(alpha_init)

    # Learn the hyperparameters of the kernel
    theta, var, alpha = bake.learn.hyperparameters(x, theta0, var_init = var_init, alpha_init = alpha_init)
    mu = bake.infer.embedding(w, x, bake.kernels.gaussian, theta)

    print('The learned length scale is: ', theta)
    print('The learned standard deviation is: ', np.sqrt(var))
    print('The learned measure length scale is: ', alpha)

    log_theta, log_var, log_alpha = bake.learn.log_hyperparameters(x, np.log(theta0), log_var_init = log_var_init, log_alpha_init = log_alpha_init)
    mu_log = bake.infer.embedding(w, x, bake.kernels.gaussian, np.exp(log_theta))

    print('The log-learned length scale is: ', np.exp(log_theta))
    print('The log-learned standard deviation is: ', np.sqrt(np.exp(log_var)))
    print('The log-learned measure length scale is: ', np.exp(log_alpha))

    # Generate some query points and evaluate the embedding at those points
    x_lim = np.max(np.abs(x)) + 1.0
    xq1 = np.linspace(-x_lim, x_lim, 150)
    xq2 = np.linspace(-x_lim, x_lim, 150)
    xv1, xv2 = np.meshgrid(xq1, xq2)
    xq = np.vstack((xv1.ravel(), xv2.ravel())).T

    # Evaluate the embedding at query points
    mu0_xq = mu0(xq)
    mu_xq = mu(xq)
    mu_log_xq = mu_log(xq)

    # Plot the query points
    plt.figure(1)
    plt.scatter(xq[:, 0], xq[:, 1], s = 20, c = mu0_xq, linewidths = 0, label = 'Initial Embedding')
    plt_training_data = plt.scatter(x[:, 0], x[:, 1], c = 'k', label = 'Training Data')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Initial Embedding')

    plt.figure(2)
    plt.scatter(xq[:, 0], xq[:, 1], s = 20, c = mu_xq, linewidths = 0, label = 'Learned Embedding')
    plt_training_data = plt.scatter(x[:, 0], x[:, 1], c = 'k', label = 'Training Data')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Learned Embedding')

    plt.figure(3)
    plt.scatter(xq[:, 0], xq[:, 1], s = 20, c = mu_log_xq, linewidths = 0, label = 'Log-Learned Embedding')
    plt_training_data = plt.scatter(x[:, 0], x[:, 1], c = 'k', label = 'Training Data')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Log-Learned Embedding')

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