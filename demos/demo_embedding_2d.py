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
        scales = [1.5, 4.0, 1.5, 2.5, 2.0], seed = 200)

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    # Initialise the hyperparameters of the kernel
    theta_init = np.array([0.4])
    mu_init = bake.infer.embedding(w, x, bake.kernels.gaussian, theta_init)

    # Learn the hyperparameters of the kernel
    theta_iso, psi_iso, sigma_iso = bake.learn.embedding(x, ([0.1], [0.1], [0.1]), ([2.], [2.], [2.]), t_init_tuple = None, n = 2000)
    mu_iso = bake.infer.embedding(w, x, bake.kernels.gaussian, theta_iso)

    print('The learned length scale is: ', theta_iso)
    print('The learned measure length scale is: ', psi_iso)
    print('The learned standard deviation is: ', sigma_iso)

    theta, psi, sigma = bake.learn.embedding(x, ([0.1, 0.1], [0.1, 0.1], [0.1]), ([2., 2.], [2., 2.], [2.]), t_init_tuple = None, n = 2000)
    mu = bake.infer.embedding(w, x, bake.kernels.gaussian, theta)

    print('The learned length scale is: ', theta)
    print('The learned measure length scale is: ', psi)
    print('The learned standard deviation is: ', sigma)

    # Generate some query points and evaluate the embedding at those points
    x_lim = np.max(np.abs(x)) + 1.0
    xq1 = np.linspace(-x_lim, x_lim, 150)
    xq2 = np.linspace(-x_lim, x_lim, 150)
    xv1, xv2 = np.meshgrid(xq1, xq2)
    xq = np.vstack((xv1.ravel(), xv2.ravel())).T

    x_modes_init = bake.infer.multiple_modes(mu_init, [-x_lim, -x_lim], [x_lim, x_lim], bake.kernels.gaussian, theta_init, n_modes = 150)
    x_modes_iso = bake.infer.multiple_modes(mu_iso, [-x_lim, -x_lim], [x_lim, x_lim], bake.kernels.gaussian, theta_iso, n_modes = 50)
    x_modes = bake.infer.multiple_modes(mu, [-x_lim, -x_lim], [x_lim, x_lim], bake.kernels.gaussian, theta, n_modes = 50)

    # Evaluate the embedding at query points
    mu_init_xq = mu_init(xq)
    mu_xq = mu(xq)
    mu_iso_xq = mu_iso(xq)

    # Plot the query points
    plt.figure(1)
    plt.scatter(xq[:, 0], xq[:, 1], s = 20, c = mu_init_xq, linewidths = 0, label = 'Initial Embedding')
    plt_training_data = plt.scatter(x[:, 0], x[:, 1], c = 'k', label = 'Training Data')
    plt.scatter(x_modes_init[:, 0], x_modes_init[:, 1], s = 20, c = 'w', marker = 'o', label = 'Modes')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Initial Embedding')

    plt.figure(2)
    plt.scatter(xq[:, 0], xq[:, 1], s = 20, c = mu_iso_xq, linewidths = 0, label = 'Learned Embedding')
    plt.scatter(x[:, 0], x[:, 1], c = 'k', label = 'Training Data')
    plt.scatter(x_modes_iso[:, 0], x_modes_iso[:, 1], s = 20, c = 'w', marker = 'o', label = 'Modes')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Learned Isotropic Embedding')

    plt.figure(3)
    plt.scatter(xq[:, 0], xq[:, 1], s = 20, c = mu_xq, linewidths = 0, label = 'Learned Embedding')
    plt.scatter(x[:, 0], x[:, 1], c = 'k', label = 'Training Data')
    plt.scatter(x_modes[:, 0], x_modes[:, 1], s = 20, c = 'w', marker = 'o', label = 'Modes')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Learned Anisotropic Embedding')

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