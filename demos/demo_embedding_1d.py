"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake
import utils
import matplotlib.pyplot as plt

def main():

    # Generate some data
    x = utils.data.generate_gaussian_mixture(n = 50, d = 1, 
        locs = [-6.0, 1.0, 4.0, 9.0], scales = [0.5, 3.0, 0.5, 1.5], seed = 200)

    # Initialise the hyperparameters of the kernel
    theta_init = np.array([0.1])
    mu_init = bake.infer.embedding(x, theta_init)

    # Learn the hyperparameters of the kernel
    hyper_min = ([0.1], [0.1], [0.1])
    hyper_max = ([2.], [2.], [2.])
    theta, psi, sigma = bake.learn.embedding(x, hyper_min, hyper_max, n = 500)
    mu = bake.infer.embedding(x, theta)
    print('The learned length scale is: ', theta)
    print('The learned measure length scale is: ', psi)
    print('The learned standard deviation is: ', sigma)

    # Generate some query points and evaluate the embedding at those points
    x_lim = (x.min() - 2.0, x.max() + 2.0)
    xq = np.linspace(*x_lim, 1000)[:, np.newaxis]
    mu_init_xq = mu_init(xq)
    mu_xq = mu(xq)

    # Find the modes of the probability density function
    x_modes = bake.infer.multiple_modes(mu, [x_lim[0]], [x_lim[1]], n_modes = 20)

    # Plot the query points
    plt.plot(xq.ravel(), mu_init_xq, 'r', label = 'Initial Embedding')
    plt.plot(xq.ravel(), mu_xq, 'g', label = 'Learned Embedding')
    plt.scatter(x.ravel(), np.zeros(x.shape[0]), label = 'Training Data')
    [plt.axvline(x = x_mode[0], linewidth = 2, color = 'k') for x_mode in x_modes]
    plt.xlim(x_lim)
    plt.xlabel('$x$')
    plt.ylabel('$\mu_{\mathbb{P}}(x)$')
    plt.title('Bayesian Learning of Kernel Embedding')
    plt.legend()

if __name__ == "__main__":
    main()
    plt.show()