"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.learn, bake.kernels
import matplotlib.pyplot as plt

def main():

    # Generate some data
    # x = generate_data(n = 20, d = 1, loc = 4, scale = 2, seed = 200)
    x = generate_data_gaussian_mixture(n = 50, d = 1, locs = [-6.0, 1.0, 4.0, 9.0], scales = [0.5, 3.0, 0.5, 1.5], seed = 200)

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    # Initialise the hyperparameters of the kernel
    theta0 = np.array([0.1])
    psi0 = np.array([0.1])
    sigma0 = np.array([0.1])
    mu0 = bake.infer.embedding(w, x, bake.kernels.gaussian, theta0)

    # Learn the hyperparameters of the kernel
    theta, psi, sigma = bake.learn.learn_joint_embedding(x, ([0.1], [0.1], [0.1]), ([2.], [2.], [2.]), t_init_tuple = None, n = 2000)
    mu = bake.infer.embedding(w, x, bake.kernels.gaussian, theta)

    print('The learned length scale is: ', theta)
    print('The learned measure length scale is: ', psi)
    print('The learned standard deviation is: ', sigma)

    # Generate some query points and evaluate the embedding at those points
    xq = np.linspace(x.min() - 2.0, x.max() + 2.0, 1000)[:, np.newaxis]
    mu0_xq = mu0(xq)
    mu_xq = mu(xq)

    # Plot the query points
    plt_initial_embedding = plt.plot(xq.flatten(), mu0_xq, 'r', label = 'Initial Embedding')
    plt_learned_embedding = plt.plot(xq.flatten(), mu_xq, 'g', label = 'Learned Embedding')
    plt_training_data = plt.scatter(x.flatten(), np.zeros(x.shape[0]), label = 'Training Data')
    plt.xlim((x.min() - 2.0, x.max() + 2.0))
    plt.xlabel('$x$')
    plt.ylabel('$\mu_{\mathbb{P}}(x)$')
    plt.title('Bayesian Learning of Kernel Embedding')
    plt.legend()
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