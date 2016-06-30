"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.infer, bake.kernels
import matplotlib.pyplot as plt
from . import utils

def main():

    # Generate some data
    x = utils.generate_data(n = 30, d = 2, loc = 4, scale = 2, seed = 200)

    # Put uniform weights on the data
    w = bake.infer.uniform_weights(x)

    # Specify the hyperparameters of the kernel
    theta = 0.1

    # Compute the kernel embedding
    mu = bake.infer.embedding(w, x, bake.kernels.matern3on2, theta)

    # Generate some query points and evaluate the embedding at those points
    xq = np.linspace(-5.0, 5.0, 200)[:, np.newaxis]
    mu_xq = mu(xq)

    # Plot the query points
    plt.plot(xq.flatten(), mu_xq)
    plt.scatter(x.flatten(), np.zeros(x.shape[0]))
    plt.xlim((-5.0, 5.0))
    plt.show()

def generate_data(n = 10, d = 1, loc = 1, scale = 1, seed = 0):

    # Set seed
    np.random.seed(seed)

    # Generate some data
    # Note that data must come in 2D Arrays
    return scale * np.exp(np.random.rand(n, d))* np.random.randn(n, d) - \
             np.sin(loc * np.random.rand(n, d))

if __name__ == "__main__":
    main()