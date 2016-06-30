"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake.inference, bake.kernels
import matplotlib.pyplot as plt

def main():

    # Generate some data
    x = generate_data(seed = 200)

    # Put uniform weights on the data
    w = bake.inference.uniform_weights(x)

    # Specify the hyperparameters of the kernel
    theta = 0.1

    # Compute the kernel embedding
    mu = bake.inference.embedding(w, x, bake.kernels.matern3on2, theta)

    # Generate some query points and evaluate the embedding at those points
    xq = np.linspace(-5.0, 5.0, 200)[:, np.newaxis]
    mu_xq = mu(xq)

    # Plot the query points
    plt.plot(xq.flatten(), mu_xq)
    plt.scatter(x.flatten(), np.zeros(x.shape[0]))
    plt.xlim((-5.0, 5.0))
    plt.show()

def generate_data(seed = 0):

    # Set seed
    np.random.seed(seed)

    # Generate some data
    # Note that data must come in 2D Arrays
    x1 = 0.5 * np.random.randn(10) - 3.0
    x2 = 0.8 * np.random.randn(10) + 2.0
    x3 = 0.25 * np.random.randn(10) - 1.0
    return np.append(np.append(x1, x2), x3)[:, np.newaxis]

if __name__ == "__main__":
    main()