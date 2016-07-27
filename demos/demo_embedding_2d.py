"""
Demonstration of simple kernel embeddings.
"""
import numpy as np
import bake
import utils
import matplotlib.pyplot as plt


def main():

    # Generate some data
    seed = 200
    n_each = 10
    d = 2
    locs = [[-6., 6.], [1., 3.], [4., -4.], [-9., 2.], [-5., -10.]]
    scales = [[1.5, 1.5], [4.0, 4.0], [1.5, 1.5], [2.5, 2.5], [2.0, 2.0]]
    x = utils.data.generate_multiple_gaussian(n_each, d, locs, scales,
                                              seed=seed)

    # Initialise an embedding
    theta_init = np.array([0.4])
    mu_init = bake.infer.embedding(x, theta_init)

    # Learn the embedding with an isotropic kernel
    hyper_min = ([0.1], [0.1], [0.1])
    hyper_max = ([2.], [2.], [2.])
    theta_iso, psi_iso, sigma_iso = bake.learn.optimal_joint_embedding(x,
                                                                       hyper_min,
                                                                       hyper_max)
    mu_iso = bake.infer.embedding(x, theta_iso)
    print('The learned length scale is: ', theta_iso)
    print('The learned measure length scale is: ', psi_iso)
    print('The learned standard deviation is: ', sigma_iso)

    # Learn the embedding with an anisotropic kernel
    hyper_min = ([0.1, 0.1], [0.1, 0.1], [0.1])
    hyper_max = ([2., 2.], [2., 2.], [2.])
    theta, psi, sigma = bake.learn.optimal_joint_embedding(x,
                                                           hyper_min, hyper_max)
    mu = bake.infer.embedding(x, theta)
    print('The learned length scale is: ', theta)
    print('The learned measure length scale is: ', psi)
    print('The learned standard deviation is: ', sigma)

    # Generate some query points and evaluate the embedding at those points
    x_lim = np.max(np.abs(x)) + 1.0
    xq1 = np.linspace(-x_lim, x_lim, 150)
    xq2 = np.linspace(-x_lim, x_lim, 150)
    xv1, xv2 = np.meshgrid(xq1, xq2)
    xq = np.vstack((xv1.ravel(), xv2.ravel())).T

    x_modes_init = bake.infer.cleaned_multiple_modes(mu_init,
                                                     [-x_lim, -x_lim],
                                                     [x_lim, x_lim],
                                                     n_modes=150)
    x_modes_iso = bake.infer.cleaned_multiple_modes(mu_iso,
                                                    [-x_lim, -x_lim],
                                                    [x_lim, x_lim],
                                                    n_modes=50)
    x_modes = bake.infer.cleaned_multiple_modes(mu,
                                                [-x_lim, -x_lim],
                                                [x_lim, x_lim],
                                                n_modes=50)

    # Evaluate the embedding at query points
    mu_init_xq = mu_init(xq)
    mu_xq = mu(xq)
    mu_iso_xq = mu_iso(xq)

    # Plot the query points
    plt.figure(1)
    plt.scatter(xq[:, 0], xq[:, 1], s=20, c = mu_init_xq,
                linewidths=0, label='Initial Embedding')
    plt.scatter(x[:, 0], x[:, 1], c='k', label='Training Data')
    plt.scatter(x_modes_init[:, 0], x_modes_init[:, 1],
                s=20, c='w', marker='o', label='Modes')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title('Bayesian Learning of Kernel Embedding: Initial Embedding')

    plt.figure(2)
    plt.scatter(xq[:, 0], xq[:, 1], s=20, c=mu_iso_xq, linewidths=0,
                label='Learned Embedding')
    plt.scatter(x[:, 0], x[:, 1], c='k', label='Training Data')
    plt.scatter(x_modes_iso[:, 0], x_modes_iso[:, 1], s=20, c='w', marker='o',
                label='Modes')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title(
        'Bayesian Learning of Kernel Embedding: Learned Isotropic Embedding')

    plt.figure(3)
    plt.scatter(xq[:, 0], xq[:, 1], s=20, c=mu_xq, linewidths=0,
                label='Learned Embedding')
    plt.scatter(x[:, 0], x[:, 1], c='k', label='Training Data')
    plt.scatter(x_modes[:, 0], x_modes[:, 1], s=20, c='w', marker='o',
                label='Modes')
    plt.legend()
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-x_lim, x_lim))
    plt.title(
        'Bayesian Learning of Kernel Embedding: Learned Anisotropic Embedding')


if __name__ == "__main__":
    main()
    plt.show()