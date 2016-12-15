"""
Demonstration of simple kernel embeddings.
"""
import bake
import utils
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import numpy as np
import time

seed = 100
x_min = np.array([-5, 0])
x_max = np.array([10, 15])
d = 2


def true_phenomenon(x):
    """
    Branin-Hoo Function.

    Parameters
    ----------
    x : numpy.ndarray
        The input to the function of size (n, 2)

    Returns
    -------
    numpy.ndarray
        The output from the function of size (n, )
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return -f

true_phenomenon.optima = np.array([[-np.pi, 12.275],
                                   [np.pi, 2.275],
                                   [9.42478, 2.475]])

def noisy_phenomenon(x, sigma=1.0):

    f = true_phenomenon(x)
    e = np.random.normal(loc=0, scale=sigma, size=x.shape[0])
    y = f + e
    return y

def create_training_data(n=100, sigma=1.0):

    x = utils.data.generate_uniform_data(n, d, x_min, x_max, seed=seed)
    y = noisy_phenomenon(x, sigma=1.0)
    return x, y


def bayesian_optimisation():

    # Create training data
    x, y = create_training_data(n=30, sigma=1.0)

    # Set the hyperparameters
    theta_x, zeta = 2.0, 0.1

    # Generate the query points
    n_query = 250
    x_1_lim = (x_min[0], x_max[0])
    x_2_lim = (x_min[1], x_max[1])
    x_1_q, x_2_q, x_1_grid, x_2_grid = \
        utils.visuals.uniform_query(x_1_lim, x_2_lim, n_query, n_query)
    x_q = np.array([x_1_grid.ravel(), x_2_grid.ravel()]).T

    # Ground Truth
    y_q_true = true_phenomenon(x_q)

    # # Setup Animation
    plt.ion()
    fig1 = plt.figure(1)
    fig1.canvas.manager.window.geometry('600x500+0+0')
    # ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(2)
    fig2.canvas.manager.window.geometry('600x500+600+0')
    # ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(3)
    fig3.canvas.manager.window.geometry('600x500+1200+0')
    # ax3 = fig1.add_subplot(111)
    #
    # ax1.pcolormesh(x_1_grid, x_2_grid, y_q_true_mesh)
    # ax1.scatter(x[:, 0], x[:, 1], c=y, label='Training Data')
    # # plt.colorbar()
    # ax1.scatter(true_phenomenon.optima[:, 0],
    #             true_phenomenon.optima[:, 1],
    #             c='w', marker='x', label='True Optima')
    # plt.xlim(x_1_lim)
    # plt.ylim(x_2_lim)
    # plt.xlabel('$x_{1}$')
    # plt.ylabel('$x_{2}$')
    # plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
    #            fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
    # plt.title('Ground Truth (Time = %d)' % i)
    # fig.canvas.draw()

    # plt.ion()
    for i in range(100):

        # Conditional weights
        w_q = bake.infer.conditional_weights(x, theta_x, x_q, zeta=zeta)

        # # Weights of the density
        # w_q_norm = bake.infer.clip_normalize(w_q)

        # Predictions
        y_q_exp = bake.infer.expectance(y, w_q)
        x_star = x_q[y_q_exp.argmax()]

        # Acquisition Function
        # ei = np.dot(np.clip(y - np.mean(y), 0, np.inf), w_q)
        # ei = bake.infer.variance(y, w_q)
        ei = np.dot(np.clip(y - np.sort(y)[-2], 0, np.inf), w_q)
        # ei = y_q_exp
        # ei = np.dot((y - np.sort(y)[-2] > 0).astype(float), w_q)
        # ei = np.dot(np.clip(y - np.sqrt(bake.infer.variance(y, w_q)), 0, np.inf), w_q)
        x_propose = x_q[[ei.argmax()]]

        # Convert to mesh
        y_q_true_mesh = np.reshape(y_q_true, (n_query, n_query))
        y_q_exp_mesh = np.reshape(y_q_exp, (n_query, n_query))
        ei_mesh = np.reshape(ei, (n_query, n_query))

        # Plot the predictions
        fig = plt.figure(1)
        # fig.canvas.manager.window.geometry('600x500+0+0')
        plt.clf()
        plt.pcolormesh(x_1_grid, x_2_grid, y_q_true_mesh)
        plt.scatter(x[:, 0], x[:, 1], c=y, label='Training Data')
        plt.colorbar()
        plt.scatter(true_phenomenon.optima[:, 0],
                    true_phenomenon.optima[:, 1],
                    c='w', marker='x', label='True Optima')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8,fancybox=True).get_frame().set_alpha(0.5)
        plt.title('Ground Truth (Time = %d)' % i)
        fig.canvas.draw()

        # Plot the entropy
        fig = plt.figure(2)
        plt.clf()
        # fig.canvas.manager.window.geometry('600x500+600+0')
        plt.pcolormesh(x_1_grid, x_2_grid, y_q_exp_mesh)
        plt.scatter(x[:, 0], x[:, 1], c=y, label='Training Data')
        plt.colorbar()
        plt.scatter(true_phenomenon.optima[:, 0],
                    true_phenomenon.optima[:, 1],
                    c='w', marker='x', label='True Optima')
        plt.scatter(x_star[0], x_star[1],
                    c='k', marker='x', label='Current Optima')
        plt.scatter(x_propose[:, 0], x_propose[:, 1],
                    c='k', marker='+', label='Proposal Point')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('Kernel Embedding Regressor (Time = %d)' % i)

        # Plot the entropy
        fig = plt.figure(3)
        # fig.canvas.manager.window.geometry('600x500+1200+0')
        plt.clf()
        plt.pcolormesh(x_1_grid, x_2_grid, ei_mesh, cmap=cm.jet)
        plt.colorbar()
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.jet, label='Training Data')
        plt.scatter(true_phenomenon.optima[:, 0],
                    true_phenomenon.optima[:, 1],
                    c='w', marker='x', label='True Optima')
        plt.scatter(x_star[0], x_star[1],
                    c='k', marker='x', label='Current Optima')
        plt.scatter(x_propose[:, 0], x_propose[:, 1],
                    c='k', marker='+', label='Proposal Point')
        plt.xlim(x_1_lim)
        plt.ylim(x_2_lim)
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0),
                   fontsize=8, fancybox=True).get_frame().set_alpha(0.5)
        plt.title('Expected Improvement (Time = %d)' % i)

        # Observe!
        y_observe = noisy_phenomenon(x_propose)
        x = np.concatenate((x, x_propose), axis=0)
        y = np.append(y, y_observe)
        print('Time = %d; Proposed Point: ' % i, x_propose)
        time.sleep(0.5)
        plt.draw()
        time.sleep(0.5)

if __name__ == "__main__":
    utils.misc.time_module(bayesian_optimisation)


# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(0, 10 * np.pi, 100)
# y = np.sin(x)
#
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'b-')
#
# for phase in np.linspace(0, 10 * np.pi, 100):
#     line1.set_ydata(np.sin(0.5 * x + phase))
#     fig.canvas.draw()


# FINDINGS
# - EI cannot be computed with kernel embedding projections if compared to the
# maximum, so must compare to second max. This function is unlikely to be in the
# RKHS too
# - Notice that the whole algorithm does not even depend on the kernel in y!
# - This suggests that Bayesian learning for regression and classification
# should not depend on the kernel on the output. In fact, there is no need to
# place a kernel on the output at all!