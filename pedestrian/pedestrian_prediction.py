"""
Predicting Pedestrian Paths.
"""
import numpy as np
import bake
import matplotlib.pyplot as plt
from pedestrian.data import parse_pedestrian

def main(learn=False):

    # Load the data
    tracks = parse_pedestrian.read_pedestrian_data('data/tracks_aug24.txt')
    print('There are originally %d tracks.' % len(tracks))

    # Determine how many tracks to pick
    n_tracks = 200
    tracks = [tracks[i] for i in range(n_tracks)]
    print('We pick the first %d tracks only.' % n_tracks)

    # Determine how many steps we will skip
    k = 4
    print('The current and previous positions are %d steps apart.' % k)

    # Extract the previous state and current position data
    pos_previous, pos_current = parse_pedestrian.extract_all_current_past_data(tracks, k=k)
    state_previous = parse_pedestrian.extract_state(pos_previous, pos_current, dt=1)

    # Determine how many original training points we have
    n_original = state_previous.shape[0]
    print('There are originally %d states to train on.' % n_original)

    # Choose a certain number of data points to train on
    n = 2000
    print('We want to choose %d data points to train on.' % n)
    ind = np.random.choice(n_original, n, replace=False)
    x_orig = state_previous[ind, :]
    y_orig = pos_current[ind, :]

    # Whiten the data
    x, x_whiten_params = whiten(x_orig)
    y, y_whiten_params = whiten(y_orig)


    # The original bounds and whitened bounds
    print('Before Whitening | x:', x_orig .min(axis=0), x_orig .max(axis=0))
    print('Before Whitening | y:', y_orig .min(axis=0), y_orig .max(axis=0))
    print('After Whitening | x:', x.min(axis=0), x.max(axis=0))
    print('After Whitening | y:', y.min(axis=0), y.max(axis=0))

    plt.figure()
    [plt.plot(track[np.arange(0, track.shape[0], k), 0], track[np.arange(0, track.shape[0], k), 1]) for track in tracks]
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Tracks')

    plt.figure()
    visualize_state(state_previous)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Original Training States')

    plt.figure()
    visualize_state(x)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Reduced and Whitened Training States')

    # # Set the hyperparameters
    # if learn:
    #     hyper_min = ([1], [10], [0.0008], [10], [1e-6])
    #     hyper_max = ([1000], [100], [0.08], [1000], [1.0])
    #     theta_pos, theta_vel, zeta, psi, sigma = \
    #         bake.learn.optimal_conditional_embedding(
    #             x, y, hyper_min, hyper_max, hyper_warp=hyper_warp)
    #     print(theta_pos, theta_vel, zeta, psi, sigma)
    # else:
    #     theta_pos, theta_vel, zeta = np.array([400]), np.array([30]), np.array([0.1])
    # theta_x = np.concatenate((theta_pos, theta_pos, theta_vel, theta_vel))
    # theta_y = np.concatenate((theta_pos, theta_pos))


def hyper_warp(theta_pos, theta_vel, zeta, psi, sigma):
    theta_x = np.concatenate((theta_pos, theta_pos, theta_vel, theta_vel))
    theta_y = np.concatenate((theta_pos, theta_pos))
    return theta_x, theta_y, zeta, psi, sigma


def visualize_state(state):

    x = state[:, 0]
    y = state[:, 1]
    u = state[:, 2]
    v = state[:, 3]

    plt.quiver(x, y, u, v)


def whiten(x_orig, whiten_params=None):

    if whiten_params is None:
        x_mean = x_orig.mean(axis=0)
        x_std = x_orig.std(axis=0)
        whiten_params = (x_mean, x_std)
    else:
        x_mean, x_std = whiten_params

    x_whiten = (x_orig - x_mean) / x_std
    return x_whiten, whiten_params


def dewhiten(x_whiten, whiten_params):

    x_mean, x_std = whiten_params
    x_orig = x_std * x_whiten + x_mean
    return x_orig




if __name__ == "__main__":
    main(learn=True)
    plt.show()