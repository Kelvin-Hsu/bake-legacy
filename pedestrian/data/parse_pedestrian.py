"""
Pedestrian Data Parsing Module.
"""
import numpy as np


def r_replace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def str_to_arr(s):
    arr = r_replace(s.replace('[', '').replace(']', ''), ';', '', 1)
    return np.array(np.matrix(arr))


def read_pedestrian_data(fname):
    tracks = []
    with open(fname, 'r') as f:
        for line in f:
            if line.strip().startswith('TRACK.'):
                tracks.append(str_to_arr(line.split('=')[1])[:, :2])
    return tracks


def extract_one_current_past_data(track, k=1):

    n, _ = track.shape

    ind = np.arange(0, n, k)
    previous_positions = track[ind[:-1], :]
    current_position = track[ind[1:], :]

    return previous_positions, current_position


def extract_all_current_past_data(tracks, k=1):

    all_previous_positions = []
    all_current_positions = []

    for track in tracks:
        previous_positions, current_position = \
            extract_one_current_past_data(track, k=k)
        all_previous_positions.append(previous_positions)
        all_current_positions.append(current_position)

    pos_previous = np.concatenate(tuple(all_previous_positions), axis=0)
    pos_current = np.concatenate(tuple(all_current_positions), axis=0)

    return pos_previous, pos_current


def extract_state(pos_previous, pos_current, dt=1):

    vel_previous = (pos_current - pos_previous)/dt

    state_previous = np.concatenate((pos_previous, vel_previous), axis=1)

    return state_previous
