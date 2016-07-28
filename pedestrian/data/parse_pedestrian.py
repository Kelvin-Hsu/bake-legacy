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


def extract_one_current_past_data(track):

    previous_positions = track[:-1, :]
    current_position = track[1:, :]

    return previous_positions, current_position


def extract_all_current_past_data(tracks):

    all_previous_positions = []
    all_current_positions = []

    for track in tracks:
        previous_positions, current_position = \
            extract_one_current_past_data(track)
        all_previous_positions.append(previous_positions)
        all_current_positions.append(current_position)

    x_previous = np.concatenate(tuple(all_previous_positions), axis=0)
    x_current = np.concatenate(tuple(all_current_positions), axis=0)

    return x_previous, x_current
