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