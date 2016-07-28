"""
Visualize Pedestrian Dataset.
"""
import matplotlib.pyplot as plt
from pedestrian.data import parse_pedestrian


def main():

    tracks = parse_pedestrian.read_pedestrian_data('data/tracks_aug24.txt')
    [plt.plot(track[:, 0], track[:, 1], c=(0.5, 0.5, 0.5)) for track in tracks]


if __name__ == "__main__":
    main()
    plt.show()