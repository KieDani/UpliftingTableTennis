import numpy as np
import matplotlib.pyplot as plt
import os

from paths import data_path
from syntheticdataset.helper import table_connections, table_points


def show_trajectory(folder, mode, number, direction):
    path = os.path.join(data_path, folder, mode, direction, f"trajectory_{number:04}")
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")
    file_name = os.path.join("positions.npy")
    file_path = os.path.join(path, file_name)

    positions = np.load(file_path)  # shape: (T, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0.4, 1.5)
    ax.set_title(f"Trajectory {number}")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='red', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='green', label='End')
    for connection in table_connections:
        ax.plot(table_points[connection, 0], table_points[connection, 1], table_points[connection, 2], 'k')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    folder = 'sanitycheck'
    mode = 'final_win'
    direction = 'right_to_left'
    for number in range(0, 10):
        show_trajectory(folder, mode, number, direction)
