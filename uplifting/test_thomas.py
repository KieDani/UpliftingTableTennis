import numpy as np
import torch
import os
import cv2
import yaml
import pandas as pd
import glob
import matplotlib.pyplot as plt
import einops as eo

from uplifting.helper import get_data_path
from syntheticdataset.helper import table_connections, table_points, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH
from uplifting.helper import cam2img, world2cam


def read_camera_info(yaml_path):
    """
    Reads camera calibration information from a YAML file.

    Parameters:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Camera calibration data including rotation vector (rvec),
              translation vector (tvec), and focal length (f).
    """
    # Load the YAML file
    with open(yaml_path, "r") as file:
        camera_info = yaml.safe_load(file)

    # Extract the relevant information
    rvec = camera_info.get("rvec")
    tvec = camera_info.get("tvec")
    w = camera_info.get("w")
    h = camera_info.get("h")
    f = camera_info.get("f")

    # Return the values as a dictionary
    return np.array(rvec), np.array(tvec), f, h, w



class TT3DDataset(torch.utils.data.Dataset):
    def __init__(self, view='back', noise=True, sequence_len=50):
        assert view in ['back', 'side', 'oblique'], "view must be one of: 'back', 'side', 'oblique'"

        folder_name = f"{view}" if noise else f"{view}_no_noise"
        base_dir = os.path.join(get_data_path(), '..', 'tt3d', 'data', 'evaluation')
        self.data_dir = os.path.join(base_dir, folder_name)
        self.traj_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if len(self.traj_files) == 0:
            raise FileNotFoundError(f"No CSV files found in directory {self.data_dir}")

        yaml_path = os.path.join(base_dir, f"{view}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Camera YAML file not found: {yaml_path}")
        self.rvec, self.tvec, self.f, self.h, self.w = read_camera_info(yaml_path)

        self.sequence_len = sequence_len

    def _count_hits(self, positions, direction, FPS):
        hits_own = []
        hits_opponent = []
        hits_ground = []
        opposite_side_fn = lambda x: -0.01 > x > -TABLE_LENGTH / 2 if direction == 'left_to_right' else TABLE_LENGTH / 2 > x > 0.01
        own_side_fn = lambda x: TABLE_LENGTH / 2 > x > 0.01 if direction == 'left_to_right' else -0.01 > x > -TABLE_LENGTH / 2
        threshold = TABLE_HEIGHT + 0.04
        binary_mask_z = np.array([pos[2] < threshold for pos in positions])  # True if ball is below the threshold
        binary_mask_y = np.array([abs(pos[1]) < TABLE_WIDTH / 2 for pos in positions])  # True if ball is within the table
        binary_mask_x_opponent = np.array([opposite_side_fn(pos[0]) for pos in positions])  # True if ball is within the table on the opposite side
        binary_mask_x_own = np.array([own_side_fn(pos[0]) for pos in positions])  # True if ball is within the table on the own side
        binary_mask_opponent = binary_mask_z & binary_mask_y & binary_mask_x_opponent
        binary_mask_own = binary_mask_z & binary_mask_y & binary_mask_x_own
        binary_mask_ground = np.array([pos[2] <= 0.08 for pos in positions])  # True if ball hits the ground
        positions = np.array(positions)
        start, end = None, None
        for i, b in enumerate(binary_mask_opponent):
            if i == 0 and b:
                start = i
            elif b and not binary_mask_opponent[i - 1]:
                start = i
            if not b and binary_mask_opponent[i - 1] and i != 0:
                end = i - 1
                hits_opponent.append(0.75 * (end + start) / 2 / FPS + 0.25 * (np.argmin(positions[start:end + 1, 2]) + start) / FPS)
        start, end = None, None
        for i, b in enumerate(binary_mask_own):
            if i == 0 and b:
                start = i
            elif b and not binary_mask_own[i - 1]:
                start = i
            if not b and binary_mask_own[i - 1] and i != 0:
                end = i - 1
                hits_own.append(0.75 * (end + start) / 2 / FPS + 0.25 * (np.argmin(positions[start:end + 1, 2]) + start) / FPS)
        start, end = None, None
        for i, b in enumerate(binary_mask_ground):
            if i == 0 and b:
                start = i
            elif b and not binary_mask_ground[i - 1]:
                start = i
            if not b and binary_mask_ground[i - 1] and i != 0:
                end = i - 1
                hits_ground.append(0.75 * (end + start) / 2 / FPS + 0.25 * (np.argmin(positions[start:end + 1, 2]) + start) / FPS)
        return hits_opponent, hits_own, hits_ground


    def __len__(self):
        return len(self.traj_files)


    def __getitem__(self, idx):
        csv_path = self.traj_files[idx]
        df = pd.read_csv(csv_path)

        # Extract arrays from CSV
        r_img = df[["u", "v"]].values
        r_world = df[["X", "Y", "Z"]].values
        times = df["Timestamp"].values

        # mask is needed to indicate which values are padding (0) and which are real values (1)
        T, _ = r_img.shape
        mask = np.empty((self.sequence_len,), dtype=np.bool)
        mask[:T] = True
        mask[T:] = False

        # Crop or pad sequence
        max_t = min(T, self.sequence_len)
        tmp = np.zeros((self.sequence_len, 2))
        tmp[:max_t] = r_img[:max_t]
        r_img = tmp

        tmp = np.zeros((self.sequence_len, 3))
        tmp[:max_t] = r_world[:max_t]
        r_world = tmp

        tmp = np.zeros((self.sequence_len,))
        tmp[:max_t] = times[:max_t]
        times = tmp

        # Camera matrices
        Mint = np.array([
                [self.f, 0, self.w / 2],
                [0, self.f, self.h / 2],
                [0, 0, 1],
            ])
        R, _ = cv2.Rodrigues(self.rvec)
        Mext = np.eye(4, dtype=np.float32)
        Mext[:3, :3] = R
        Mext[:3, 3] = self.tvec.flatten()
        # transform from thomas coordinate system into my coordinate system
        trans_matrix = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, -TABLE_HEIGHT],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        Mext = Mext @ trans_matrix

        # Convert world table coordinates to image coordinates
        table_img = cam2img(world2cam(np.array(table_points, dtype=np.float32), Mext), Mint)

        # Convert world coordinates from Thomas coordinate system to my coordinate system
        r_world = eo.einsum(np.linalg.inv(trans_matrix), np.concatenate([r_world, np.ones((r_world.shape[0], 1))], axis=1), 'i j,t j->t i')[:, :3]

        # Framerate (via mean delta t)
        if max_t > 1:
            dt = np.diff(times[:max_t])
            avg_dt = np.mean(dt)
            framerate = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            raise ValueError("Not enough data points to calculate framerate. At least 2 points are required.")

        # Normalize r_img to be in the range [0, 1]
        r_img[:, 0] = r_img[:, 0] / self.w
        r_img[:, 1] = r_img[:, 1] / self.h
        table_img[:, 0] = table_img[:, 0] / self.w
        table_img[:, 1] = table_img[:, 1] / self.h

        # calculate bounces
        direction = 'left_to_right' if r_world[0, 0] < 0 else 'right_to_left'
        hits_own, hits_opponent, hits_ground = self._count_hits(r_world, direction, framerate)
        hits = sorted(hits_opponent + hits_own)
        if len(hits) == 0:
            hits = np.array([-1,], dtype=np.float32)
        else:
            hits = hits[0:1]

        # Convert to torch.Tensor
        dtype = torch.float32
        r_img = torch.tensor(r_img, dtype=dtype)
        table_img = torch.tensor(table_img, dtype=dtype)
        mask = torch.tensor(mask, dtype=dtype)
        r_world = torch.tensor(r_world, dtype=dtype)
        times = torch.tensor(times, dtype=dtype)
        hits = torch.tensor(hits, dtype=dtype)
        Mint = torch.tensor(Mint, dtype=dtype)
        Mext = torch.tensor(Mext, dtype=dtype)
        framerate = torch.tensor(framerate, dtype=dtype)

        return r_img, table_img, mask, r_world, times, hits, Mint, Mext, framerate


# TODO: Plot table and introduce rotation/translation matrix until it looks correct from each view (no image frame needed)


def show_trajectory_3d(r_world):
    positions = r_world  # shape: (T, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0.4, 1.5)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='red', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='green', label='End')
    for connection in table_connections:
        ax.plot(table_points[connection, 0], table_points[connection, 1], table_points[connection, 2], 'k')
    ax.legend()
    plt.show()


def show_trajectory_2d(r_img, table_img, r_world, Mext, Mint, h, w):
    # Convert to pixel coordinates
    r_img = r_img.numpy()
    r_img[:, 0] = r_img[:, 0] * w
    r_img[:, 1] = r_img[:, 1] * h

    # table_img = cam2img(world2cam(torch.tensor(table_points, dtype=torch.float32), Mext), Mint).numpy()
    table_img = table_img.numpy()
    table_img[:, 0] = table_img[:, 0] * w
    table_img[:, 1] = table_img[:, 1] * h

    for connection in table_connections:
        plt.plot(table_img[connection, 0], table_img[connection, 1], 'k')

    for i in range(len(r_img)):
        plt.plot(r_img[i, 0], r_img[i, 1], 'o', markersize=3, color='blue')

    # calculate the reprojected trajectory as sanity check
    r_img_rep = cam2img(world2cam(r_world, Mext), Mint).numpy()
    for i in range(len(r_img_rep)):
        plt.plot(r_img_rep[i, 0], r_img_rep[i, 1], 'x', markersize=3, color='red')


    # plot the coordinate system axis
    origin = torch.tensor([0, 0, 0.76], dtype=torch.float32)
    x_axis = torch.tensor([1, 0, 0.76], dtype=torch.float32)
    y_axis = torch.tensor([0, 1, 0.76], dtype=torch.float32)
    z_axis = torch.tensor([0, 0, 1.76], dtype=torch.float32)
    origin_img = cam2img(world2cam(origin.unsqueeze(0), Mext), Mint).numpy()
    x_axis_img = cam2img(world2cam(x_axis.unsqueeze(0), Mext), Mint).numpy()
    y_axis_img = cam2img(world2cam(y_axis.unsqueeze(0), Mext), Mint).numpy()
    z_axis_img = cam2img(world2cam(z_axis.unsqueeze(0), Mext), Mint).numpy()
    plt.plot([origin_img[0, 0], x_axis_img[0, 0]], [origin_img[0, 1], x_axis_img[0, 1]], 'r', linewidth=2, label='X-axis')
    plt.plot([origin_img[0, 0], y_axis_img[0, 0]], [origin_img[0, 1], y_axis_img[0, 1]], 'g', linewidth=2, label='Y-axis')
    plt.plot([origin_img[0, 0], z_axis_img[0, 0]], [origin_img[0, 1], z_axis_img[0, 1]], 'b', linewidth=2, label='Z-axis')


    plt.xlim(0, w)
    plt.ylim(h, 0)

    plt.show()


def main():
    dataset = TT3DDataset(view='side', noise=True, sequence_len=50)
    sample = dataset[0]
    r_img, table_img, mask, r_world, times, hits, Mint, Mext, framerate = sample
    seq_len = int(mask.sum().item())
    r_img = r_img[:seq_len]
    r_world = r_world[:seq_len]

    times = times[:seq_len]

    show_trajectory_2d(r_img, table_img, r_world, Mext, Mint, dataset.h, dataset.w)


if __name__ == '__main__':
    main()










