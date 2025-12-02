import numpy as np
import random
import torch
import os
import math
import cv2
import yaml
import pandas as pd
import glob
import einops as eo

from uplifting.helper import get_Mext, cam2img, world2cam
from uplifting.helper import HEIGHT, WIDTH, base_fx, base_fy
from uplifting.helper import get_data_path
from uplifting.helper import table_points, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH
from uplifting.helper import KEYPOINT_VISIBLE, KEYPOINT_INVISIBLE
from paths import data_path as DATA_PATH


BACKSPIN_CLASS = 2
TOPSPIN_CLASS = 1
NotANNOTATED_CLASS = 0


class TableTennisDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', transforms=None):
        self.mode = mode
        path = get_data_path()

        trajectory_modes = ['intermediate', 'final_win', 'final_lose', 'first_good', 'first_short', 'first_long']
        directions = ['left_to_right', 'right_to_left']
        data_paths = []
        for tm in trajectory_modes:
            for direction in directions:
                dps = sorted([os.path.join(path, tm, direction, f'trajectory_{i:04}') for i, __ in enumerate(os.listdir(os.path.join(path, tm, direction)))])
                rnd = random.Random(0)
                rnd.shuffle(data_paths)
                # get the same number of train/val/test data for each mode and direction
                if mode == 'train':
                    dps = dps[:int(0.7 * len(dps))]
                elif mode == 'val':
                    dps = dps[int(0.7 * len(dps)):int(0.8 * len(dps))]
                elif mode == 'test':
                    dps = dps[int(0.8 * len(dps)):]
                else:
                    raise ValueError(f'Unknown mode {mode}')
                data_paths.extend(dps)
        self.data_paths = data_paths
        self.length = len(self.data_paths)

        self.sequence_len = 50 # crop sequence if it is longer, else padding of sequence

        # The resolution of the simulated video frames -> rescale the coordinates from this resolution to the working resolution (WIDTH, HEIGHT)
        self.original_resolution = (2560, 1440)

        self.transforms = transforms

        # minimum and maximum phi that is sampled for the camera position
        self.sampled_phis = (np.rad2deg(math.atan2(TABLE_WIDTH/2, TABLE_LENGTH/2)), np.rad2deg(math.atan2(TABLE_WIDTH/2, TABLE_LENGTH/2))+180)
        self.sampled_distances = (7, 17)  # minimum and maximum distance of the camera to the origin
        self.sampled_thetas = (30, 70)  # minimum and maximum theta that is sampled for the camera position
        self.sampled_fx = (0.6 * base_fx, 2.0 * base_fx)  # minimum and maximum focal length in x direction
        self.sampled_fy = (0.6 * base_fy, 2.0 * base_fy)

        self.num_cameras = 1 if mode in ['val', 'test'] else 1
        self.cam_num = 0

        self.fps_bounds = (20, 65)  # minimum and maximum framerate
        self.eval_fps = 50  # frames per second for evaluation


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        blur_positions = np.load(os.path.join(data_path, 'positions.npy'))
        blur_times = np.load(os.path.join(data_path, 'times.npy'))
        bounces = np.load(os.path.join(data_path, 'bounces.npy'))
        rotation = np.load(os.path.join(data_path, 'rotations.npy'))[0]

        # sample a framerate and calculate times and r_world based on the new framerate
        fps = random.randint(self.fps_bounds[0], self.fps_bounds[1]) if self.mode == 'train' else self.eval_fps
        start_time = blur_times[0]
        end_time = blur_times[-1]
        times = np.arange(start_time, end_time, 1.0 / fps)
        insertion_points = np.searchsorted(blur_times, times)
        idx_right = np.clip(insertion_points, 0, len(blur_times) - 1)
        idx_left = np.clip(insertion_points - 1, 0, len(blur_times) - 1)
        diff_left = np.abs(blur_times[idx_left] - times)
        diff_right = np.abs(blur_times[idx_right] - times)
        nearest_frame_indices = np.where(diff_right < diff_left, idx_right, idx_left)
        r_world = blur_positions[nearest_frame_indices]

        if self.mode == 'train':
            Mint, Mext, r_img, table_img, success = self.sample_camera(r_world)
        else:
            Mint = np.load(os.path.join(data_path, 'Mint.npy'))
            Mext = np.load(os.path.join(data_path, 'Mext.npy'))
            Mint = Mint[nearest_frame_indices]
            Mext = Mext[nearest_frame_indices]
            assert np.sum(Mext[1:] - Mext[:-1]) < 1e-6, 'Actual computations are only correct for static camera'
            Mint, Mext = Mint[0], Mext[0]
            Mext, Mint = self.transform_evaluation_camera(self.cam_num, Mext, Mint)
            r_cam = world2cam(r_world, Mext)
            r_img = cam2img(r_cam, Mint)
            table_cam = world2cam(table_points, Mext)
            table_img = cam2img(table_cam, Mint)

        # mask is needed to indicate which values are padding (0) and which are real values (1)
        T, __ = r_img.shape
        mask = np.empty((self.sequence_len,), dtype=np.bool)
        mask[:T] = True # real values
        mask[T:] = False # padding

        # crop or pad sequence
        max_t = min(T, self.sequence_len)
        tmp = np.zeros((self.sequence_len, 2))
        tmp[:max_t] = r_img[:max_t]
        r_img = tmp
        tmp = np.zeros((self.sequence_len, 3))
        tmp[:max_t] = r_world[:max_t]
        r_world = tmp
        tmp = np.zeros((self.sequence_len))
        tmp[:max_t] = times[:max_t]
        times = tmp

        # if no bounces are present, set to -1  --> Relevant for RandomStopAugmentation
        if len(bounces) == 0:
            bounces = np.array([-1,], dtype=np.float32)

        # Add visibility to table keypoints as extra dimension: (13, 2) --> (13, 3) ; All keypoints are visible at the moment
        table_img = np.concatenate([table_img, np.full((table_img.shape[0], 1), KEYPOINT_VISIBLE, dtype=table_img.dtype)], axis=1)

        # rescale coordinates from original resolution to processing resolution
        data = {
            'r_img': r_img,
            'table_img': table_img,
            'Mint': Mint,
        }
        data = transform_resolution(data, self.original_resolution, (WIDTH, HEIGHT))
        r_img, table_img, Mint = data['r_img'], data['table_img'], data['Mint']

        # apply transforms
        data = {
            'r_img': r_img,
            'r_world': r_world,
            'times': times,
            'hits': bounces,
            'rotation': rotation,
            'mask': mask,
            'table_img': table_img,
            'Mint': Mint,
            'Mext': Mext,
            'blur_positions': blur_positions,
            'blur_times': blur_times
        }
        if self.transforms is not None: data = self.transforms(data)
        r_img, table_img, mask, r_world, rotation = data['r_img'], data['table_img'], data['mask'], data['r_world'], data['rotation']
        times, bounces, Mint, Mext = data['times'], data['hits'], data['Mint'], data['Mext']
        blur_positions, blur_times = data['blur_positions'], data['blur_times']

        r_img, table_img, mask = torch.tensor(r_img, dtype=torch.float32), torch.tensor(table_img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        r_world, rotation, times = torch.tensor(r_world, dtype=torch.float32), torch.tensor(rotation, dtype=torch.float32), torch.tensor(times, dtype=torch.float32)
        bounces, Mint, Mext = torch.tensor(bounces, dtype=torch.float32), torch.tensor(Mint, dtype=torch.float32), torch.tensor(Mext, dtype=torch.float32)

        # Just return the first bounce --> Matters for RandomStopAugmentation; negative if no bounce in trajectory
        bounces = bounces[0:1]
        return r_img, table_img, mask, r_world, rotation, times, bounces, Mint, Mext

    def sample_camera(self, r_world):
        valid = False
        try_num = 0
        max_tries = 100
        while not valid and try_num < max_tries:
            fx, fy = random.uniform(self.sampled_fx[0], self.sampled_fx[1]), random.uniform(self.sampled_fy[0], self.sampled_fy[1])
            Mint = np.array([[fx, 0, (WIDTH - 1) / 2], [0, fy, (HEIGHT - 1) / 2], [0, 0, 1]])

            # extrinsic matrix
            # distance between 5m and 15m
            distance = random.uniform(self.sampled_distances[0], self.sampled_distances[1])
            # phi angle between 20 and 160 degrees
            phi = random.uniform(self.sampled_phis[0], self.sampled_phis[1])
            # theta angle between 30 and 70 degrees
            theta = random.uniform(self.sampled_thetas[0], self.sampled_thetas[1])
            # lookat point somewhere around the center of the table
            lookat = np.array((random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), TABLE_HEIGHT))

            # camera location
            c = np.array([distance * np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                          distance * np.sin(np.radians(theta)) * np.sin(np.radians(phi)),
                          distance * np.cos(np.radians(theta))])
            c += np.array([0., 0., TABLE_HEIGHT])
            # forward direction
            f = -(c - lookat) / np.linalg.norm(c - lookat)
            # right direction (choose a random vector approximately in the x-y plane)
            epsilon = random.uniform(-0.1, 0.1)  # small random value that controls the deviation from the x-y plane
            r = np.array([-f[1] / f[0] - f[2] / f[0] * epsilon, 1, epsilon])
            r /= np.linalg.norm(r)
            # up direction
            u = -np.cross(f, r)
            if u[2] < 0:  # The up vector has to be in the positive z direction
                r = np.array([f[1] / f[0] - f[2] / f[0] * epsilon, -1, epsilon])  # choose the other direction for r_y
                r /= np.linalg.norm(r)
                u = -np.cross(f, r)
            u /= np.linalg.norm(u)
            # extrinsic matrix
            Mext = get_Mext(c, f, r)

            # calculate image coordinates of trajectory with estimated camera matrices
            r_cam = world2cam(r_world, Mext)
            r_img = cam2img(r_cam, Mint)
            # calculate the table position in image coordinates
            table_cam = world2cam(table_points, Mext)
            table_img = cam2img(table_cam, Mint)
            # check if trajectory is completely inside the image
            valid = np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT])))
            # check if trajectory is not too small in the image
            valid = valid and (r_img[:, 0].max() - r_img[:, 0].min() > 0.15 * WIDTH or r_img[:, 1].max() - r_img[:, 1].min() > 0.15 * HEIGHT)

            try_num += 1
        success = True if try_num < max_tries else False
        return Mint, Mext, r_img, table_img, success

    def mirror_trajectory(self, r_world):
        '''
            Mirror the trajectory along the y-z plane with 50% probability.
            The synthetic trajectories are all simulated from left to right, so this transformation is needed to simulate the right to left trajectory.
            Also mirror along the x-z plane to increase the data diversity.
        '''
        # mirror the trajectory
        # if random.random() < 0.5:
        #     r_world[:, 0] = -r_world[:, 0]
        if random.random() < 0.5:
            r_world[:, 1] = -r_world[:, 1]
        return r_world


    def transform_evaluation_camera(self, cam_num, Mext, Mint):
        '''Load specific evaluation cameras'''
        assert self.mode in ['val', 'test'], 'Evaluation camera can only be used for validation or test set'
        if cam_num == 0:
            pass
        else:
            raise ValueError(f'Unknown camera number {cam_num}')
        return Mext, Mint


class RealInferenceDataset(torch.utils.data.Dataset):
    '''Dataset for real data inference'''
    def __init__(self, mode, transforms=None):
        path = os.path.join(DATA_PATH, 'ttst')
        self.path = path
        self.transforms = transforms
        assert mode in ['val', 'test'], "mode must be one of: 'val', 'test'"

        self.data_paths = sorted([os.path.join(path, foldername) for i, foldername in enumerate(os.listdir(path)) if foldername.startswith('trajectory_')])

        self.sequence_len = 50 # crop sequence if it is longer, else padding of sequence

        self.original_resolution = (2560, 1440)

        if mode == 'val':
            self.data_paths = self.data_paths[:int(0.33 * len(self.data_paths))]  # 33% for validation
        elif mode == 'test':
            self.data_paths = self.data_paths[int(0.33 * len(self.data_paths)):]
        else:
            raise ValueError(f'Unknown mode {mode}')
        self.length = len(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        r_img = np.load(os.path.join(data_path, 'r_img.npy'))
        times = np.load(os.path.join(data_path, 'times.npy'))
        hits = np.load(os.path.join(data_path, 'hits.npy'))
        # TODO: At the moment we have one matrix per sequence -> In the future maybe adjust it to one matrix per frame
        Mint = np.load(os.path.join(data_path, 'Mint.npy'))
        Mext = np.load(os.path.join(data_path, 'Mext.npy'))
        spin_class = np.load(os.path.join(data_path, 'spin_class.npy'))

        # TODO: In the future, use annotated points directly instead of regressed matrices
        table_cam = world2cam(table_points, Mext)
        table_img = cam2img(table_cam, Mint)

        # mask is needed to indicate which values are padding (0) and which are real values (1)
        T, __ = r_img.shape
        mask = np.empty((self.sequence_len,), dtype=np.bool)
        mask[:T] = True
        mask[T:] = False

        # crop or pad sequence
        max_t = min(T, self.sequence_len)
        tmp = np.zeros((self.sequence_len, 2))
        tmp[:max_t] = r_img
        r_img = tmp
        tmp = np.zeros((self.sequence_len))
        tmp[:max_t] = times
        times = tmp

        # Add visibility to table keypoints as extra dimension: (13, 2) --> (13, 3) ; All keypoints are visible at the moment
        table_img = np.concatenate([table_img, np.full((table_img.shape[0], 1), KEYPOINT_VISIBLE, dtype=table_img.dtype)], axis=1)

        # rescale coordinates from original resolution to processing resolution
        data = {
            'r_img': r_img,
            'table_img': table_img,
            'Mint': Mint
        }
        data = transform_resolution(data, self.original_resolution, (WIDTH, HEIGHT))
        r_img, table_img, Mint = data['r_img'], data['table_img'], data['Mint']

        # apply transforms
        data = {
            'r_img': r_img,
            'times': times,
            'hits': hits,
            'mask': mask,
            'table_img': table_img,
            'Mint': Mint,
            'Mext': Mext,
            'spin_class': spin_class
        }
        if self.transforms is not None: data = self.transforms(data)
        r_img, table_img, mask = data['r_img'], data['table_img'], data['mask']
        times, hits, Mint, Mext, spin_class = data['times'], data['hits'], data['Mint'], data['Mext'], data['spin_class']

        dtype = torch.float32
        r_img, table_img, mask = torch.tensor(r_img, dtype=dtype), torch.tensor(table_img, dtype=dtype), torch.tensor(mask, dtype=dtype)
        times, hits, Mint, Mext = torch.tensor(times, dtype=dtype), torch.tensor(hits, dtype=dtype), torch.tensor(Mint, dtype=dtype), torch.tensor(Mext, dtype=dtype)
        spin_class = torch.tensor(spin_class, dtype=dtype)

        return r_img, table_img, mask, times, hits, Mint, Mext, spin_class


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
    def __init__(self, view='back', noise=True):
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

        self.sequence_len = 50

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

        hits = np.zeros((self.sequence_len,))  # Dummy

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
            raise ValueError("Not enough data points to calculate framerate.")

        # Normalize r_img to be in the range [0, 1]
        r_img[:, 0] = r_img[:, 0] / self.w
        r_img[:, 1] = r_img[:, 1] / self.h
        table_img[:, 0] = table_img[:, 0] / self.w
        table_img[:, 1] = table_img[:, 1] / self.h

        # Add visibility to table keypoints as extra dimension: (13, 2) --> (13, 3) ; All keypoints are visible at the moment
        table_img = np.concatenate([table_img, np.full((table_img.shape[0], 1), KEYPOINT_VISIBLE, dtype=table_img.dtype)], axis=1)

        # calculate bounces
        direction = 'left_to_right' if r_world[0, 0] < 0 else 'right_to_left'
        hits_own, hits_opponent, hits_ground = self._count_hits(r_world, direction, framerate)
        hits = sorted(hits_opponent + hits_own)
        if len(hits) == 0:
            hits = np.array([-1, ], dtype=np.float32)
        else:
            hits = hits[0:1]  # Only return the first hit

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


def transform_resolution(data, original_resolution, processing_resolution):
    '''    Transform coordinates from original resolution to processing resolution.
    Arguments:
        data: dict with keys 'r_img', 'table_img', Mint
        original_resolution: tuple (width, height)
        processing_resolution: tuple (width, height)
    '''
    assert 'r_img' in data and 'table_img' in data and 'Mint' in data, "data must contain keys 'r_img', 'table_img', 'Mint'"
    orig_w, orig_h = original_resolution
    proc_w, proc_h = processing_resolution
    scale_x = proc_w / orig_w
    scale_y = proc_h / orig_h
    r_img = data['r_img']
    r_img[..., 0] = (r_img[..., 0] + 0.5) * scale_x - 0.5
    r_img[..., 1] = (r_img[..., 1] + 0.5) * scale_y - 0.5
    data['r_img'] = r_img
    table_img = data['table_img']
    table_img[..., 0] = (table_img[..., 0] + 0.5) * scale_x - 0.5
    table_img[..., 1] = (table_img[..., 1] + 0.5) * scale_y - 0.5
    data['table_img'] = table_img
    Mint = data['Mint']
    Mint[0, 0] = Mint[0, 0] * scale_x
    Mint[1, 1] = Mint[1, 1] * scale_y
    Mint[0, 2] = (Mint[0, 2] + 0.5) * scale_x - 0.5
    Mint[1, 2] = (Mint[1, 2] + 0.5) * scale_y - 0.5
    data['Mint'] = Mint
    return data




if __name__ == '__main__':
    dataset = TT3DDataset(view='side', noise=True, sequence_len=50)
    for i in range(len(dataset)):
        r_img, table_img, mask, r_world, times, hits, Mint, Mext, framerate = dataset[i]
        print(f"Sample {i}:")
        print(f"  r_img shape: {r_img.shape}")
        print(f"  table_img shape: {table_img.shape}")
        print(f"  mask shape: {mask.shape}")
        print(f"  r_world shape: {r_world.shape}")
        print(f"  times shape: {times.shape}")
        print(f"  hits shape: {hits.shape}")
        print(f"  Mint shape: {Mint.shape}")
        print(f"  Mext shape: {Mext.shape}")
        print(f"  Framerate: {framerate.item()}")



    quit()

    import tqdm
    from helper import transform_rotationaxes
    dataset = TableTennisDataset(mode='train')
    avgrot_pos = np.array([0., 0, 0])
    avgrot_neg = np.array([0., 0, 0])
    stdrot_pos = np.array([0., 0, 0])
    stdrot_neg = np.array([0., 0, 0])
    num_pos = np.array([0., 0, 0])
    num_neg = np.array([0., 0, 0])
    number_frames = 0
    for i, data in enumerate(tqdm.tqdm(dataset)):
        rotation, r_gt = data[4], data[3]
        mask = data[2]
        rotation = transform_rotationaxes(torch.tensor(rotation), torch.tensor(r_gt)).numpy()
        avgrot_pos += np.where(rotation > 0, rotation, 0)
        stdrot_pos += np.where(rotation > 0, rotation ** 2, 0)
        num_pos += np.where(rotation > 0, 1, 0)
        avgrot_neg += np.where(rotation < 0, rotation, 0)
        stdrot_neg += np.where(rotation < 0, rotation ** 2, 0)
        num_neg += np.where(rotation < 0, 1, 0)
        number_frames += mask.sum()
    print('Positive rotations:', avgrot_pos / num_pos)
    print('Std of positive:', np.sqrt(stdrot_pos / num_pos - (avgrot_pos / num_pos) ** 2))
    print('Number of positive:', num_pos)
    print('Negative rotations:', avgrot_neg / num_neg)
    print('Std of negative:', np.sqrt(stdrot_neg / num_neg - (avgrot_neg / num_neg) ** 2))
    print('Number of negative:', num_neg)
    print('Number of frames:', number_frames)
    quit()
    print('-'*50)

    from paths import data_path
    dataset = RealInferenceDataset(path=os.path.join(data_path, 'tabletennis_annotations_processed/'))
    number_frames = 0
    num_pos, num_neg = 0, 0
    for i, data in enumerate(tqdm.tqdm(dataset)):
        mask = data[2]
        spin_class = data[7]
        if spin_class == 1:
            num_pos += 1
        elif spin_class == 2:
            num_neg += 1
        else:
            print('Number of frames without spin class:', mask.sum())
            continue # spin class was not annotated
        number_frames += mask.sum()
    print('Number of frames:', number_frames)
    print('Number of positive:', num_pos)
    print('Number of negative:', num_neg)






