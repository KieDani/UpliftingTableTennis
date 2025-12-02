import torch
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import einops as eo

from balldetection.helper_balldetection import HEIGHT, WIDTH, get_data_path
from uplifting.data import BACKSPIN_CLASS, TOPSPIN_CLASS, RealInferenceDataset
from uplifting.helper import HEIGHT as SYNTHETIC_HEIGHT, WIDTH as SYNTHETIC_WIDTH


class TTHQ(torch.utils.data.Dataset):
    def __init__(self, transform_ball=None, transform_table=None):
        self.in_frames = 3
        self.transform_ball = transform_ball
        self.transform_table = transform_table
        if self.transform_ball is None or self.transform_table is None:
            print('WARNING: A transform is not provided. This might lead to unexpected results!')
        val_test_vids = [1, 3, 10]  # Videos used for validation and testing

        data_path = os.path.join(get_data_path(), 'tthq')
        self.data_path = data_path

        # Load ball detection data
        ball_dict = pd.read_csv(os.path.join(data_path, 'ball_detection.csv'), sep=';').to_dict()
        videos_ball = list(ball_dict['video'].values())
        frames_ball = list(ball_dict['frame'].values())
        ball_annotations = {k: [] for k in val_test_vids}
        for v, f in zip(videos_ball, frames_ball):
            if v in val_test_vids:
                ball_annotations[v].append(f)

        # load table keypoint data
        table_dict = pd.read_csv(os.path.join(data_path, 'table_detection.csv'), sep=';').to_dict()
        videos_table = list(table_dict['video'].values())
        frames_table = list(table_dict['frame'].values())
        table_annotations = {k: [] for k in val_test_vids}
        for v, f in zip(videos_table, frames_table):
            if v in val_test_vids:
                table_annotations[v].append(f)

        # load event data
        trajectories_dict = pd.read_csv(os.path.join(data_path, 'trajectories.csv'), sep=';').to_dict()
        videos_traj, start_frames, end_frames = list(trajectories_dict['video'].values()), list(trajectories_dict['start_frame'].values()), list(trajectories_dict['end_frame'].values())
        usable_traj, status_traj = list(trajectories_dict['usable'].values()), list(trajectories_dict['status'].values())
        spin_classes = list(trajectories_dict['spin_class'].values())
        framerates = list(trajectories_dict['fps'].values())
        event_annotations = {k: [] for k in val_test_vids}
        for v, s, e, usable, status, fps, sc in zip(videos_traj, start_frames, end_frames, usable_traj, status_traj, framerates, spin_classes):
            if v in val_test_vids and usable == True and status != 'last':
                event_annotations[v].append((s, e, fps, sc))

        num_total_trajectories = sum([len(event_annotations[v]) for v in val_test_vids])
        print(f'Loaded {num_total_trajectories} total trajectories in testing videos. Still need to filter all trajectories with annotations.')

        # remove all trajectories with at least one annotated frame
        filtered_events = {k: [] for k in val_test_vids}
        for video in val_test_vids:
            ball_frames = set(ball_annotations[video])
            table_frames = set(table_annotations[video])
            trajectories_list = event_annotations[video]
            for (s, e, fps, sc) in trajectories_list:
                valid = True
                for frame in range(int(s), int(e) + 1):
                    if frame in ball_frames or frame in table_frames:
                        valid = False  # If any frame is annotated, the whole trajectory is invalid
                        break
                if valid:
                    filtered_events[video].append((int(s), int(e), fps, sc))

        self.data = []  # List to hold (video, fps, spin_class, [(frame1, prev_frame1, next_frame1), ...]) tuples
        for video in val_test_vids:
            trajectories_list = filtered_events[video]
            for (s, e, fps, sc) in trajectories_list:
                trajectory = []
                trajectory_valid = True
                for frame in range(s+1, e):  # TODO: Extract the frames such that I can do range(s, e+1)
                    if not self._check_if_frame_exists(video, frame) or not self._check_if_frame_exists(video, frame - 1) or not self._check_if_frame_exists(video, frame + 1):
                        trajectory_valid = False
                        break
                    trajectory.append((frame, frame - 1, frame + 1))
                if trajectory_valid:
                    self.data.append((video, fps, sc, trajectory))

        print(f'Loaded {len(self.data)} trajectories for testing.')

        # shuffle data
        # rnd = np.random.RandomState(42)
        # rnd.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video, fps, spin_class, trajectory = self.data[idx]
        images_ball, images_table = [], []
        for (frame, prev_frame, next_frame) in trajectory:
            im_path = os.path.join(self.data_path, f'{video:02d}')
            img = cv2.imread(os.path.join(im_path, f'{video:02d}_{frame:06d}.png'))
            prev_img = cv2.imread(os.path.join(im_path, f'{video:02d}_{prev_frame:06d}.png'))
            next_img = cv2.imread(os.path.join(im_path, f'{video:02d}_{next_frame:06d}.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
            next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)

            H, W, C = img.shape

            # transforms for balldetection
            if self.transform_ball is not None:
                data = {
                    'image': img.copy(),
                    'prev_image': prev_img.copy(),
                    'next_image': next_img.copy(),
                }
                data = self.transform_ball(data)
                img_ball, prev_img, next_img = data['image'], data['prev_image'], data['next_image']
            else:
                img_ball = img.copy()
            img_ball = np.concatenate([prev_img, img_ball, next_img], axis=2)
            img_ball = eo.rearrange(img_ball, 'h w c -> c h w')


            # transforms for table detection
            if self.transform_table is not None:
                data = {
                    'image': img.copy(),
                }
                data = self.transform_table(data)
                img_table = data['image']
            else:
                img_table = img.copy()
            img_table = eo.rearrange(img_table, 'h w c -> c h w')
            images_ball.append(img_ball)
            images_table.append(img_table)

        images_ball = np.stack(images_ball, axis=0)
        images_table = np.stack(images_table, axis=0)
        images_ball = torch.from_numpy(images_ball).float()
        images_table = torch.from_numpy(images_table).float()

        return images_ball, images_table, fps, spin_class

    def _check_if_frame_exists(self, video, frame):
        """
        Check if the frame exists in the video.
        """
        im_path = os.path.join(self.data_path, f'{video:02d}')
        return os.path.exists(os.path.join(im_path, f'{video:02d}_{frame:06d}.png'))


class TTST(RealInferenceDataset):
    def __init__(self, transform_ball=None, transform_table=None, transforms_uplifting=None):
        super().__init__(mode='test', transforms=transforms_uplifting)
        self.in_frames = 3
        self.transform_ball = transform_ball
        self.transform_table = transform_table
        print('TTST Dataset initialized for real inference.')

    def __getitem__(self, idx):
        r_img, table_img, mask, times, hits, Mint, Mext, spin_class = super().__getitem__(idx)
        data_path = self.data_paths[idx]

        # rescale Camera matrices from synthetic resolution to real resolution ; Not needed for coordinates due to normalization
        scale_x = WIDTH / SYNTHETIC_WIDTH
        scale_y = HEIGHT / SYNTHETIC_HEIGHT
        Mint = Mint.clone()
        Mint[0, 0] *= scale_x
        Mint[1, 1] *= scale_y
        Mint[0, 2] = (Mint[0, 2] + 0.5) * scale_x - 0.5
        Mint[1, 2] = (Mint[1, 2] + 0.5) * scale_y - 0.5

        frames_path = os.path.join(data_path, 'frames')

        images_ball, images_table = [], []
        for i, __ in enumerate(range(0, int(torch.sum(mask)))):
            frame_idx = i + 1
            prev_idx = frame_idx - 1
            next_idx = frame_idx + 1
            img = cv2.imread(os.path.join(frames_path, f'{frame_idx:03}.png'))
            prev_img = cv2.imread(os.path.join(frames_path, f'{prev_idx:03}.png'))
            next_img = cv2.imread(os.path.join(frames_path, f'{next_idx:03}.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)

            H, W, C = img.shape

            # transforms for balldetection
            if self.transform_ball is not None:
                data = {
                    'image': img.copy(),
                    'prev_image': prev_img.copy(),
                    'next_image': next_img.copy(),
                }
                data = self.transform_ball(data)
                img_ball, prev_img, next_img = data['image'], data['prev_image'], data['next_image']
            else:
                img_ball = img.copy()
            img_ball = np.concatenate([prev_img, img_ball, next_img], axis=2)
            img_ball = eo.rearrange(img_ball, 'h w c -> c h w')

            # transforms for table detection
            if self.transform_table is not None:
                data = {
                    'image': img.copy(),
                }
                data = self.transform_table(data)
                img_table = data['image']
            else:
                img_table = img.copy()
            img_table = eo.rearrange(img_table, 'h w c -> c h w')
            images_ball.append(img_ball)
            images_table.append(img_table)

        images_ball = np.stack(images_ball, axis=0)
        images_table = np.stack(images_table, axis=0)
        images_ball = torch.from_numpy(images_ball).float()
        images_table = torch.from_numpy(images_table).float()

        return images_ball, images_table, r_img, table_img, times, mask, Mint, Mext, spin_class



if __name__ == '__main__':
    from balldetection.transforms import get_transform as get_transforms_ball
    from tabledetection.transforms import get_transform as get_transforms_table
    from uplifting.transformations import get_transforms as get_transforms_uplifting

    transform_ball = get_transforms_ball('test', (HEIGHT, WIDTH))
    transform_table = get_transforms_table('test', (HEIGHT, WIDTH))
    transform_uplifting = get_transforms_uplifting(None, mode='test')

    dataset = TTST(transform_ball=transform_ball, transform_table=transform_table, transforms_uplifting=transform_uplifting)

    # dataset = TTHQ(transform_ball=transform_ball, transform_table=transform_table)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        print(i, data[0].shape, data[1].shape)
        # print(data[2].shape, torch.sum(data[5]))
        if i == 10:
            break




