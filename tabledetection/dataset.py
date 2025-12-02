import torch
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import yaml

from tabledetection.helper_tabledetection import HEIGHT, WIDTH, get_data_path, TABLE_HEIGHT, table_points, cam2img, world2cam, \
    KEYPOINT_VISIBLE, KEYPOINT_INVISIBLE


class TableTennisTable(torch.utils.data.Dataset):
    def __init__(self, mode, heatmap_sigma, transform=None):
        raise NotImplementedError("This dataset is deprecated. Please use the TTHQ dataset instead. I should implement this dataset using the extracted data from extract_ttst_data.py")
        assert mode in ['train', 'val', 'test']
        if mode == 'train':
            vids = [1, 3, 4, 6]
        elif mode == 'val':
            vids = [2, 5]
        elif mode == 'test':
            vids = [2, 5]

        data_path = os.path.join(get_data_path(), 'TTST', 'tabledetection_data')
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform

        labels_df = pd.read_csv(os.path.join(data_path, 'labels.csv'))

        self.data = []  # [(number in dataname, imagepath, (01_x, 01_y), (02_x, 02_y), (03_x, 03_y), ...)), ...]
        for path in os.listdir(data_path):
            if path.endswith('.png'):
                frame_number = int(path.split('.')[0])

                # get the image path
                image_path = os.path.join(data_path, path)

                coord_list = []
                # get the coordinates
                for coord_num in range(1, 14):
                    x = labels_df.loc[labels_df['number'] == frame_number][f'{coord_num:02d}_x'].values[0]
                    y = labels_df.loc[labels_df['number'] == frame_number][f'{coord_num:02d}_y'].values[0]
                    v = KEYPOINT_VISIBLE  # always visible in this dataset
                    coord_list.append((x, y, v))

                # check if the video number is in the respective set (train, val, test)
                vid_num = labels_df.loc[labels_df['number'] == frame_number]['vid_num'].values[0]
                if vid_num in vids:
                    self.data.append([frame_number, image_path, *coord_list])

        # random sort
        rnd = np.random.RandomState(0)
        rnd.shuffle(self.data)

        # split val and test data (they are the same videos)
        if mode == 'val':
            self.data = self.data[:len(self.data)//2]
        elif mode == 'test':
            self.data = self.data[len(self.data)//2:]

        print(f'Loaded {len(self.data)} images in {mode} mode')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        frame_number, image_path, coords_list = data[0], data[1], data[2:]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        coords_list = np.array(coords_list)
        if self.transform is not None:
            data = {
                'image': image,
                'coords_list': coords_list,
            }
            data = self.transform(data)
            image = data['image']
            coords_list = data['coords_list']

        # rescale the ball_coords to the evaluation resolution (HEIGHT, WIDTH)
        H, W, __ = image.shape
        coords_list = [(int((x+0.5) * WIDTH / W - 0.5), int((y+0.5) * HEIGHT / H - 0.5), v) for (x, y, v) in coords_list]
        coords_list = np.array(coords_list)

        # create the heatmaps
        heatmaps = np.zeros((13, HEIGHT, WIDTH), dtype=np.float32)
        for i, (x, y, v) in enumerate(coords_list):
            if v == KEYPOINT_VISIBLE:
                heatmaps[i] = self.create_heatmap((x, y), (HEIGHT, WIDTH), sigma=self.heatmap_sigma)

        # convert to torch
        image = torch.tensor(image).permute(2, 0, 1).float()
        heatmaps = torch.tensor(heatmaps).float()
        coords = torch.tensor(coords_list).float()

        return image, heatmaps, coords

    def create_heatmap(self, ball_coords, image_size, sigma=6):
        # create a gaussian heatmap centered around the ball
        heatmap = np.zeros((image_size[0], image_size[1]))
        ball_x, ball_y = ball_coords
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        heatmap = np.exp(-((x - ball_x) ** 2 + (y - ball_y) ** 2) / (2 * sigma ** 2))
        return heatmap


class TTHQ(torch.utils.data.Dataset):
    def __init__(self, mode, heatmap_sigma, transform=None):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform

        val_test_vids = [1, 3, 10]  # Videos used for validation and testing

        data_path = os.path.join(get_data_path(), 'tthq')
        self.data_path = data_path
        keypoints_dict = pd.read_csv(os.path.join(data_path, 'table_detection.csv'), sep=';').to_dict()
        videos = list(keypoints_dict['video'].values())
        frames = list(keypoints_dict['frame'].values())
        keypoints_x = [keypoints_dict[f'{i:02d}_x'] for i in range(1, 14)]
        keypoints_y = [keypoints_dict[f'{i:02d}_y'] for i in range(1, 14)]
        keypoints_flag = [keypoints_dict[f'{i:02d}_flag'] for i in range(1, 14)]

        data = []  # [(video, frame, (01_x, 01_y, 01_flag), (02_x, 02_y, 02_flag), ...), ...]
        for i, (video, frame) in enumerate(zip(videos, frames)):
            if mode == 'train' and video in val_test_vids:
                continue
            elif mode in ['val', 'test'] and video not in val_test_vids:
                continue
            else:
                keypoints = [(keypoints_x[j][i], keypoints_y[j][i], keypoints_flag[j][i]) for j in range(13)]
                data.append((video, frame, *keypoints))

        # random sort
        rnd = np.random.RandomState(0)
        rnd.shuffle(data)

        # split val and test data (should be the same videos)
        if mode == 'val':
            data = data[:len(data)//2]  # First half for validation
        elif mode == 'test':
            data = data[len(data)//2:]
        self.data = data

        print(f'Loaded {len(self.data)} images in {mode} mode')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get the image path
        stuff = self.data[idx]
        video, frame = stuff[0], stuff[1]
        keypoints = stuff[2:]
        coords_list = [(k[0], k[1], KEYPOINT_VISIBLE) if k[2] == 2 else (k[0], k[1], KEYPOINT_INVISIBLE) for k in keypoints]  # If keypoint is visible, flag is 2

        # get the image path
        image_path = os.path.join(self.data_path, f'{video:02d}', f'{video:02d}_{frame:06d}.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        coords_list = np.array(coords_list)
        if self.transform is not None:
            data = {
                'image': image,
                'coords_list': coords_list,
            }
            data = self.transform(data)
            image = data['image']
            coords_list = data['coords_list']

        # rescale the ball_coords to the evaluation resolution (HEIGHT, WIDTH)
        H, W, __ = image.shape
        coords_list = [(int((x + 0.5) * WIDTH / W - 0.5), int((y + 0.5) * HEIGHT / H - 0.5), v) for (x, y, v) in coords_list]
        coords_list = np.array(coords_list)

        # create the heatmaps
        heatmaps = np.zeros((13, HEIGHT, WIDTH), dtype=np.float32)
        for i, (x, y, v) in enumerate(coords_list):
            if x >= 0 and y >= 0 and v == KEYPOINT_VISIBLE:
                heatmaps[i] = self.create_heatmap((x, y), (HEIGHT, WIDTH), sigma=self.heatmap_sigma)

        # convert to torch
        image = torch.tensor(image).permute(2, 0, 1).float()
        heatmaps = torch.tensor(heatmaps).float()
        coords = torch.tensor(coords_list).float()

        return image, heatmaps, coords

    def create_heatmap(self, ball_coords, image_size, sigma=6):
        # create a gaussian heatmap centered around the ball
        heatmap = np.zeros((image_size[0], image_size[1]))
        ball_x, ball_y = ball_coords
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        heatmap = np.exp(-((x - ball_x) ** 2 + (y - ball_y) ** 2) / (2 * sigma ** 2))
        return heatmap


class BlurBall(torch.utils.data.Dataset):
    def __init__(self, mode, heatmap_sigma, transform=None):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform

        data_path = os.path.join(get_data_path(), 'blurball')

        annotations_path = os.path.join(data_path, 'all_csv_annotations')

        data = []  # [((vid, seq), (frame_path, prev_frame_path, next_frame_path), ((ball_x, ball_y, l, theta), (prev_x, prev_y, prev_l, prev_theta), (next_x, next_y, next_l, next_theta))), ...]
        not_annotated = [13]
        vids = [number for number in range(0, 26) if number not in not_annotated]  # There are 26 videos
        # filter vids based on the mode
        if mode == 'train':
            vids = [number for number in vids if number not in [3, 6, 8, 15, 20, 22, 24]]  # No camera calibration data for video 13
        elif mode == 'val':
            vids = [3, 6, 8, 15, 20, 22, 24]
        elif mode == 'test':
            vids = [3, 6, 8, 15, 20, 22, 24]

        print('Loading the Videos in the dataset...')
        for vid in tqdm(vids):
            vid_path = os.path.join(data_path, f'{vid:02d}')
            sequences = sorted([int(number) for number in os.listdir(os.path.join(vid_path, 'frames')) if os.path.isdir(os.path.join(vid_path, 'frames', number)) and number.isdecimal()])
            for seq in sequences:
                # check if folder is empty and then skip
                if len(os.listdir(os.path.join(vid_path, 'frames'))) == 0:
                    print(f"Skipping {vid:02d} because the folder is empty")
                    continue
                annotation_path = os.path.join(annotations_path, f'{vid:02d}_csv_{seq:03d}.csv')
                annotation_df = pd.read_csv(annotation_path)
                seq_path = os.path.join(vid_path, 'frames', f'{seq:03d}')
                frames = sorted([int(number.strip('.png')) for number in os.listdir(seq_path) if number.endswith('.png')])
                for i, frame in enumerate(frames):
                    # get the image path
                    frame_path = os.path.join(seq_path, f'{frame:05d}.png')

                    # get the ball coordinates
                    ball_x = annotation_df.loc[annotation_df['Frame'] == frame]['X'].values[0]
                    ball_y = annotation_df.loc[annotation_df['Frame'] == frame]['Y'].values[0]

                    # load camera info
                    camera_path = os.path.join(data_path, 'all_calib_files', f'{vid:02d}_table_pose.yaml')
                    with open(camera_path, 'r') as f:
                        camera_info = yaml.safe_load(f)

                    data.append( ((vid, seq), frame_path, camera_info) )
        # random sort
        rnd = np.random.RandomState(0)
        rnd.shuffle(data)
        self.data = data

        # split val and test data (should be the same videos)
        if mode == 'val':
            self.data = self.data[:len(self.data)//2]
            self.data = self.data[::10]  # Validation takes way too much time without this
        elif mode == 'test':
            self.data = self.data[len(self.data)//2:]

        print(f'Loaded {len(self.data)} images in {mode} mode')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        __, image_path, camera_info = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_H, orig_W, __ = image.shape  # Resolution of the image that matches the annotations

        # create camera matrices
        rvec = np.array(camera_info['rvec'], dtype=np.float32)
        tvec = np.array(camera_info['tvec'], dtype=np.float32)
        f = camera_info['f']

        # Camera matrices
        Mint = np.array([
            [f, 0, (orig_W-1) / 2],
            [0, f, (orig_H-1) / 2],
            [0, 0, 1],
        ])
        R, _ = cv2.Rodrigues(rvec)
        Mext = np.eye(4, dtype=np.float32)
        Mext[:3, :3] = R
        Mext[:3, 3] = tvec.flatten()
        # transform from thomas coordinate system into my coordinate system
        trans_matrix = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, -TABLE_HEIGHT],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        Mext = Mext @ trans_matrix
        # reproject table points to image coordinates
        table_img = cam2img(world2cam(np.array(table_points, dtype=np.float32), Mext), Mint)
        coords_list = table_img
        visibility_list = np.full(13, KEYPOINT_VISIBLE)
        coords_list = np.concatenate([coords_list, visibility_list[:, None]], axis=1)

        if self.transform is not None:
            data = {
                'image': image,
                'coords_list': coords_list,
            }
            data = self.transform(data)
            image = data['image']
            coords_list = data['coords_list']
        else:
            print("Warning: No transform applied to the image. Coords now do not match the image.")

        # rescale the ball_coords to the evaluation resolution (HEIGHT, WIDTH)
        H, W, __ = image.shape  # Resolution of the image that is fed to the network
        coords_list = [(int((x + 0.5) * WIDTH / W - 0.5), int((y + 0.5) * HEIGHT / H - 0.5), v) for (x, y, v) in coords_list]
        coords_list = np.array(coords_list)

        # scale the camera matrices  -> Now the matrices match the evaluation resolution (Not really needed, but here for visualization)
        Mint = np.array([
            [f * WIDTH / W, 0, (WIDTH - 1) / 2],
            [0, f * HEIGHT / H, (HEIGHT - 1) / 2],
            [0, 0, 1],
        ], dtype=np.float32)

        # create the heatmaps
        heatmaps = np.zeros((13, HEIGHT, WIDTH), dtype=np.float32)
        for i, (x, y, vis) in enumerate(coords_list):
            if vis == KEYPOINT_VISIBLE:
                heatmaps[i] = self.create_heatmap((x, y), (HEIGHT, WIDTH), sigma=self.heatmap_sigma)

        # convert to torch
        image = torch.tensor(image).permute(2, 0, 1).float()
        heatmaps = torch.tensor(heatmaps).float()
        coords = torch.tensor(coords_list).float()

        return image, heatmaps, coords #, Mint, Mext

    def create_heatmap(self, ball_coords, image_size, sigma=6):
        # create a gaussian heatmap centered around the ball
        heatmap = np.zeros((image_size[0], image_size[1]))
        ball_x, ball_y = ball_coords
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        heatmap = np.exp(-((x - ball_x) ** 2 + (y - ball_y) ** 2) / (2 * sigma ** 2))
        return heatmap




if __name__ == "__main__":
    from tabledetection.transforms import get_transform, plot_transforms
    transform = get_transform('val', (WIDTH, HEIGHT))
    # dataset = TableTennisTable(mode='val', heatmap_sigma=6, transform=transform)
    dataset = TTHQ(mode='val', heatmap_sigma=6, transform=transform)
    # dataset = BlurBall(mode='val', heatmap_sigma=6, transform=transform)
    # overlay the heatmap on the image and plot
    import matplotlib.pyplot as plt

    for i in tqdm(range(len(dataset))):
        image, heatmap, ball_coords = dataset[i]
        # print(image.shape)
        # print(heatmap.shape)
        # print(ball_coords.shape)

        # overlay the heatmap on the image
        image = image.numpy()
        image = plot_transforms({'image': image})['image']
        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        heatmap = heatmap.permute(1, 2, 0).numpy()
        heatmap = np.sum(heatmap, axis=2)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        plt.imshow(image)
        plt.imshow(heatmap, alpha=0.5)
        plt.show()

    pass

