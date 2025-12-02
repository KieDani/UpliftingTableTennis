import torch
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from balldetection.helper_balldetection import HEIGHT, WIDTH, get_data_path, BALL_VISIBLE, BALL_INVISIBLE


class TableTennisBall(torch.utils.data.Dataset):
    def __init__(self, mode, heatmap_sigma, in_frames, transform=None, use_invisible=True):
        raise NotImplementedError("This dataset is deprecated. Please use the TTHQ dataset instead. I should implement this dataset using the extracted data from extract_ttst_data.py")
        # Note that use_invisible is not used in this dataset
        assert mode in ['train', 'val', 'test']
        if mode == 'train':
            vids = [1, 3, 4, 6]
        elif mode == 'val':
            vids = [2, 5]
        elif mode == 'test':
            vids = [2, 5]

        data_path = os.path.join(get_data_path(), 'TTST', 'balldetection_data')
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform
        assert in_frames in [1, 3], f"Only in_frames=1 or in_frames=3 is supported, but got {in_frames}"
        self.in_frames = in_frames

        labels_df = pd.read_csv(os.path.join(data_path, 'labels.csv'))

        self.data = [] # [(number in dataname, (image path, prev path, next path), ((ball_x, ball_y), (prev_x, prev_y), (next_x, next_y))), ...]
        for path in os.listdir(data_path):
            if path.endswith('.png'):
                frame_number = int(path.split('.')[0])

                # only use frame if it is not the beginning or end of the sequence
                previous_frame = labels_df.loc[labels_df['number'] == frame_number]['prev_number'].values[0]
                next_frame = labels_df.loc[labels_df['number'] == frame_number]['next_number'].values[0]
                if previous_frame == -1 or next_frame == -1:
                    continue

                # get the image path
                image_path = os.path.join(data_path, path)
                prev_path = os.path.join(data_path, f'{previous_frame:05d}.png')
                next_path = os.path.join(data_path, f'{next_frame:05d}.png')

                # get the ball coordinates
                ball_x = labels_df.loc[labels_df['number'] == frame_number]['x'].values[0]
                ball_y = labels_df.loc[labels_df['number'] == frame_number]['y'].values[0]

                # get the previous and next frame coordinates
                prev_x = labels_df.loc[labels_df['number'] == previous_frame]['x'].values[0]
                prev_y = labels_df.loc[labels_df['number'] == previous_frame]['y'].values[0]
                next_x = labels_df.loc[labels_df['number'] == next_frame]['x'].values[0]
                next_y = labels_df.loc[labels_df['number'] == next_frame]['y'].values[0]

                # check if the video number is in the respective set (train, val, test)
                vid_num = labels_df.loc[labels_df['number'] == frame_number]['vid_num'].values[0]
                if vid_num in vids:
                    self.data.append((frame_number, (image_path, prev_path, next_path), (ball_x, ball_y)))

        #random sort
        rnd = np.random.RandomState(0)
        rnd.shuffle(self.data)

        # split val and test set, since they are the same videos
        if mode == 'val':
            self.data = self.data[:len(self.data) // 2]
        elif mode == 'test':
            self.data = self.data[len(self.data) // 2:]

        print(f'Loaded {len(self.data)} images in {mode} mode')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_number, image_path_tuple, ball_coords_tuple = self.data[idx]
        ball_coords = ball_coords_tuple  # No visibility in this dataset since everything is visible
        image, prev_image, next_image = cv2.imread(image_path_tuple[0]), cv2.imread(image_path_tuple[1]), cv2.imread(image_path_tuple[2])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, prev_image, next_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            data = {
                'image': image,
                'prev_image': prev_image if self.in_frames == 3 else None,
                'next_image': next_image if self.in_frames == 3 else None,
                'ball_coords': ball_coords,
                'visibility': BALL_VISIBLE  # always visible in this dataset
            }
            data = self.transform(data)
            image = data['image']
            prev_image = data['prev_image']
            next_image = data['next_image']
            ball_coords = data['ball_coords']
            vis = data['visibility']  # Ball might not be visible anymore after transformation
        else:
            print("Warning: No transform applied to the image. Coords now do not match the image.")

        # rescale the ball_coords to the evaluation resolution (HEIGHT, WIDTH)
        H, W, __ = image.shape
        ball_x, ball_y = ball_coords
        ball_x, ball_y = ball_x + 0.5, ball_y + 0.5
        ball_x = ball_x * WIDTH / W - 0.5
        ball_y = ball_y * HEIGHT / H - 0.5
        ball_coords = (ball_x, ball_y)

        # heatmap does not have to be in transforms if it is computed after applying transforms
        # The resolution is always the evaluation resolution (HEIGHT, WIDTH)
        heatmap = self.create_heatmap(ball_coords, (HEIGHT, WIDTH), sigma=self.heatmap_sigma)

        # convert to torch
        image = torch.tensor(image).permute(2, 0, 1).float()
        if self.in_frames == 3:
            prev_image = torch.tensor(prev_image).permute(2, 0, 1).float()
            next_image = torch.tensor(next_image).permute(2, 0, 1).float()
            image = torch.concatenate((prev_image, image, next_image), dim=0)
        heatmap = torch.tensor(heatmap).unsqueeze(0).float()
        ball_coords = torch.tensor(ball_coords).float()

        return image, heatmap, ball_coords, ball_coords, ball_coords, vis

    def create_heatmap(self, ball_coords, image_size, sigma=6):
        # create a gaussian heatmap centered around the ball
        heatmap = np.zeros((image_size[0], image_size[1]))
        ball_x, ball_y = ball_coords
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        heatmap = np.exp(-((x - ball_x) ** 2 + (y - ball_y) ** 2) / (2 * sigma ** 2))
        return heatmap



class BlurBall(torch.utils.data.Dataset):
    def __init__(self, mode, heatmap_sigma, in_frames, transform=None, use_invisible=True):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform
        assert in_frames in [1, 3], f"Only in_frames=1 or in_frames=3 is supported, but got {in_frames}"
        self.in_frames = in_frames

        data_path = os.path.join(get_data_path(), 'blurball')

        annotations_path = os.path.join(data_path, 'all_csv_annotations')

        self.data = []  # [((vid, seq), (frame_path, prev_frame_path, next_frame_path), ((ball_x, ball_y, l, theta), (prev_x, prev_y, prev_l, prev_theta), (next_x, next_y, next_l, next_theta))), ...]
        vids = [number for number in range(0, 26)]  # There are 26 videos
        # filter vids based on the mode
        if mode == 'train':
            vids = [number for number in vids if number not in [3, 6, 8, 15, 20, 22, 24]]
        elif mode == 'val':
            vids = [3, 6, 8, 15, 20, 22, 24]
        elif mode == 'test':
            vids = [3, 6, 8, 15, 20, 22, 24]

        print('Loading the Videos in the dataset...')
        if not use_invisible:
            print("Frames with invisible ball are skipped")
        num_vis = 0
        num_invis = 0
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
                    if i == 0 or i == len(frames) - 1:  # skip the first and last frame -> Important if in_frames=3
                        continue
                    prev_frame = frames[i - 1]
                    next_frame = frames[i + 1]

                    # get the image path
                    frame_path = os.path.join(seq_path, f'{frame:05d}.png')
                    prev_path = os.path.join(seq_path, f'{prev_frame:05d}.png')
                    next_path = os.path.join(seq_path, f'{next_frame:05d}.png')

                    # check if ball was visible
                    frame_flag = annotation_df.loc[annotation_df['Frame'] == frame]['Visibility'].values[0]
                    frame_flag = BALL_INVISIBLE if frame_flag == 0 else BALL_VISIBLE
                    prev_flag = annotation_df.loc[annotation_df['Frame'] == prev_frame]['Visibility'].values[0]
                    prev_flag = BALL_INVISIBLE if prev_flag == 0 else BALL_VISIBLE
                    next_flag = annotation_df.loc[annotation_df['Frame'] == next_frame]['Visibility'].values[0]
                    next_flag = BALL_INVISIBLE if next_flag == 0 else BALL_VISIBLE

                    if (not use_invisible) and (frame_flag == BALL_INVISIBLE or prev_flag == BALL_INVISIBLE or next_flag == BALL_INVISIBLE):
                        continue

                    if frame_flag == BALL_VISIBLE:
                        num_vis += 1
                    else:
                        num_invis += 1

                    # get the ball coordinates
                    ball_x = float(annotation_df.loc[annotation_df['Frame'] == frame]['X'].values[0])
                    ball_y = float(annotation_df.loc[annotation_df['Frame'] == frame]['Y'].values[0])
                    l = float(annotation_df.loc[annotation_df['Frame'] == frame]['l'].values[0])
                    theta = float(annotation_df.loc[annotation_df['Frame'] == frame]['theta'].values[0])

                    self.data.append(((vid, seq), (frame_path, prev_path, next_path), (ball_x, ball_y, l, theta, frame_flag)))
        # random sort
        rnd = np.random.RandomState(0)
        rnd.shuffle(self.data)

        # split val and test data (should be the same videos)
        if mode == 'val':
            self.data = self.data[:len(self.data) // 2]
            self.data = self.data[::10]  # Validation takes way too much time without this
        elif mode == 'test':
            self.data = self.data[len(self.data) // 2:]

        self.length = len(self.data)

        print(f'Loaded {len(self.data)} images in {mode} mode')
        print(f"In {mode} set: {num_vis} images with visible ball, {num_invis} images with invisible ball.")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        __, image_path_tuple, ball_coords_tuple = self.data[idx]
        ball_x, ball_y, l, theta, vis = ball_coords_tuple
        ball_coords = (ball_x, ball_y)
        image, prev_image, next_image = cv2.imread(image_path_tuple[0]), cv2.imread(image_path_tuple[1]), cv2.imread(image_path_tuple[2])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, prev_image, next_image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                         cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB),
                                         cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

        orig_H, orig_W, __ = image.shape  # Resolution of the image that matches the annotations

        if self.transform is not None:
            data = {
                'image': image,
                'prev_image': prev_image if self.in_frames == 3 else None,
                'next_image': next_image if self.in_frames == 3 else None,
                'ball_coords': ball_coords,
                'visibility': vis
            }
            data = self.transform(data)
            image = data['image']
            prev_image = data['prev_image']
            next_image = data['next_image']
            ball_coords = data['ball_coords']
            vis = data['visibility']
        else:
            print("Warning: No transform applied to the image. Coords now do not match the image.")

        # rescale the ball_coords to the evaluation resolution (HEIGHT, WIDTH)
        H, W, __ = image.shape
        ball_x, ball_y = ball_coords
        ball_x = (ball_x + 0.5) * WIDTH / W - 0.5
        ball_y = (ball_y + 0.5) * HEIGHT / H - 0.5
        ball_coords = (ball_x, ball_y)

        # calculate the minimum and maximum blur points
        min_x, max_x = ball_x - l * np.cos(np.deg2rad(theta)) * WIDTH / orig_W, ball_x + l * np.cos(np.deg2rad(theta)) * WIDTH / orig_W
        min_y, max_y = ball_y - l * np.sin(np.deg2rad(theta)) * HEIGHT / orig_H, ball_y + l * np.sin(np.deg2rad(theta)) * HEIGHT / orig_H
        # check if the blur points are within the image bounds
        min_x, max_x = max(0, min_x), min(WIDTH - 1, max_x)
        min_y, max_y = max(0, min_y), min(HEIGHT - 1, max_y)

        # heatmap does not have to be in transforms if it is computed after applying transforms
        # The resolution is always the evaluation resolution (HEIGHT, WIDTH)
        if vis == BALL_INVISIBLE:
            heatmap = np.zeros((HEIGHT, WIDTH))
        else:
            heatmap = self.create_heatmap(ball_coords, (HEIGHT, WIDTH), sigma=self.heatmap_sigma)

        # convert to torch
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if self.in_frames == 3:
            prev_image = torch.tensor(prev_image, dtype=torch.float32).permute(2, 0, 1)
            next_image = torch.tensor(next_image, dtype=torch.float32).permute(2, 0, 1)
            image = torch.concatenate((prev_image, image, next_image), dim=0)
        del prev_image, next_image
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)
        ball_coords = torch.tensor(ball_coords, dtype=torch.float32)
        max_coords = torch.tensor((max_x, max_y), dtype=torch.float32)
        min_coords = torch.tensor((min_x, min_y), dtype=torch.float32)

        return image, heatmap, ball_coords, max_coords, min_coords, vis

    def create_heatmap(self, ball_coords, image_size, sigma=6):
        # create a gaussian heatmap centered around the ball
        heatmap = np.zeros((image_size[0], image_size[1]))
        ball_x, ball_y = ball_coords
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        heatmap = np.exp(-((x - ball_x) ** 2 + (y - ball_y) ** 2) / (2 * sigma ** 2))
        return heatmap


class TTHQ(torch.utils.data.Dataset):
    def __init__(self, mode, heatmap_sigma, in_frames, transform=None, use_invisible=True):
        assert mode in ['train', 'val', 'test']
        assert in_frames in [1, 3], f"Only in_frames=1 or in_frames=3 is supported, but got {in_frames}"
        self.in_frames = in_frames
        self.mode = mode
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform

        val_test_vids = [1, 3, 10]  # Videos used for validation and testing

        data_path = os.path.join(get_data_path(), 'tthq')
        self.data_path = data_path
        keypoints_dict = pd.read_csv(os.path.join(data_path, 'ball_detection.csv'), sep=';').to_dict()
        videos = list(keypoints_dict['video'].values())
        frames = list(keypoints_dict['frame'].values())
        ball_x = list(keypoints_dict['ball_x'].values())
        ball_y = list(keypoints_dict['ball_y'].values())
        ball_flag = list(keypoints_dict['ball_flag'].values())

        data = []  # [(video, (frame, prev_frame, next_frame), ((ball_x, ball_y, ball_flag), (prev_x, prev_y, prev_flag), (next_x, next_y, next_flag))), ...]
        num_vis, num_invis = 0, 0
        for i, (video, frame) in enumerate(zip(videos, frames)):
            if mode == 'train' and video in val_test_vids:
                continue
            elif mode in ['val', 'test'] and video not in val_test_vids:
                continue
            # else
            frames_in_video = [f for f, v in zip(frames, videos) if v == video]
            if not self._check_if_frame_exists(video, frame - 1) or not self._check_if_frame_exists(video, frame + 1): # skip to be able to use in_frames=3
                continue
            # get the previous and next frame
            x_in_video = [x for x, v in zip(ball_x, videos) if v == video]
            y_in_video = [y for y, v in zip(ball_y, videos) if v == video]
            flag_in_video = [flag for flag, v in zip(ball_flag, videos) if v == video]

            x = x_in_video[frames_in_video.index(frame)]
            y = y_in_video[frames_in_video.index(frame)]
            flag = flag_in_video[frames_in_video.index(frame)]
            if flag == 2:
                flag = BALL_VISIBLE
                num_vis += 1
            else:  # TODO: Occluded case
                flag = BALL_INVISIBLE
                num_invis += 1

            if (not use_invisible) and (flag == BALL_INVISIBLE):
                # skip invisible frames if USE_INVISIBLE is False
                continue

            data.append((video, (frame, frame - 1, frame + 1), (x, y, flag)))

        # random sort
        rnd = np.random.RandomState(0)
        rnd.shuffle(data)

        # split val and test data (should be the same videos)
        if mode == 'val':
            data = data[:len(data) // 2]  # First half for validation
        elif mode == 'test':
            data = data[len(data) // 2:]
        self.data = data

        print(f'Loaded {len(self.data)} images in {mode} mode')
        print(f"In {mode} set: {num_vis} images with visible ball, {num_invis} images with invisible ball.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video, frame_tuple, ball_coords_tuple = self.data[idx]
        frame, prev_frame, next_frame = frame_tuple
        ball_x, ball_y, vis = ball_coords_tuple  # actual coordinates
        ball_coords = (ball_x, ball_y)
        im_path = os.path.join(self.data_path, f'{video:02d}')
        image = cv2.imread(os.path.join(im_path, f'{video:02d}_{frame:06d}.png'))
        prev_image = cv2.imread(os.path.join(im_path, f'{video:02d}_{prev_frame:06d}.png'))
        next_image = cv2.imread(os.path.join(im_path, f'{video:02d}_{next_frame:06d}.png'))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, prev_image, next_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

        orig_H, orig_W, __ = image.shape  # Resolution of the image that matches the annotations

        if self.transform is not None:
            data = {
                'image': image,
                'prev_image': prev_image if self.in_frames == 3 else None,
                'next_image': next_image if self.in_frames == 3 else None,
                'ball_coords': ball_coords,
                'visibility': vis
            }
            data = self.transform(data)
            image = data['image']
            prev_image = data['prev_image']
            next_image = data['next_image']
            ball_coords = data['ball_coords']
            vis = data['visibility']
        else:
            print("Warning: No transform applied to the image. Coords now do not match the image.")

        # rescale the ball_coords to the evaluation resolution (HEIGHT, WIDTH)
        H, W, __ = image.shape
        ball_x, ball_y = ball_coords
        ball_x = (ball_x + 0.5) * WIDTH / W - 0.5
        ball_y = (ball_y + 0.5) * HEIGHT / H - 0.5
        ball_coords = (ball_x, ball_y)

        # calculate the minimum and maximum blur points
        # We skip this now, maybe add later
        max_x, min_x = ball_x, ball_x
        max_y, min_y = ball_y, ball_y

        # heatmap does not have to be in transforms if it is computed after applying transforms
        # The resolution is always the evaluation resolution (HEIGHT, WIDTH)
        if vis == BALL_INVISIBLE:
            heatmap = np.zeros((HEIGHT, WIDTH))
        else:
            heatmap = self.create_heatmap(ball_coords, (HEIGHT, WIDTH), sigma=self.heatmap_sigma)

        # convert to torch
        image = torch.tensor(image).permute(2, 0, 1).float()
        if self.in_frames == 3:
            prev_image = torch.tensor(prev_image).permute(2, 0, 1).float()
            next_image = torch.tensor(next_image).permute(2, 0, 1).float()
            image = torch.concatenate((prev_image, image, next_image), dim=0)
        heatmap = torch.tensor(heatmap).unsqueeze(0).float()
        ball_coords = torch.tensor(ball_coords).float()
        max_coords = torch.tensor((max_x, max_y)).float()
        min_coords = torch.tensor((min_x, min_y)).float()

        return image, heatmap, ball_coords, max_coords, min_coords, vis

    def create_heatmap(self, ball_coords, image_size, sigma=6):
        # create a gaussian heatmap centered around the ball
        heatmap = np.zeros((image_size[0], image_size[1]))
        ball_x, ball_y = ball_coords
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        heatmap = np.exp(-((x - ball_x) ** 2 + (y - ball_y) ** 2) / (2 * sigma ** 2))
        return heatmap

    def _check_if_frame_exists(self, video, frame):
        """
        Check if the frame exists in the video.
        """
        im_path = os.path.join(self.data_path, f'{video:02d}')
        return os.path.exists(os.path.join(im_path, f'{video:02d}_{frame:06d}.png'))


if __name__ == "__main__":

    from balldetection.transforms import Resize
    transform = Resize((WIDTH, HEIGHT))
    in_frames = 3  # Change to 1 if you want to use single frame images
    # dataset = TableTennisBall(mode='val', heatmap_sigma=6, in_frames=in_frames, transform=transform)
    # dataset = BlurBall(mode='val', heatmap_sigma=6, in_frames=in_frames, transform=transform)
    dataset = TTHQ(mode='val', heatmap_sigma=6, in_frames=in_frames, transform=transform)
    # overlay the heatmap on the image and plot
    import matplotlib.pyplot as plt
    for i in tqdm(range(len(dataset))):
        image, heatmap, ball_coords, max_coords, min_coords, vis = dataset[i]
        if in_frames == 3:
            image = image[3:6, :, :]
        if torch.linalg.norm(max_coords - min_coords) > 25:
            print("Max coords and min coords are too far apart")
            continue
        if i < 10:
            #print(image.shape)
            # draw a circle with cv2 at the ball coordinates, max_coords and min_coords
            image = image.permute(1, 2, 0).numpy().astype(np.uint8).copy()
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ball_coords = ball_coords.numpy()
            max_coords = max_coords.numpy()
            min_coords = min_coords.numpy()
            if vis == BALL_VISIBLE:
                cv2.circle(image, (int(ball_coords[0]), int(ball_coords[1])), 10, (0, 255, 0), -1)
                cv2.circle(image, (int(max_coords[0]), int(max_coords[1])), 10, (255, 0, 0), -1)
                cv2.circle(image, (int(min_coords[0]), int(min_coords[1])), 10, (0, 0, 255), -1)
            else:
                cv2.putText(image, "Keypoints INVISIBLE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), thickness=4)
            # plot the image in high resolution
            fig = plt.figure(dpi=300)
            plt.imshow(image)
            plt.show()

            # plot the heatmap
            heatmap = heatmap.squeeze().numpy()
    pass
