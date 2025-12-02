'''
Script to extract the TTST data of Towards Ball Spin and Trajectory Analysis in Table Tennis Broadcast Videos via Physically Grounded Synthetic-to-Real Transfer.
In the original paper, there is a little mistake. Thus, we use the following code to extract the data from the original annotations.
'''

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import cv2

from dataprocessing.regress_cameramatrices import calc_cameramatrices
from paths import data_path as DATA_PATH

FPS = 50
num_table_points = 13

PATHS = ['01', '02', '03', '04', '05', '06',]
data_path = os.path.join(DATA_PATH, 'tabletennis_annotations')
# save_path = os.path.join(DATA_PATH, 'tabletennis_annotations_processed_corrected')
save_path = os.path.join(DATA_PATH, 'ttst')
os.makedirs(save_path, exist_ok=True)

WIDTH, HEIGHT = 2560, 1440  # original resolution of the videos
print(f'Using original resolution of {WIDTH}x{HEIGHT}. CHeck if this is correct!!!')


def transform_annotations(data_path):
    '''Converts the annotation dataformat to the data format needed for rotation prediction. Also computes camera matrices.
    Args:
        data_path (str): Path to the events and keypoint files
    Returns:
        r_img (list(np.array)): Array of shape (N, T, 2) containing the image coordinates of the ball
        timestamps (list(np.array)): Array of shape (N, T) containing the timestamps of the events
        Mint (list(np.array)): intrinsic camera matrix, shape (N, 3, 3), we assume a static camera per sequence
        Mext (list(np.array)): extrinsic camera matrix, shape (N, 3, 3), we assume a static camera per sequence
    '''
    events = []  # list of tuples (begin, bounce, end)
    spin_classes = []  # 0: no annotation, 1: frontspin, 2: backspin
    ball_list = []  # list of ball coordinates for each event [{frame: (x, y), ...}, ...]
    video_list = []  # list of video, one entry for each entry in ball_list
    table_list = []  # [{point index: [(x, y)]}, ...]
    for path in PATHS:
        # Load the data
        keypoints_df = pd.read_csv(os.path.join(data_path, f'{path}_keypoints.csv'), sep=';', header=1)
        events_df = pd.read_csv(os.path.join(data_path, f'{path}_events.csv'), sep=';', header=1)

        # load all events
        events_local = [] # only used inside of the for loop
        spin_class = 0
        begin, bounce, end = None, None, None
        for frame, event in zip(events_df['frame'], events_df['event']):
            if event == 'Frontspin' and begin is not None:
                spin_class = 1
            elif event == 'Backspin' and begin is not None:
                spin_class = 2

            if event == 'Begin' and bounce is None and end is None:
                begin = frame
            elif event == 'Bounce' and begin is not None and end is None:
                bounce = frame
            elif event == 'End' and begin is not None and bounce is not None:
                end = frame
                if spin_class != 0:
                    events_local.append((begin, bounce, end))
                    spin_classes.append(spin_class)
                spin_class = 0
                begin, bounce, end = None, None, None
        print(f'Loaded {len(events_local)} events')
        events += events_local

        # load all ball keypoints
        ball_dict_tmp = {}
        for frame, ball_x, ball_y, ball_flag in zip(keypoints_df['frame'], keypoints_df['ball_x'], keypoints_df['ball_y'], keypoints_df['ball_flag']):
            if ball_flag == 2:  # ball was annotated
                ball_dict_tmp[frame] = (ball_x, ball_y)
        print(f'Found {len(ball_dict_tmp)} ball keypoints')

        # load all table keypoints
        for event_ind, (begin, __, end) in enumerate(events_local):
            tmp = {}
            for i, frame in enumerate(keypoints_df['frame']):
                if frame >= begin and frame <= end:
                    for point_ind in range(1, num_table_points + 1):
                        if keypoints_df[f'{point_ind:02}_flag'][i] == 2:
                            if point_ind not in tmp: tmp[point_ind] = []
                            tmp[point_ind].append((keypoints_df[f'{point_ind:02}_x'][i], keypoints_df[f'{point_ind:02}_y'][i]))
            table_list.append(tmp)
        print(f'Found {len(table_list)} frames with table keypoints')

        for event_ind, (begin, __, end) in enumerate(events_local):
            tmp = {}
            for frame in range(begin, end + 1):
                if frame not in ball_dict_tmp: print(f'Frame {frame} is missing in event {event_ind}')
                tmp[frame] = ball_dict_tmp[frame]
            ball_list.append(tmp)
            video_list.append(path)

    # get bounces
    bounces = []  # list of bounces for each event [[t_bounce1, ...], [t_bounce1], ...] -> there is always exactly one bounce
    for (begin, bounce, end) in events:
        bounce_time = (bounce - begin) / FPS
        bounces.append(np.array([bounce_time]))

    # compute timestamps and convert ball_list to r_img (I don't use ball_list directly because there copuld be missing/unannotated frames)
    timestamps = [] # list of timestamps for each event [[0, ...], [0, ...], ...]
    r_imgs = []  # list of ball coordinates for each event [[(x, y), ...], [(x, y), ...], ...]
    vidsNframes = []  # [(video_num, [frame1, frame2, ...]), ...]
    for ball_event_dict, video in zip(ball_list, video_list):
        tmp1, tmp2, tmp3 = [], [], []
        time = 0
        start_frame, end_frame = min(ball_event_dict.keys()), max(ball_event_dict.keys())
        for frame in range(start_frame, end_frame + 1):
            if frame in ball_event_dict:
                tmp1.append(time)
                tmp2.append(ball_event_dict[frame])
                tmp3.append(frame)
            time += 1 / FPS
        timestamps.append(np.array(tmp1))
        r_imgs.append(np.array(tmp2))
        vidsNframes.append((video, tmp3))

    # regress camera matrices using table_list -> One matrix per event (TODO: maybe calculate it separately for each frame in the future)
    Mints, Mexts = [], []
    for event_ind, table_annotations in enumerate(tqdm(table_list)):
        Mint, Mext, __ = calc_cameramatrices(table_annotations, resolution=(WIDTH, HEIGHT), use_prints=False, use_lm=False, use_ransac=True)
        Mints.append(Mint)
        Mexts.append(Mext)
    print(f'Computed {len(Mints)} camera matrices')
    print(f'Computed {len(Mexts)} camera matrices')

    return r_imgs, timestamps, Mints, Mexts, bounces, spin_classes, vidsNframes


def extract_images(video_string, frames, path):
    '''Extracts images from the videos and saves them in the corresponding folders.
    Args:
        video_string (str): Video number as string (one element of PATHS)
        frames (list(int)): List of frames to extract
        path (str): Path to the save folder
    '''
    path = os.path.join(path, 'frames')
    os.makedirs(path, exist_ok=True)

    video_path = os.path.join(data_path, f'{video_string}.mp4')
    cap = cv2.VideoCapture(video_path)
    min_frame, max_frame = min(frames), max(frames)
    for frame_idx, frame in enumerate(range(min_frame - 1, max_frame + 2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        if not ret:
            print(f'Could not read frame {frame} from video {video_string}')
            continue
        cv2.imwrite(os.path.join(path, f'{frame_idx:03}.png'), img)






def generate_inference_dataset(data_path, save_path):
    '''Generates the dataset for inference of rotation prediction.
    Args:
        data_path (str): Path to the events and keypoint files
        save_path (str): Path to save the dataset
    '''
    r_imgs, timestamps, Mints, Mexts, bounces, spin_classes, vidsNframes = transform_annotations(data_path)
    for i, (r_img, timestamp, Mint, Mext, bounce, spin_class, vNf) in enumerate(tqdm(zip(r_imgs, timestamps, Mints, Mexts, bounces, spin_classes, vidsNframes))):
        path = os.path.join(save_path, f'trajectory_{i:04}')
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'r_img.npy'), r_img)
        np.save(os.path.join(path, 'times.npy'), timestamp)
        np.save(os.path.join(path, 'Mint.npy'), Mint)
        np.save(os.path.join(path, 'Mext.npy'), Mext)
        np.save(os.path.join(path, 'hits.npy'), bounce)
        np.save(os.path.join(path, 'spin_class.npy'), spin_class)
        extract_images(vNf[0], vNf[1], path)



if __name__ == '__main__':
    # transform_annotations(data_path)
    # generate_inference_dataset(data_path, save_path)
    pass

