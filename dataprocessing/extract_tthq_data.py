'''Script to extract the TTHQ data introduced in this project.'''
import numpy as np
import scipy as sp
import cv2
import pandas as pd
import os
from tqdm import tqdm
import ast

from paths import data_path
from dataprocessing.regress_cameramatrices import calc_cameramatrices


video_names = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
# video_names = ['03',]
additional_video_names = ['101', '102', '103', '201', '202', '203', '204', '205']  # not for trajectories, but for ball and table detections
# additional_video_names = []

video_path = os.path.join(data_path, 'tthq_videos')
annotations_path = os.path.join(video_path, 'tthq_annotations')
save_path = os.path.join(data_path, 'tthq')
os.makedirs(save_path, exist_ok=True)


visible_keypoint = 2  # keypoint flag for visible keypoints
invisible_keypoint = 1  # keypoint flag for unvisible keypoints
unannotated_keypoint = 0  # keypoint flag for unannotated keypoints

inlier_threshold = 6 # minimum number of inliers to accept a camera matrix


def load_trajectories(events_df, video_name, fps):
    # status encodes if it is the first (2 bounces), the last (? bounces) or intermediate trajectory (1 bounce).
    # usable encodes if there was an "unphysical event like net hit. True or False
    # spin_class encodes whether the ball was frontspin (1), backspin (2) or no spin (0)
    trajectories_local = []  # trajectories of this video
    start, end, bounce, spin_class, status, usable = None, None, None, 0, 'intermediate', True
    for frame, event in zip(events_df['frame'], events_df['event']):
        if event == 'Begin':
            start = frame
            status = 'first'
        elif event == 'Hit' and start is None:
            start = frame
        elif event == 'Hit' and start == frame - 1:  # If two consecutive frames were annotated as hit, take the later one as start point
            start = frame
        elif event == 'End':
            end = frame
            status = 'last'
            trajectories_local.append((video_name, start, end, bounce, spin_class, status, usable, fps))
            start, end, bounce = None, None, None
            spin_class = 0
            usable = True
            status = 'intermediate'
        elif event == 'Hit' and start is not None:
            end = frame
            trajectories_local.append((video_name, start, end, bounce, spin_class, status, usable, fps))
            start = frame
            end, bounce = None, None
            spin_class = 0
            usable = True
            status = 'intermediate'
        elif event == 'Bounce':
            if bounce is None:
                bounce = frame
            elif bounce == frame - 1:
                bounce = 0.5 * (bounce + frame)  # If two frames are annotated as bounce, the bounce was in between
        elif event == 'Netz':
            usable = False
        elif event == 'Frontspin':
            spin_class = 1
        elif event == 'Backspin':
            spin_class = 2
    return trajectories_local


def load_ball_keypoints(keypoints_df, video_name):
    ball_detections_local = {}
    for i, frame in enumerate(keypoints_df['frame']):
        ball_x = keypoints_df['ball center_x'][i]
        ball_y = keypoints_df['ball center_y'][i]
        ball_flag = keypoints_df['ball center_flag'][i]
        if ball_flag != 0:
            ball_detections_local[frame] = (ball_x, ball_y, ball_flag)
    return ball_detections_local


def load_table_keypoints(keypoints_df):
    table_detections_local = {}
    for i, frame in enumerate(keypoints_df['frame']):
        tmp_list = []
        frame_is_annotated = True
        for keypoint_num in range(1, 14):
            x = keypoints_df[f'{keypoint_num:02d}_x'][i]
            y = keypoints_df[f'{keypoint_num:02d}_y'][i]
            flag = keypoints_df[f'{keypoint_num:02d}_flag'][i]
            if flag == 0:  # no annotations for this frame
                frame_is_annotated = False
            tmp_list.append((x, y, flag))
        if frame_is_annotated:  # If frame has table keypoint annotations
            table_detections_local[frame] = tmp_list
    return table_detections_local


def extract_data():
    '''Extracts the data from the raw annotations'''
    ball_detections = {}  # {video_name: {frame: (ball_x, ball_y, ball_flag), ...}, ...}
    table_detections = {}  # {video_name: {frame: [(01_x, 01_y, 01_flag), ...], ...}, ...}
    trajectories = [] # [(video_name, start_frame, end_frame, bounce_frame, spin_class, status, usable, fps), ...]

    for __, video_name in tqdm(enumerate(video_names)):
        # Load video
        cap = cv2.VideoCapture(os.path.join(video_path, f'{video_name}_cut.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ORIG_HEIGHT, ORIG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f'Video {video_name} has {fps} fps, {ORIG_WIDTH}x{ORIG_HEIGHT} resolution')

        # Load annotations
        try:
            keypoints_df = pd.read_csv(os.path.join(annotations_path, f'{video_name}_cut_keypoints.csv'), sep=';', header=1)
            events_df = pd.read_csv(os.path.join(annotations_path, f'{video_name}_cut_events.csv'), sep=';', header=1)
        except FileNotFoundError:
            print(f'Annotations for video {video_name} not found. Skipping.')
            continue

        # Load all trajectories.
        trajectories_local = load_trajectories(events_df, video_name, fps)
        trajectories.extend(trajectories_local)
        print(f'Loaded {len(trajectories_local)} trajectories from {video_name}')
        print(f'Loaded {len([e for e in trajectories_local if e[6] == True])} usable trajectories from {video_name}')
        print(f'Loaded {len([e for e in trajectories_local if e[6] == True and e[5] == "intermediate"])} usable intermediate trajectories from {video_name}')

        # Load all table keypoints
        table_detections_local = load_table_keypoints(keypoints_df)
        table_detections[video_name] = table_detections_local
        print(f'Loaded {len(table_detections_local)} frames with table keypoints from {video_name}')

        # Load all ball keypoints
        ball_detections_local = load_ball_keypoints(keypoints_df, video_name)
        ball_detections[video_name] = ball_detections_local
        print(f'Loaded {len(ball_detections_local)} frames with ball keypoints from {video_name}')


    for __, video_name in tqdm(enumerate(additional_video_names)):
        # Load video
        cap = cv2.VideoCapture(os.path.join(video_path, f'{video_name}_cut.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ORIG_HEIGHT, ORIG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_resolution = (ORIG_WIDTH, ORIG_HEIGHT)
        print(f'Video {video_name} has {fps} fps, {ORIG_WIDTH}x{ORIG_HEIGHT} resolution')

        # Load annotations
        try:
            keypoints_df = pd.read_csv(os.path.join(annotations_path, f'{video_name}_cut_keypoints.csv'), sep=';', header=1)
            events_df = pd.read_csv(os.path.join(annotations_path, f'{video_name}_cut_events.csv'), sep=';', header=1)
        except FileNotFoundError:
            print(f'Annotations for video {video_name} not found. Skipping.')
            continue

        # Load all table keypoints
        table_detections_local = load_table_keypoints(keypoints_df)
        table_detections[video_name] = table_detections_local
        print(f'Loaded {len(table_detections_local)} frames with table keypoints from {video_name}')

        # Load all ball keypoints
        ball_detections_local = load_ball_keypoints(keypoints_df, video_name)
        ball_detections[video_name] = ball_detections_local
        print(f'Loaded {len(ball_detections_local)} frames with ball keypoints from {video_name}')

        print('-------')


    print(f'Loaded {len(trajectories)} total trajectories')
    print(f'Loaded {len([e for e in trajectories if e[6] == True])} total usable trajectories')
    print(f'Loaded {len([e for e in trajectories if e[6] == True and e[5] == "intermediate"])} total usable intermediate trajectories')
    print(f'Loaded {len(ball_detections)} videos with ball keypoints')
    print(f'Loaded {len(table_detections)} videos with table keypoints')
    print(f'Loaded {len([f for v in table_detections.values() for f in v.keys()])} total frames with table keypoints')
    print(f'Loaded {len([f for v in ball_detections.values() for f in v.keys()])} total frames with ball keypoints')

    print('calculating camera matrices')
    camera_matrices = {}  # {video_name: {frame: (M_int, M_ext)}}
    for video_name in tqdm(video_names):
        camera_matrices[video_name] = {}
        for frame in tqdm(table_detections[video_name].keys()):
            keypoints_dict = {}  # {keypoint_num: [(x, y), ...]}
            for keypoint_num, value in enumerate(table_detections[video_name][frame]):  # value = (x, y, flag)
                if value[2] == visible_keypoint: # look only at visible keypoints
                    keypoints_dict[keypoint_num + 1] = [(value[0], value[1])]
            M_int, Mext, num_inliers = calc_cameramatrices(keypoints_dict, resolution=orig_resolution, use_lm=False, use_ransac=True, use_prints=True)
            if num_inliers >= inlier_threshold:
                camera_matrices[video_name][frame] = (M_int, Mext)
    print('Finished calculating camera matrices')

    # Save the data
    # Save trajectories
    df = pd.DataFrame(trajectories, columns=['video', 'start_frame', 'end_frame', 'bounce_frame', 'spin_class', 'status', 'usable', 'fps'])
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'trajectories.csv'), index=False, sep=';')

    # Save ball detections
    ball_list = []  # [(video_name, frame, ball_x, ball_y, ball_flag), ...]
    for video_name in ball_detections.keys():
        for frame in ball_detections[video_name].keys():
            ball_x, ball_y, ball_flag = ball_detections[video_name][frame]
            ball_list.append((video_name, frame, ball_x, ball_y, ball_flag))
    df = pd.DataFrame(ball_list, columns=['video', 'frame', 'ball_x', 'ball_y', 'ball_flag'])
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'ball_detection.csv'), index=False, sep=';')

    # Save table detections
    table_list = []  # [(video_name, frame, 01_x, 01_y, 01_flag, ...), ...]
    for video_name in table_detections.keys():
        for frame in table_detections[video_name].keys():
            tmp = [video_name, frame]
            for keypoint in table_detections[video_name][frame]:
                tmp += keypoint
            table_list.append(tmp)
    df = pd.DataFrame(table_list, columns=['video', 'frame'] + [f'{i:02d}_{j}' for i in range(1, 14) for j in ['x', 'y', 'flag']])
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'table_detection.csv'), index=False, sep=';')

    # Save camera matrices
    camera_matrices_list = []  # [(video_name, frame, M_int, M_ext), ...]
    for video_name in camera_matrices.keys():
        for frame in camera_matrices[video_name].keys():
            M_int, M_ext = camera_matrices[video_name][frame]
            camera_matrices_list.append((video_name, frame, M_int.tolist(), M_ext.tolist()))
    df = pd.DataFrame(camera_matrices_list, columns=['video', 'frame', 'M_int', 'M_ext'])
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'camera_matrices.csv'), index=False, sep=';')

    print('Data extraction completed and saved to disk.')

    # Extract relevant frames from videos
    print('Extracting frames from videos...')
    all_frames = {} # {video_name: set of frames, ...}
    for video_name in video_names + additional_video_names:
        all_frames[video_name] = set()
        ball_frames = [tmp[1] for tmp in ball_list if tmp[0] == video_name]
        prev_ball_frames = [tmp[1] - 1 for tmp in ball_list if tmp[0] == video_name]
        next_ball_frames = [tmp[1] + 1 for tmp in ball_list if tmp[0] == video_name]
        table_frames = [tmp[1] for tmp in table_list if tmp[0] == video_name]
        camera_frames = [tmp[1] for tmp in camera_matrices_list if tmp[0] == video_name]
        traj_frames = []
        for stuff in trajectories:
            video, start, end = stuff[0], stuff[1], stuff[2]
            if video == video_name:
                if start is None or end is None or list(range(start, end + 1)) is None or len(list(range(start, end + 1))) == 0:
                    continue
                else:
                    traj_frames.extend(list(range(start, end + 1)))
        all_frames[video_name].update(ball_frames)
        all_frames[video_name].update(prev_ball_frames)
        all_frames[video_name].update(next_ball_frames)
        all_frames[video_name].update(table_frames)
        all_frames[video_name].update(camera_frames)
        all_frames[video_name].update(traj_frames)
        all_frames[video_name] = sorted(all_frames[video_name])

    for video_name in all_frames.keys():
        print(f'Video {video_name} has {len(all_frames[video_name])} relevant frames to extract')
    for video_name in tqdm(all_frames.keys()):
        if len(all_frames[video_name]) == 0:
            continue
        cap = cv2.VideoCapture(os.path.join(video_path, f'{video_name}_cut.mp4'))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        path = os.path.join(save_path, video_name)
        os.makedirs(path, exist_ok=True)
        for frame in tqdm(all_frames[video_name]):
            if frame < 0 or frame >= length:
                print(f'Frame {frame} is out of bounds for video {video_name}. Skipping.')
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, im = cap.read()
            if not ret:
                print(f'Frame {frame} could not be read from video {video_name}')
                continue
            im_path = os.path.join(path, f'{video_name}_{frame:06d}.png')
            cv2.imwrite(im_path, im)
        cap.release()




def visualize_data():
    """
    For each annotated table, visualize the frame, annotated keypoints and reprojected 3D keypoints using our camera matrices
    """
    import matplotlib.pyplot as plt
    from uplifting.helper import table_points, world2cam, cam2img, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH

    points_3d = np.array(table_points)  # 3D points of the table in world coordinates

    keypoints_dict = pd.read_csv(os.path.join(save_path, 'table_detection.csv'), sep=';').to_dict()
    camera_matrices_dict = pd.read_csv(os.path.join(save_path, 'camera_matrices.csv'), sep=';').to_dict()

    videos = list(keypoints_dict['video'].values())
    frames = list(keypoints_dict['frame'].values())
    keypoints_x = [keypoints_dict[f'{i:02d}_x'] for i in range(1, 14)]
    keypoints_y = [keypoints_dict[f'{i:02d}_y'] for i in range(1, 14)]
    keypoints_flag = [keypoints_dict[f'{i:02d}_flag'] for i in range(1, 14)]

    for i, (video, frame) in tqdm(enumerate(zip(videos, frames))):
        # load the image
        img_path = os.path.join(save_path, f'{video:02d}', f'{video:02d}_{frame:06d}.png')
        if not os.path.exists(img_path):
            print(f'Image {img_path} does not exist. Skipping.')
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # draw the annotated keypoints
        for keypoint_num in range(1, 14):
            x = keypoints_x[keypoint_num - 1][i]
            y = keypoints_y[keypoint_num - 1][i]
            flag = keypoints_flag[keypoint_num - 1][i]
            if flag == visible_keypoint:
                cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), -1)


        M_int = None
        M_ext = None
        for j, (f, v) in enumerate(zip(camera_matrices_dict['frame'].values(), camera_matrices_dict['video'].values())):
            if f == frame and v == video:
                M_int = np.array(ast.literal_eval(camera_matrices_dict['M_int'][j]), dtype=np.float32)
                M_ext = np.array(ast.literal_eval(camera_matrices_dict['M_ext'][j]), dtype=np.float32)
                break
        if M_int is not None and M_ext is not None: # If camera matrices are available for this frame
            # reproject the 3D table points to the image
            points_2d = cam2img(world2cam(points_3d, M_ext), M_int)
            points_2d = points_2d.astype(int)

            # draw the reprojected points
            for point in points_2d:
                cv2.circle(img, (point[0], point[1]), 7, (0, 255, 0), -1)

            # draw the coordinate axes
            origin = np.array([0, 0, TABLE_HEIGHT])
            x_axis = np.array([TABLE_LENGTH/2, 0, TABLE_HEIGHT])
            y_axis = np.array([0, TABLE_WIDTH/2, TABLE_HEIGHT])
            z_axis = np.array([0, 0, 1 + TABLE_HEIGHT])

            orig_img = cam2img(world2cam(origin, M_ext), M_int).astype(int)
            x_img = cam2img(world2cam(x_axis, M_ext), M_int).astype(int)
            y_img = cam2img(world2cam(y_axis, M_ext), M_int).astype(int)
            z_img = cam2img(world2cam(z_axis, M_ext), M_int).astype(int)
            cv2.arrowedLine(img, (orig_img[0], orig_img[1]), (x_img[0], x_img[1]), (255, 0, 0), 3)
            cv2.arrowedLine(img, (orig_img[0], orig_img[1]), (y_img[0], y_img[1]), (0, 255, 0), 3)
            cv2.arrowedLine(img, (orig_img[0], orig_img[1]), (z_img[0], z_img[1]), (0, 0, 255), 3)

        # show the image with reprojected points
        plt.imshow(img)
        plt.title(f'Video: {video}, Frame: {frame}')
        plt.axis('off')
        plt.show()






if __name__ == '__main__':
    extract_data()
    # visualize_data()

