import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import einops as eo
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import cv2
import matplotlib.pyplot as plt

from balldetection.train import get_model as get_ball_model
from tabledetection.train import get_model as get_table_model
from uplifting.model import get_model as get_uplifting_model
from balldetection.transforms import get_transform as get_transforms_ball
from tabledetection.transforms import get_transform as get_transforms_table, plot_transforms
# from uplifting.transformations import get_transforms as get_uplifting_transforms
from balldetection.helper_balldetection import extract_position_torch_gaussian as extract_position_ball
from tabledetection.helper_tabledetection import extract_position_torch_gaussian as extract_position_table
from balldetection.helper_balldetection import HEIGHT, WIDTH, BALL_VISIBLE
from tabledetection.helper_tabledetection import KEYPOINT_VISIBLE, KEYPOINT_INVISIBLE
from balldetection.config import TrainConfig as BallConfig
from tabledetection.config import TrainConfig as TableConfig
from uplifting.config import EvalConfig as UpliftingConfig
from uplifting.helper import transform_rotationaxes
from dataprocessing.regress_cameramatrices import calc_cameramatrices
from uplifting.helper import table_points, cam2img, world2cam, table_connections


from inference.dataset import TTHQ

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def process_trajectory_ball(ball_model, images, move_weights=True):
    '''
    Process one trajectory and return the valid ball positions, their indices and the corresponding times.
    Uses two models to filter out inconsistent predictions.
    Args:
        ball_model: first ball detection model
        images: (1, T, C, H, W) tensor of images for model 1
        move_weights: whether to move the model to the GPU and back to CPU
    Returns:
        pred_positions: (T', 3) array of predictions from model 1 (x, y, v)
    '''
    B, T, C, H, W = images.shape

    pred_positions1 = []
    with torch.no_grad():
        predictions_at_once = 4
        images_flat = images.squeeze(0)  # Expect batch size 1
        ball_model.to(device)
        for start in range(0, T, predictions_at_once):
            end = min(start + predictions_at_once, T)
            input_images = images_flat[start:end].to(device)
            preds, __ = ball_model(input_images)
            del input_images
            pred = extract_position_ball(preds, WIDTH, HEIGHT)
            pred = eo.rearrange(pred, '(b t) d -> b t d', b=B, t=(end - start))
            pred = pred[0]  # B=1
            pred_positions1.append(pred)
        if move_weights: ball_model.to('cpu')

        pred_positions = np.concatenate(pred_positions1, axis=0)  # T X 3

        return pred_positions


def filter_trajectory_ball(pred_positions1, pred_positions2, fps):
    '''
    Filter the ball trajectory using the predictions from two models.
    Args:
        pred_positions1: (T, 3) array of ball positions from model 1 (x, y, v)
        pred_positions2: (T, 3) array of ball positions from model 2 (x, y, v)
        fps: frames per second of the video
    Returns:
        filtered_positions: (T', 2) array of valid ball positions (x, y)
        valid_indices: (T',) array of indices of valid predictions in the original trajectory
        times: (T',) array of times corresponding to the valid predictions
    '''
    THRESHOLD = 20
    T, __ = pred_positions1.shape
    fps = float(fps)

    # Calculate the difference between the two models' predictions
    diff_models = np.linalg.norm(pred_positions1[:, :2] - pred_positions2[:, :2], axis=1)

    valid_predictions, valid_indices, times = [], [], []
    for t in range(T):
        if diff_models[t] > THRESHOLD or pred_positions1[t, 2] != BALL_VISIBLE or pred_positions2[t, 2] != BALL_VISIBLE:
            continue
        time = float(t / fps)
        times.append(time)
        valid_indices.append(t)
        valid_predictions.append(pred_positions1[t])

    valid_predictions = np.array(valid_predictions)[:, :2]  # T' x 2
    valid_indices = np.array(valid_indices)  # T'
    times = np.array(times)  # T'

    return valid_predictions, valid_indices, times


def process_trajectory_table(table_model, images, move_weights=True):
    '''
    Process one trajectory and return the filtered table keypoints.
    Uses two models + Clustering to filter out inconsistent predictions.
    Args:
        table_model: first table detection model
        images: (1, T, C, H, W) tensor of images
        move_weights: whether to move the model to the GPU and back to CPU
    Returns:
        pred_positions: (T, 13, 3) array of table keypoints (x, y, v) for each frame
    '''
    B, T, C, H, W = images.shape

    pred_positions = []
    with torch.no_grad():
        predictions_at_once = 8
        images_flat = images.squeeze(0)  # Expect batch size 1
        table_model.to(device)
        for start in range(0, T, predictions_at_once):
            end = min(start + predictions_at_once, T)
            preds = table_model(images_flat[start:end].to(device))
            pred = extract_position_table(preds, WIDTH, HEIGHT, threshold=0.1)
            pred = eo.rearrange(pred, '(b t) n d -> b t n d', b=B, t=(end - start))
            pred = pred[0]  # B=1
            pred_positions.append(pred)
        if move_weights: table_model.to('cpu')

    pred_positions = np.concatenate(pred_positions, axis=0)  # T X 13 X 2

    return pred_positions


def filter_trajectory_table(pred_positions1, pred_positions2):
    '''
    Filter the table keypoints using the predictions from two models. Additionally, cluster with dbscan.
    Args:
        pred_positions1: (T, 13, 3) array of table keypoints from model 1 (x, y, v)
        pred_positions2: (T, 13, 3) array of table keypoints from model 2 (x, y, v)
    Returns:
        filtered_positions: (13, 3) array of filtered table keypoints (x, y, v)
    '''
    THRESHOLD = 10
    T, __, ___ = pred_positions1.shape

    preds_x = pred_positions1[:, :, 0]
    preds_x2 = pred_positions2[:, :, 0]
    preds_y = pred_positions1[:, :, 1]
    preds_y2 = pred_positions2[:, :, 1]
    preds_v = pred_positions1[:, :, 2]
    preds_v2 = pred_positions2[:, :, 2]

    filtered_positions = []  # shape (13, 3)
    for n in range(preds_x.shape[1]):  # for each keypoint
        valids_x, valids_y = [], []
        for t in range(T):
            if preds_v[t, n] == KEYPOINT_VISIBLE and preds_v2[t, n] == KEYPOINT_VISIBLE:
                dist = np.linalg.norm([preds_x[t, n] - preds_x2[t, n], preds_y[t, n] - preds_y2[t, n]])
                if dist < THRESHOLD:
                    valids_x.append(preds_x[t, n])
                    valids_y.append(preds_y[t, n])

        if len(valids_x) < 3:  # Use a minimum threshold for clustering
            filtered_positions.append([-1, -1, KEYPOINT_INVISIBLE])
        else:
            valid_detections = np.stack([valids_x, valids_y], axis=1)
            filtered_point = _filter_keypoints_with_dbscan(valid_detections, eps=10, min_samples=3)
            if filtered_point is not None:
                filtered_positions.append([filtered_point[0], filtered_point[1], KEYPOINT_VISIBLE])
            else:
                # This case is now handled inside the function, but as a fallback:
                filtered_positions.append([-1, -1, KEYPOINT_INVISIBLE])
                print(f'Keypoint {n + 1} had no stable cluster, setting to invisible')

    filtered_positions = np.array(filtered_positions)

    return filtered_positions  # shape (13, 3)



def _filter_keypoints_with_dbscan(detections, eps=10, min_samples=5):
    """
    Filters a series of 2D keypoint detections using DBSCAN clustering.

    This function takes a history of detections for a single keypoint,
    clusters them based on density, and returns the centroid of the largest cluster.
    This is effective at filtering out noise and outliers from occlusions or
    mis-detections.

    Args:
        detections (np.ndarray): A NumPy array of shape (N, 2) where N is the
                                 number of detections over time. Each row is an (x, y)
                                 coordinate.
        eps (float): The maximum distance between two samples for one to be
                     considered as in the neighborhood of the other. This is the
                     most important parameter to tune.
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point.

    Returns:
        np.ndarray or None: A NumPy array of shape (2,) representing the
                            (x, y) coordinates of the centroid of the largest
                            cluster. Returns None if no clusters are found.
    """
    if not isinstance(detections, np.ndarray):
        detections = np.array(detections)

    if detections.shape[0] < min_samples:
        # Not enough detections to form a reliable cluster, fallback to mean
        return np.mean(detections, axis=0) if detections.shape[0] > 0 else None

    # 1. Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(detections)
    labels = db.labels_

    # 2. Find the largest cluster (ignoring noise label -1)
    valid_labels = [label for label in labels if label != -1]
    if not valid_labels:
        # All points were considered noise, fallback to mean of all points
        return np.mean(detections, axis=0)

    cluster_counts = Counter(valid_labels)
    largest_cluster_label = cluster_counts.most_common(1)[0][0]

    # 3. Calculate the centroid of the largest cluster
    largest_cluster_points = detections[labels == largest_cluster_label]
    filtered_keypoint = np.mean(largest_cluster_points, axis=0)

    return filtered_keypoint


def process_trajectory_uplifting(uplifting_model, predictions_ball, predictions_table, times, mask, transform_mode, move_weights=True):
    '''
    Process one trajectory and return the predicted 3D ball positions and initial spin.
    Args:
        uplifting_model: uplifting model
        predictions_ball: (T', 2) array of valid ball positions (x, y)
        predictions_table: (13, 3) array of table keypoints (x, y, v)
        times: (T',) array of times corresponding to the valid predictions
        mask: (1, seq_len) torch tensor mask indicating valid ball positions
        transform_mode: transformation mode used in the uplifting model, either 'global' or 'local'
        move_weights: whether to move the model to the GPU and back to CPU
    Returns:
        pred_positions_3d: (T', 3) array of predicted 3D ball positions (x, y, z)
        pred_spin: (3,) array of predicted initial spin (omega_x, omega_y, omega_z)
    '''
    with torch.no_grad():
        uplifting_model.to(device)
        predictions_ball, predictions_table, times, mask = predictions_ball.to(device), predictions_table.to(device), times.to(device), mask.to(device)

        pred_spin, pred_positions_3d = uplifting_model(predictions_ball, predictions_table, mask, times)
        if transform_mode == 'global':
            pred_spin = transform_rotationaxes(pred_spin, pred_positions_3d)

        T_prime = int(mask.sum().item())
        pred_positions_3d = pred_positions_3d[0, :T_prime, :].cpu().numpy()  # T' x 3
        pred_spin = pred_spin[0].cpu().numpy()  # 3,

        if move_weights: uplifting_model.to('cpu')
        predictions_ball, predictions_table, times, mask = predictions_ball.to('cpu'), predictions_table.to('cpu'), times.to('cpu'), mask.to('cpu')

    return pred_spin, pred_positions_3d


def _uplifting_transform(ball_coords, table_coords, times):
    '''
    Transforms and normalizes the ball and table coordinates to the input format of the uplifting model.
    Cannot use the standard transformations since there is the resolution mismatch between detection and uplifting model.
    We use seq_len and masking is done to allow for batching of trajectories with different lengths T'.
    Args:
        ball_coords: (T', 2) numpy array of valid ball positions (x, y)
        table_coords: (13, 3) numpy array of table keypoints (x, y, v)
        times: (T',) numpy array of times corresponding to the valid predictions
    Returns:
        ball_coords: (1, seq_len, 2) torch tensor of normalized ball positions with T' out of seq_len valid positions
        table_coords: (1, 13, 3) torch tensor of normalized table keypoints
        times: (1, seq_len) torch tensor of times with T' out of seq_len valid positions
        mask: (1, seq_len) torch tensor mask indicating valid ball positions
    '''
    # Normalize ball coordinates
    ball_coords = ball_coords.copy()
    ball_coords = ball_coords / np.array([WIDTH, HEIGHT])
    ball_coords = torch.tensor(ball_coords, dtype=torch.float32).unsqueeze(0)  # 1 x T' x 2
    # Normalize table coordinates
    table_coords = table_coords.copy()
    table_coords[:, 0] = table_coords[:, 0] / WIDTH
    table_coords[:, 1] = table_coords[:, 1] / HEIGHT
    table_coords = torch.tensor(table_coords, dtype=torch.float32).unsqueeze(0)  # 1 x 13 x 3

    # Increase length to seq_len with padding
    seq_len = 50
    T_prime = ball_coords.shape[1]
    if T_prime < seq_len:
        tmp = torch.zeros((1, seq_len, 2), dtype=torch.float32)
        tmp[:, :T_prime, :] = ball_coords
        ball_coords = tmp
        tmp = torch.zeros((1, seq_len), dtype=torch.float32)
        tmp[:, :T_prime] = torch.tensor(times, dtype=torch.float32).unsqueeze(0)
        times = tmp
        mask = torch.zeros((1, seq_len), dtype=torch.float32)
        mask[:, :T_prime] = 1.0
    else:
        ball_coords = ball_coords[:, :seq_len, :]
        times = torch.tensor(times[:seq_len], dtype=torch.float32).unsqueeze(0)  # 1 x seq_len
        mask = torch.ones((1, seq_len), dtype=torch.float32)
    return ball_coords, table_coords, times, mask


def calibrate_camera(table_coords):
    '''
    Use the detected table keypoints to calibrate the camera and obtain the camera parameters.
    Args:
        table_coords: (13, 3) array of table keypoints (x, y, v)
    Returns:
        M_int: (3, 3) intrinsic camera matrix
        M_ext: (3, 4) extrinsic camera matrix
    '''
    # calculate the camera matrices
    keypoints_dict = {}
    for i, coord in enumerate(table_coords):
        x, y, v = coord
        if v == KEYPOINT_VISIBLE:
            keypoints_dict[i + 1] = [(x, y)]
    M_int, M_ext, num_inliers = calc_cameramatrices(keypoints_dict, resolution=(WIDTH, HEIGHT), use_lm=False, use_ransac=True, use_prints=False)

    return M_int, M_ext






if __name__ == '__main__':
    test_all()