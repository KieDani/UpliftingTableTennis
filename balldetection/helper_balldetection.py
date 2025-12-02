import os
import torch
import torch.nn.functional as F
import random
from scipy.optimize import minimize
import numpy as np

from paths import data_path as DATA_PATH
from paths import logs_path as LOGS_PATH


HEIGHT, WIDTH = 1080, 1920  # Original image size -> We evaluate on this size and calculate the metrics based on this size
THRESHOLD = -float('inf')  # Threshold for the heatmap to consider a point as a valid detection

BALL_VISIBLE = 1
BALL_OCCLUDED = -1
BALL_INVISIBLE = 0


def get_logs_path():
    path = os.path.join(LOGS_PATH, 'balldetection')
    return path


def get_data_path():
    path = os.path.join(DATA_PATH)
    return path

def extract_position_torch_gaussian(heatmaps, image_width, image_height):
    """
    Extract the subpixel position of the ball from the heatmaps (batch) using a 2D gaussian fit with torchmin.
    Position is scaled to the original image size.
    Handles image border cases with padding.
    Args:
        heatmaps (torch.Tensor): The heatmaps from which to extract the positions. Shape should be (B, H, W).
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        threshold (float): The threshold for the heatmap to consider a point as a valid detection.
    Returns:
        np.ndarray: An array of shape (B, 2) containing the (x, y) image coordinates for each heatmap in the batch.
    """
    if len(heatmaps.shape) == 4:  # If the heatmaps have an additional channel dimension, squeeze it out
        heatmaps = heatmaps.squeeze(1)
    if len(heatmaps.shape) != 3:
        raise ValueError("Heatmaps must have shape (B, H, W)")

    batch_size, heatmap_height, heatmap_width = heatmaps.shape

    # Find the maximum points in the heatmaps
    max_indices = torch.argmax(heatmaps.view(batch_size, -1), dim=1)
    y_max = max_indices // heatmap_width
    x_max = max_indices % heatmap_width

    # Extract small windows around the maximum points with padding
    window_size = 3
    pad = window_size // 2
    padded_heatmaps = F.pad(heatmaps, (pad, pad, pad, pad), mode='constant', value=0)

    y_max_padded = y_max + pad
    x_max_padded = x_max + pad

    windows = torch.stack(
        [padded_heatmaps[b, y_max_padded[b] - pad:y_max_padded[b] + pad + 1, x_max_padded[b] - pad:x_max_padded[b] + pad + 1] for b in
         range(batch_size)]).cpu().numpy()

    # Create grid for gaussian fitting
    y_window, x_window = np.meshgrid(np.arange(windows.shape[1]), np.arange(windows.shape[2]), indexing='ij')
    xy_window = np.stack((x_window.flatten(), y_window.flatten()))

    def gaussian_2d_loss(params, xy, window):
        x0, y0, sigma_x, sigma_y = params
        x, y = xy
        gaussian = np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))
        return np.mean((gaussian - window.flatten()) ** 2)

    positions = np.zeros((batch_size, 3))

    for b in range(batch_size):

        # check if the heatmap is significant enough to be considered as detection
        activation = heatmaps[b, y_max[b], x_max[b]].item()
        visibility = BALL_VISIBLE if activation > THRESHOLD else BALL_INVISIBLE

        params_init = np.array([windows.shape[2] // 2, windows.shape[1] // 2, 1.0, 1.0], dtype=np.float32)
        bounds = [(0, windows.shape[2]), (0, windows.shape[1]), (0.5, 50), (0.5, 50)]
        # result = torchmin.minimize(lambda params: gaussian_2d_loss(params, xy_window, windows[b]), params_init, method='bfgs')
        result = minimize(lambda params: gaussian_2d_loss(params, xy_window, windows[b]), params_init, method='L-BFGS-B', bounds=bounds)
        if result.success:
            x_offset, y_offset = result.x[0], result.x[1]
        else:
            # print('fitting failed, using max position')
            y_com, x_com = np.where(windows[b] == windows[b].max())
            x_offset = x_com.float().mean()
            y_offset = y_com.float().mean()

        x_subpixel = x_max[b].float().cpu().numpy() - window_size // 2 + x_offset
        y_subpixel = y_max[b].float().cpu().numpy() - window_size // 2 + y_offset

        positions[b] = np.array([x_subpixel, y_subpixel, visibility])

    # Scale from heatmap coordinates to image coordinates, accounting for pixel centers
    scale_x = image_width / heatmap_width
    scale_y = image_height / heatmap_height
    # Adjust coordinates for pixel centers before scaling
    centered_positions_x = positions[:, 0] + 0.5
    centered_positions_y = positions[:, 1] + 0.5
    image_x = centered_positions_x * scale_x - 0.5
    image_y = centered_positions_y * scale_y - 0.5

    return np.stack([image_x, image_y, positions[:, 2]], axis=1)


# def taylor_refine_torch(heatmap, coord):
#     """
#     DARK-style Taylor expansion to refine keypoint coordinate.
#     Args:
#         heatmap: (H, W) tensor
#         coord: tensor of shape (2,) [x, y]
#     Returns:
#         Refined coordinate: (2,) tensor [x, y]
#     """
#     H, W = heatmap.shape
#     px, py = int(coord[0]), int(coord[1])
#
#     if not (1 <= px < W - 1 and 1 <= py < H - 1):
#         return coord  # Skip refinement if too close to border
#
#     # Log transform
#     patch = heatmap[py - 1:py + 2, px - 1:px + 2].clone()
#     patch = torch.clamp(patch, min=1e-10)
#     patch = patch.log()
#
#     dx = 0.5 * (patch[1, 2] - patch[1, 0])
#     dy = 0.5 * (patch[2, 1] - patch[0, 1])
#     dxx = 0.25 * (patch[1, 2] - 2 * patch[1, 1] + patch[1, 0])
#     dyy = 0.25 * (patch[2, 1] - 2 * patch[1, 1] + patch[0, 1])
#     dxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])
#
#     H_mat = torch.tensor([[dxx, dxy], [dxy, dyy]], device=patch.device)
#     g = torch.tensor([-dx, -dy], device=patch.device)
#
#     if torch.det(H_mat) != 0:
#         offset = torch.linalg.solve(H_mat, g)
#         coord += offset
#
#     return coord
#
#
# def extract_position_torch_dark(heatmaps, image_width, image_height, threshold=THRESHOLD):
#     """
#     Extract subpixel keypoint position from heatmaps using DARK Taylor refinement.
#     Args:
#         heatmaps (torch.Tensor): Shape (B, H, W)
#         image_width (int): Width of original image
#         image_height (int): Height of original image
#         threshold (float): Threshold for heatmap activation
#     Returns:
#         np.ndarray: (B, 2) coordinates in image space
#     """
#     if len(heatmaps.shape) == 4:  # (B, 1, H, W)
#         heatmaps = heatmaps.squeeze(1)
#     if len(heatmaps.shape) != 3:
#         raise ValueError("Heatmaps must have shape (B, H, W)")
#
#     B, H, W = heatmaps.shape
#
#     # Apply Gaussian blur if needed (optional)
#     heatmaps = torch.clamp(heatmaps, min=1e-10)
#     heatmaps_log = heatmaps.log()
#
#     # Get coarse max locations
#     max_vals, max_idxs = torch.max(heatmaps.view(B, -1), dim=1)
#     y = (max_idxs // W).int()
#     x = (max_idxs % W).int()
#
#     coords = torch.stack([x.float(), y.float()], dim=1)
#
#     # Refine each heatmap individually
#     refined_coords = []
#     for i in range(B):
#
#         # Check if the activation is above the threshold
#         activation = heatmaps[i, y[i], x[i]].item()
#         if activation < threshold:
#             refined_coords.append(torch.tensor([-10000, -10000], dtype=torch.float32))
#             continue
#
#         refined = taylor_refine_torch(heatmaps_log[i], coords[i])
#         refined_coords.append(refined)
#
#     refined_coords = torch.stack(refined_coords, dim=0)  # [B, 2]
#
#     # Adjust for pixel center
#     refined_coords += 0.5
#
#     # Scale to image space
#     scale_x = image_width / W
#     scale_y = image_height / H
#     refined_coords[:, 0] = refined_coords[:, 0] * scale_x - 0.5
#     refined_coords[:, 1] = refined_coords[:, 1] * scale_y - 0.5
#
#     return refined_coords.cpu().numpy()


def calculate_pck_fixed_tolerance(preds, gts, gts_min, gts_max, tolerance_pixels):
    """
    Calculates Percentage of Correct Keypoints (PCK) for ball detection using a fixed pixel tolerance.
    Assumes one prediction per ground truth.
    Args:
        preds (ndarray): ndarray of predicted ball positions (x, y).
        gts (ndarray): ndarray of ground truth ball positions (x, y).
        gts_min (ndarray): ndarray of gt blur minimum annotations (x, y).
        gts_max (ndarray): ndarray of gt blur maximum annotations (x, y).
        tolerance_pixels (int): Fixed tolerance in pixels for a prediction to be correct.
    Returns:
        float: Percentage of Correct Keypoints (PCK).
    """
    if len(preds) != len(gts):
        raise ValueError("Number of predictions must equal number of ground truths.")

    valid_detections = preds[..., 2] == BALL_VISIBLE
    # if no detection is valid
    if not np.any(valid_detections):
        return -1

    # Calculate Euclidean distance to the steak
    # distances = np.sqrt(np.sum((preds - gts) ** 2, axis=-1))
    distances_to_segment1 = _batched_distance_point_to_segment(preds[..., :2], gts_min[..., :2], gts[..., :2])  # first segment (r_min_batch to r_b_batch)
    distances_to_segment2 = _batched_distance_point_to_segment(preds[..., :2], gts[..., :2], gts_max[..., :2])  # second segment (r_b_batch to r_max_batch)
    distances = np.minimum(distances_to_segment1, distances_to_segment2)  # choose shortest distance to the closer streak

    # Determine correctness for each keypoint in each sample (N, C)
    is_correct = (distances <= tolerance_pixels) & valid_detections

    # Calculate average and standard deviation over keypoints (C)
    average_pck = np.sum(is_correct) / np.sum(valid_detections)

    return float(average_pck)


def average_distance(predictions, ground_truths):
    """
    Calculates the average Euclidean distance between predicted and ground truth ball positions.
    Assumes a one-to-one correspondence between predictions and ground truths.

    Args:
        predictions (list or np.ndarray): List or array of predicted ball positions (x, y).
                                          Shape should be (N, 2), where N is the number of samples.
        ground_truths (list or np.ndarray): List or array of ground truth ball positions (x, y).
                                           Shape should be (N, 2), where N is the number of samples.

    Returns:
        float: The average Euclidean distance between the predictions and ground truths.
               Returns NaN if the input lists are empty or have mismatched lengths.
    """
    if len(predictions) != len(ground_truths):
        return np.nan

    preds = np.array(predictions, dtype=np.float32)
    gts = np.array(ground_truths, dtype=np.float32)

    valid_detections = (preds[..., 0] > -100) & (preds[..., 1] > -100)
    # if no detection is valid
    if not np.any(valid_detections):
        return 10000

    distances = np.sqrt(np.sum((preds - gts)**2, axis=-1))
    distances_mask = np.where(valid_detections, 1, 0)
    distances = distances * distances_mask

    # Calculate average and standard deviation over keypoints (C)
    average_avg_dist = np.sum(distances) / np.sum(distances_mask)

    return float(average_avg_dist)


def ratio_visible_detected(predictions):
    '''Calculates the ratio of valid keypoints among visible keypoints (visible in GT). Keypoint is valid if maximum in heatmap is significant.

    Args:
        predictions (list or np.ndarray): List or array of predicted ball positions (x, y).
                                          Shape should be (N, 2), where N is the number of samples.
    '''
    preds = np.array(predictions, dtype=np.float32)

    valid_detections = (preds[..., 0] > -100) & (preds[..., 1] > -100) & (preds[..., 2] == BALL_VISIBLE)
    distances_mask = np.where(valid_detections, 1, 0)
    detected_keypoints = np.sum(distances_mask)
    all_keypoints = np.prod(distances_mask.shape)
    ratio = detected_keypoints / all_keypoints

    return ratio


# def acc_visible_invisible_keypoints(predictions, label_vis):
#     thresholds = np.arange(1, 10) / 10
#     ap_vis, ap_invis = [], []
#
#     gt_vis = label_vis == BALL_VISIBLE
#     gt_invis = label_vis == BALL_INVISIBLE
#     for threshold in thresholds:
#         preds_vis = predictions[:, 2] >= threshold
#         preds_invis = predictions[:, 2] < threshold
#
#         correct_vis = np.logical_and(preds_vis, gt_vis)
#         correct_invis = np.logical_and(preds_invis, gt_invis)
#
#         ap_vis.append(np.sum(correct_vis) / np.sum(gt_vis))
#         if np.sum(gt_invis) > 0:
#             ap_invis.append(np.sum(correct_invis) / np.sum(gt_invis))
#         else:
#             ap_invis.append(0)
#
#     return ap_vis, ap_invis, np.sum(gt_vis), np.sum(gt_invis), thresholds


def acc_visible_invisible_keypoints(predictions, label_vis):
    gt_vis = label_vis == BALL_VISIBLE
    gt_invis = label_vis == BALL_INVISIBLE
    preds_vis = predictions[:, 0] <= predictions[:, 1]
    preds_invis = predictions[:, 0] > predictions[:, 1]

    correct_vis = np.logical_and(preds_vis, gt_vis)
    correct_invis = np.logical_and(preds_invis, gt_invis)

    ap_vis = np.sum(correct_vis) / np.sum(gt_vis)
    if np.sum(gt_invis) > 0:
        ap_invis = np.sum(correct_invis) / np.sum(gt_invis)
    else:
        ap_invis = 0

    return ap_vis, ap_invis


def _batched_distance_point_to_segment(P_batch: np.ndarray,
                                       E1_batch: np.ndarray,
                                       E2_batch: np.ndarray) -> np.ndarray:
    """
    Calculates the shortest distance from a batch of points to a batch of line segments.

    Args:
        P_batch (np.ndarray): Batch of points, shape (N, D), where N is the batch size
                              and D is the dimensionality (e.g., 2 for 2D points).
        E1_batch (np.ndarray): Batch of first endpoints of the segments, shape (N, D).
        E2_batch (np.ndarray): Batch of second endpoints of the segments, shape (N, D).

    Returns:
        np.ndarray: Batch of shortest distances, shape (N,).
    """
    if P_batch.ndim == 1:  # Single point, not a batch
        P_batch = P_batch.reshape(1, -1)
        E1_batch = E1_batch.reshape(1, -1)
        E2_batch = E2_batch.reshape(1, -1)

    if P_batch.shape[1] != E1_batch.shape[1] or P_batch.shape[1] != E2_batch.shape[1]:
        raise ValueError("Dimensionality of P, E1, and E2 must match.")

    # Vector from E1 to E2 for each segment in the batch
    # segment_vec_batch.shape: (N, D)
    segment_vec_batch = E2_batch - E1_batch

    # Squared length of each segment.
    # L_squared_batch.shape: (N,)
    # Add a small epsilon for numerical stability if E1 and E2 are identical.
    L_squared_batch = np.sum(segment_vec_batch ** 2, axis=1)

    # Vector from E1 to P for each point-segment pair
    # point_vec_batch.shape: (N, D)
    point_vec_batch = P_batch - E1_batch

    # Projection parameter t for each pair: t = dot(P - E1, E2 - E1) / ||E2 - E1||^2
    # We need to handle the case where L_squared_batch is zero (segment is a point).
    # dot_product_batch.shape: (N,)
    dot_product_batch = np.sum(point_vec_batch * segment_vec_batch, axis=1)

    # Initialize t_batch. For zero-length segments, t effectively is 0
    # as the closest point is E1 itself.
    t_batch = np.zeros_like(L_squared_batch)

    # Avoid division by zero for non-zero length segments
    non_zero_length_mask = L_squared_batch > 1e-12  # A small epsilon
    t_batch[non_zero_length_mask] = dot_product_batch[non_zero_length_mask] / L_squared_batch[non_zero_length_mask]

    # Clamp t to the range [0, 1] to find the point on the segment closest to P.
    # If t < 0, closest point is E1.
    # If t > 1, closest point is E2.
    # Otherwise, it's the projection E1 + t * (E2 - E1).
    t_clamped_batch = np.clip(t_batch, 0, 1)

    # Calculate the closest point on each segment to P
    # closest_point_batch.shape: (N, D)
    # We use t_clamped_batch[:, np.newaxis] to enable broadcasting with segment_vec_batch
    closest_point_batch = E1_batch + t_clamped_batch[:, np.newaxis] * segment_vec_batch

    # Calculate the Euclidean distance from P to the closest point on the segment
    # distances_batch.shape: (N,)
    distances_batch = np.linalg.norm(P_batch - closest_point_batch, axis=1)

    return distances_batch


def distance_to_streak(
        r_pred_batch: np.ndarray,
        r_min_batch: np.ndarray,
        r_b_batch: np.ndarray,
        r_max_batch: np.ndarray) -> np.ndarray:
    """
    Calculates the shortest distance from a batch of predicted ball positions (r_pred)
    to their corresponding motion blur streaks. A streak is defined by two line
    segments: (r_min, r_b) and (r_b, r_max).

    Args:
        r_pred_batch (np.ndarray): Batch of predicted ball center positions.
                                   Shape: (N, D), where N is batch size, D is dimensionality (e.g., 2 for 2D).
        r_min_batch (np.ndarray): Batch of ground truth start points of the motion blur.
                                  Shape: (N, D).
        r_b_batch (np.ndarray): Batch of ground truth ball center positions (mid-point of streak).
                                Shape: (N, D).
        r_max_batch (np.ndarray): Batch of ground truth end points of the motion blur.
                                  Shape: (N, D).

    Returns:
        np.ndarray: An array of shortest distances for each item in the batch.
                    Shape: (N,).
    """
    # convert everything to numpy arrays
    r_pred_batch = np.asarray(r_pred_batch)
    r_min_batch = np.asarray(r_min_batch)
    r_b_batch = np.asarray(r_b_batch)
    r_max_batch = np.asarray(r_max_batch)

    # Check if all input arrays have the same shape
    if not (r_pred_batch.shape == r_min_batch.shape == r_b_batch.shape == r_max_batch.shape):
        raise ValueError("All input arrays must have the same shape.")
    if r_pred_batch.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional (batch_size, num_coordinates).")

    # Check which detections are valid (max in heatmap was significant enough)
    valid_detections = (r_pred_batch[..., 0] > -100) & (r_pred_batch[..., 1] > -100)
    # if no detection is valid
    if not np.any(valid_detections):
        return 10000

    # Calculate distances to the first segment (r_min_batch to r_b_batch)
    distances_to_segment1 = _batched_distance_point_to_segment(r_pred_batch, r_min_batch, r_b_batch)

    # Calculate distances to the second segment (r_b_batch to r_max_batch)
    distances_to_segment2 = _batched_distance_point_to_segment(r_pred_batch, r_b_batch, r_max_batch)

    # The final metric is the minimum of the distances to the two segments
    shortest_distances = np.minimum(distances_to_segment1, distances_to_segment2)
    distances_mask = np.where(valid_detections, 1, 0)
    shortest_distances = shortest_distances * distances_mask

    # calculate average distance
    shortest_distance = np.sum(shortest_distances) / np.sum(distances_mask)

    return shortest_distance



def update_ema(model, model_ema, alpha=0.95):
    '''Update the EMA model with the current model.
    Args:
        model (torch.nn.Module): current model
        model_ema (torch.nn.Module): EMA model
        alpha (float): EMA decay factor
    Returns:
        model_ema (torch.nn.Module): updated EMA model
    '''
    with torch.no_grad():
        for name, param in model_ema.named_parameters():
            param.data = alpha * param + (1 - alpha) * model.state_dict()[name].data
        for name, param in model_ema.named_buffers():
            param.data = alpha * param + (1 - alpha) * model.state_dict()[name].data
        return model_ema


def weighted_mse_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Weighted Mean Squared Error (MSE) loss.

    Args:
        input: The predicted output tensor (e.g., predicted heatmaps).
               Shape: (batch_size, channels, height, width) or similar.
        target: The ground truth tensor (e.g., target Gaussian heatmaps).
                Must have the same shape as input.
    """
    # Calculate squared difference
    squared_error = (input - target) ** 2

    # weight = 100 for each value larger than 0.1 in the target, else 1
    weight = torch.where(target > 0.1, torch.ones_like(target) * 100, torch.ones_like(target))

    # Apply weights element-wise
    weighted_squared_error = weight * squared_error

    # Calculate the mean over all elements (batch, channels, spatial dims)
    loss = torch.mean(weighted_squared_error)

    return loss


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_model(model, config, epoch):
    '''Saves the model weights and additional information about the run
    Args:
        model (nn.Module): model to save
        config (MyConfig): configuration object
        epoch (int): epoch number
    '''
    save_path = config.saved_models_path
    os.makedirs(save_path, exist_ok=True)
    h_params = config.get_hparams()
    additional_info = {
        'epoch': epoch,
        **h_params
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'identifier': config.ident,
        'additional_info': additional_info
    }, os.path.join(save_path, f'model.pt'))



if __name__ == "__main__":
    # Example usage
    heatmap = torch.zeros(2, 5, 5)  # Example heatmap
    heatmap[0, 2, 2] = 1.0  # Example ball position in the heatmap
    heatmap[0, 3, 2] = 0.8  # Example ball position in the heatmap
    heatmap[1, 4, 1] = 1.0  # Example ball position in the heatmap
    image_width = 5
    image_height = 5
    positions_gaussian = extract_position_torch_gaussian(heatmap, image_width, image_height)
    print(positions_gaussian)
    positions_dark = extract_position_torch_dark(heatmap, image_width, image_height)
    print(positions_dark)
    # Expected output: tensor([[x1, y1], [x2, y2]])
    # where x1, y1 and x2, y2 are the subpixel coordinates of the ball in the original image

    print('-------------------')

    # Example usage (with dummy data):
    predictions = [(100, 150), (120, 160), (200, 250)]
    ground_truths = [(105, 155), (205, 255), (200, 251)]
    tolerance = 10
    pck = calculate_pck_fixed_tolerance(predictions, ground_truths, tolerance)
    print(f"PCK: {pck:.2f}%")


    print('-------------------')
    # Example usage (with dummy data):
    predictions = [(100, 150), (120, 160), (200, 250)]
    ground_truths = [(105, 155), (205, 255), (200, 251)]
    avg_dist = average_distance(predictions, ground_truths)
    print(f"Average Distance: {avg_dist:.2f} pixels")


