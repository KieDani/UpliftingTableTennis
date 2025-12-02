import os
import torch
import torch.nn.functional as F
import random
#import torchmin
from scipy.optimize import minimize
import numpy as np
import einops as eo

from paths import data_path as DATA_PATH
from paths import logs_path as LOGS_PATH


HEIGHT, WIDTH = 1080, 1920  # Original image size -> We evaluate on this size and calculate the metrics based on this size
THRESHOLD = 0.1  # Threshold for the heatmap to consider a point as a valid detection

TABLE_HEIGHT = 0.76
TABLE_WIDTH = 1.525
TABLE_LENGTH = 2.74

table_points = np.array([
    [-TABLE_LENGTH/2, TABLE_WIDTH/2, TABLE_HEIGHT], # 0 close left
    [-TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT], # 1 close right
    [0.0, TABLE_WIDTH/2, TABLE_HEIGHT], # 2 center left
    [0.0, -TABLE_WIDTH/2, TABLE_HEIGHT], # 3 center right
    [TABLE_LENGTH/2, TABLE_WIDTH/2, TABLE_HEIGHT], # 4 far left
    [TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT], # 5 far right
    [0.0, TABLE_WIDTH/2+0.1525, TABLE_HEIGHT], # 6 net left bottom
    [0.0, -(TABLE_WIDTH/2+0.1525), TABLE_HEIGHT], # 7 net right bottom
    [0.0, 0.0, TABLE_HEIGHT], # 8 net center bottom
    [0.0, TABLE_WIDTH/2+0.1525, TABLE_HEIGHT+0.1525], # 9 net left top
    [0.0, -(TABLE_WIDTH/2+0.1525), TABLE_HEIGHT+0.1525], # 10 net right top
    [-TABLE_LENGTH/2, 0, TABLE_HEIGHT], # 11 close center
    [TABLE_LENGTH/2, 0, TABLE_HEIGHT], # 12 far center
])

KEYPOINT_VISIBLE = 1
KEYPOINT_INVISIBLE = 0


def get_logs_path():
    path = os.path.join(LOGS_PATH, 'tabledetection')
    return path


def get_data_path():
    path = os.path.join(DATA_PATH)
    return path

def extract_position_torch_gaussian(heatmaps, image_width, image_height, threshold=THRESHOLD):
    """
    Extract the subpixel position of the ball from the heatmaps (batch) using a 2D gaussian fit with torchmin.
    Position is scaled to the original image size.
    Handles image border cases with padding.
    Args:
        heatmaps (torch.Tensor): The heatmaps from which to extract the positions. Shape should be (B, C, H, W).
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        threshold (float): The threshold for the heatmap to consider a point as a valid detection.
    Returns:
        np.ndarray: An array of shape (B, C, 2) containing the (x, y) image coordinates for each heatmap in the batch.
    """
    if len(heatmaps.shape) != 4:
        raise ValueError("Heatmaps must have shape (B, C, H, W)")

    batch_size, num_channels, heatmap_height, heatmap_width = heatmaps.shape

    positions = np.zeros((batch_size, num_channels, 3))

    window_size = 3
    pad = window_size // 2

    # Grid for gaussian fitting relative to window
    y_window, x_window = np.meshgrid(np.arange(window_size), np.arange(window_size), indexing='ij')
    xy_window = np.stack((x_window.flatten(), y_window.flatten()))

    def gaussian_2d_loss(params, xy, window_flat):
        x0, y0, sigma_x, sigma_y = params
        x, y = xy
        sigma_x = max(0.5, sigma_x)  # Ensure positive sigma
        sigma_y = max(0.5, sigma_y)
        gaussian = np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))
        return np.mean((gaussian - window_flat) ** 2)

    # Process each item in the batch
    for b in range(batch_size):
        # Extract heatmaps for the current batch item
        heatmaps_b = heatmaps[b, :, :, :]  # Shape (C, H, W)

        # Pad the heatmaps for window extraction
        padded_heatmaps_b = F.pad(heatmaps_b, (pad, pad, pad, pad), mode='constant', value=0)  # Shape (C, H+2*pad, W+2*pad)

        # Find max points for all channels in the current batch item
        max_indices_b = torch.argmax(heatmaps_b.view(num_channels, -1), dim=1)  # Shape (C,)
        y_max_b = max_indices_b // heatmap_width  # Shape (C,)
        x_max_b = max_indices_b % heatmap_width  # Shape (C,)

        y_max_padded_b = y_max_b + pad
        x_max_padded_b = x_max_b + pad

        # Process each channel for the current batch item
        for c in range(num_channels):

            # check if the heatmap is significant enough to be considered as detection
            activation = heatmaps_b[c, y_max_b[c], x_max_b[c]].item()
            if activation < threshold:
                # If the activation is below the threshold, set position to -10000
                positions[b, c, 2] = KEYPOINT_INVISIBLE
            else:
                positions[b, c, 2] = KEYPOINT_VISIBLE

            # Extract window around the max point for the current channel
            window_c = padded_heatmaps_b[c,
                       y_max_padded_b[c] - pad: y_max_padded_b[c] + pad + 1,
                       x_max_padded_b[c] - pad: x_max_padded_b[c] + pad + 1].cpu().numpy()  # Shape (window_size, window_size)

            # Perform Gaussian fitting on the window
            params_init = np.array([window_size // 2, window_size // 2, 1.0, 1.0], dtype=np.float32)
            bounds = [(0, window_size), (0, window_size), (0.5, window_size), (0.5, window_size)]

            window_flat = window_c.flatten()

            result = minimize(lambda params: gaussian_2d_loss(params, xy_window, window_flat),
                              params_init,
                              method='L-BFGS-B',
                              bounds=bounds)

            if result.success:
                x_offset, y_offset = result.x[0], result.x[1]
            else:
                # Fallback to max position if fitting fails
                y_peak_in_window, x_peak_in_window = np.where(window_c == window_c.max())
                x_offset = float(np.mean(x_peak_in_window))
                y_offset = float(np.mean(y_peak_in_window))

            # Calculate subpixel position in original heatmap coordinates
            x_subpixel = x_max_b[c].float().cpu().numpy() - pad + x_offset
            y_subpixel = y_max_b[c].float().cpu().numpy() - pad + y_offset

            positions[b, c, 0] = x_subpixel
            positions[b, c, 1] = y_subpixel
            positions[b, c, 2] = KEYPOINT_VISIBLE

    # Scale heatmap coordinates to image coordinates
    scale_x = image_width / heatmap_width
    scale_y = image_height / heatmap_height

    # Apply scaling to all positions
    # positions[:, :, 0] *= scale_x
    # positions[:, :, 1] *= scale_y

    # Adjust coordinates by +0.5 for pixel center mapping
    positions[:, :, 0] = (positions[:, :, 0] + 0.5) * scale_x - 0.5
    positions[:, :, 1] = (positions[:, :, 1] + 0.5) * scale_y - 0.5

    return positions


# def taylor_refine_torch(heatmap: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
#     """
#     DARK-style Taylor expansion for subpixel refinement.
#     Args:
#         heatmap: (H, W) tensor representing a single keypoint's heatmap.
#         coord: (2,) tensor [x, y] integer coordinate for refinement center.
#     Returns:
#         Refined coordinate: (2,) tensor [x, y].
#     """
#     H, W = heatmap.shape
#     px, py = int(coord[0]), int(coord[1])
#
#     # Check bounds for 3x3 patch extraction
#     if not (1 <= px < W - 1 and 1 <= py < H - 1):
#         return coord.clone() # Return copy to avoid modifying in place if no refinement
#
#     # Extract 3x3 patch and apply log transform
#     # Clamp minimum value to avoid log(0)
#     patch = heatmap[py - 1:py + 2, px - 1:px + 2].clone()
#     patch = torch.clamp(patch, min=1e-10).log()
#
#     # Calculate first and second partial derivatives
#     # Using central differences for first derivatives
#     dx = 0.5 * (patch[1, 2] - patch[1, 0])
#     dy = 0.5 * (patch[2, 1] - patch[0, 1])
#     # Using second central differences for diagonal second derivatives
#     dxx = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]
#     dyy = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]
#     # Using mixed second derivative approximation
#     dxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])
#
#     # Hessian matrix (second partial derivatives)
#     H_mat = torch.tensor([[dxx, dxy], [dxy, dyy]], device=patch.device)
#     # Gradient vector (negative first partial derivatives)
#     g = torch.tensor([-dx, -dy], device=patch.device)
#
#     # Solve for the offset using Newton's method: offset = -H_mat_inv * g
#     # Use torch.linalg.solve for robustness
#     try:
#         offset = torch.linalg.solve(H_mat, g)
#         # Apply offset to the original integer coordinate
#         refined_coord = coord + offset
#     except torch.linalg.LinAlgError:
#         # If Hessian is singular, skip refinement
#         refined_coord = coord.clone()
#         # print("Hessian singular, skipping refinement")
#
#     return refined_coord
#
#
# def extract_position_torch_dark(heatmaps, image_width, image_height, threshold=THRESHOLD) -> np.ndarray:
#     """
#     Extract subpixel keypoint positions (B, C, 2) from multichannel heatmaps (B, C, H, W)
#     using DARK Taylor refinement. Positions scaled to image size.
#     Args:
#         heatmaps (torch.Tensor): Heatmaps (B, C, H, W).
#         image_width (int): Width of original image.
#         image_height (int): Height of original image.
#     Returns:
#         np.ndarray: (B, C, 2) coordinates in image space (x, y).
#     """
#     if len(heatmaps.shape) != 4:
#         raise ValueError("Heatmaps must have shape (B, C, H, W)")
#
#     batch_size, num_channels, heatmap_height, heatmap_width = heatmaps.shape
#
#     # Initialize tensor for refined coordinates
#     refined_coords_bc = torch.zeros(batch_size, num_channels, 2, device=heatmaps.device)
#
#     # Clamp and log transform the heatmaps
#     # Add small epsilon before log to prevent log(0)
#     heatmaps_log = torch.clamp(heatmaps, min=1e-10).log()
#
#     # Process each item in the batch
#     for b in range(batch_size):
#         # Get log heatmaps for the current batch item
#         heatmaps_log_b = heatmaps_log[b, :, :, :] # Shape (C, H, W)
#         # Get original heatmaps for finding initial max (optional, could use log too)
#         heatmaps_b = heatmaps[b, :, :, :] # Shape (C, H, W)
#
#         # Find initial coarse max locations for all channels in the current batch item
#         # max_idxs will have shape (C,)
#         _, max_idxs_b = torch.max(heatmaps_b.view(num_channels, -1), dim=1)
#         # Convert flat index to (y, x) coordinates
#         y_b = (max_idxs_b // heatmap_width).int() # Shape (C,)
#         x_b = (max_idxs_b % heatmap_width).int() # Shape (C,)
#
#         # Stack initial coordinates for all channels in this batch item
#         initial_coords_b = torch.stack([x_b.float(), y_b.float()], dim=1) # Shape (C, 2)
#
#         # Refine each channel's coordinate individually for the current batch item
#         for c in range(num_channels):
#
#             # Check if the heatmap is significant enough to be considered as detection
#             activation = heatmaps_b[c, y_b[c], x_b[c]].item()
#             if activation < threshold:
#                 # If the activation is below the threshold, set position to -1
#                 refined_coords_bc[b, c, 0] = -10000
#                 refined_coords_bc[b, c, 1] = -10000
#                 continue
#
#             # Get the log heatmap and initial coordinate for the current channel
#             heatmap_log_bc = heatmaps_log_b[c, :, :] # Shape (H, W)
#             initial_coord_bc = initial_coords_b[c, :] # Shape (2,)
#
#             # Apply Taylor refinement
#             refined_coords_bc[b, c, :] = taylor_refine_torch(heatmap_log_bc, initial_coord_bc)
#
#     # Scale from heatmap coordinates to image coordinates
#     scale_x = image_width / heatmap_width
#     scale_y = image_height / heatmap_height
#
#     # Apply scaling and adjust back by -0.5 to map pixel centers correctly
#     refined_coords_bc[:, :, 0] = (refined_coords_bc[:, :, 0] + 0.5) * scale_x - 0.5
#     refined_coords_bc[:, :, 1] = (refined_coords_bc[:, :, 1] + 0.5) * scale_y - 0.5
#
#     # Return as numpy array
#     return refined_coords_bc.cpu().numpy()


def calculate_pck_fixed_tolerance(predictions, ground_truths, tolerance_pixels):
    """
    Calculates Percentage of Correct Keypoints (PCK) per keypoint channel and
    returns the average and standard deviation over all channels.
    Assumes predictions and ground truths are lists/arrays convertible to (N, C, 2).

    Args:
        predictions (list or np.ndarray): Predicted keypoint positions (N, C, 2).
        ground_truths (list or np.ndarray): Ground truth keypoint positions (N, C, 2).
        tolerance_pixels (float): Fixed tolerance in pixels for a prediction to be correct.

    Returns:
        tuple[float, float]: A tuple containing:
            - The average PCK over all channels.
            - The standard deviation of PCK over all channels.
            Returns (np.nan, np.nan) if inputs are empty or have mismatched shapes.
    """
    preds = np.array(predictions, dtype=np.float32)
    gts = np.array(ground_truths, dtype=np.float32)

    if preds.shape != gts.shape or preds.ndim != 3 or preds.shape[2] != 3:
        print(f"Shape mismatch: predictions {preds.shape}, ground_truths {gts.shape}")
        return np.nan, np.nan

    if preds.shape[0] == 0: # Handle empty input
        return np.nan, np.nan

    valid_detections = (preds[..., 2] == KEYPOINT_VISIBLE)
    # if no detection is valid
    if not np.any(valid_detections):
        return -1
    visible_detections = (gts[..., 2] == KEYPOINT_VISIBLE)

    # Calculate Euclidean distance for each keypoint in each sample (N, C)
    distances = np.sqrt(np.sum((preds[..., :2] - gts[..., :2])**2, axis=2))

    # Determine correctness for each keypoint in each sample (N, C)
    is_correct = (distances <= tolerance_pixels) & valid_detections & visible_detections


    # Calculate average and standard deviation over keypoints (C)
    average_pck = np.sum(is_correct) / np.sum(valid_detections & visible_detections)

    return float(average_pck)


def average_distance(predictions, ground_truths):
    """
    Calculates the average Euclidean distance per keypoint channel and
    returns the average and standard deviation over all channels.
    Assumes predictions and ground truths are lists/arrays convertible to (N, C, 3).

    Args:
        predictions (list or np.ndarray): Predicted keypoint positions (N, C, 3).
        ground_truths (list or np.ndarray): Ground truth keypoint positions (N, C, 3).

    Returns:
        tuple[float, float]: A tuple containing:
            - The average of the per-keypoint average distances.
            - The standard deviation of the per-keypoint average distances.
            Returns (np.nan, np.nan) if inputs are empty or have mismatched shapes.
    """
    preds = np.array(predictions, dtype=np.float32)
    gts = np.array(ground_truths, dtype=np.float32)

    if preds.shape != gts.shape or preds.ndim != 3 or preds.shape[2] != 3:
        print(f"Shape mismatch: predictions {preds.shape}, ground_truths {gts.shape}")
        return np.nan

    if preds.shape[0] == 0:  # Handle empty input
        return np.nan

    # Calculate Euclidean distance for each keypoint in each sample (N, C)
    valid_detections = (preds[..., 2] == KEYPOINT_VISIBLE)
    # if no detection is valid
    if not np.any(valid_detections):
        return 10000
    visible_detections = (gts[..., 2] == KEYPOINT_VISIBLE)
    distances = np.sqrt(np.sum((preds[..., :2] - gts[..., :2])**2, axis=2))
    distances_mask = np.where(valid_detections & visible_detections, 1, 0)
    distances = distances * distances_mask

    # Calculate average and standard deviation over keypoints (C)
    average_avg_dist = np.sum(distances) / np.sum(distances_mask)

    return float(average_avg_dist)


def ratio_detected(predictions):
    '''Calculates the ratio of valid keypoints. Keypoint is valid if maximum in heatmap is significant.

    Args:
        predictions (list or np.ndarray): List or array of predicted ball positions (x, y).
                                          Shape should be (N, 3), where N is the number of samples.
    '''
    preds = np.array(predictions, dtype=np.float32)

    valid_detections = (preds[..., 2] == KEYPOINT_VISIBLE)
    distances_mask = np.where(valid_detections, 1, 0)
    detected_keypoints = np.sum(distances_mask)
    all_keypoints = np.prod(distances_mask.shape)
    ratio = detected_keypoints / all_keypoints

    return ratio


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


def weighted_mse_loss(input: torch.Tensor, target: torch.Tensor, visibilities: torch.Tensor) -> torch.Tensor:
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

    # Invisible keypoints should be excluded from loss
    # include_mask = (visibilities == KEYPOINT_VISIBLE)
    # weighted_squared_error = weighted_squared_error * include_mask[:, :, None, None]

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


def concat(x, shape):
    """
    Concatenates a tensor `x` with a tensor of ones along the last dimension.
    Parameters:
    - x: Input tensor, either a numpy array or a PyTorch tensor.
    - shape: Desired shape of the tensor to concatenate (should match `x` except for the last dimension).
    Returns:
    - Concatenated tensor with an additional column of ones.
    """
    if isinstance(x, np.ndarray):
        ones = np.ones((*shape[:-1], 1))
        return np.concatenate([x, ones], axis=-1)
    elif isinstance(x, torch.Tensor):
        ones = torch.ones((*shape[:-1], 1), device=x.device)
        return torch.cat([x, ones], dim=-1)
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor.")


def cam2img(r_cam, Mints):
    '''Project a batch of 3D points to image coordinates.'''
    if len(r_cam.shape) == 1:
        if len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, j -> i')
            r_img = r_img[:2] / r_img[2]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_cam.shape) == 2:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, 'b i j, b j -> b i')
            r_img = r_img[:, :2] / r_img[:, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, b j -> b i')
            r_img = r_img[:, :2] / r_img[:, 2:3]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_cam.shape) == 3:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, 'b i j, b t j -> b t i')
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, b t j -> b t i')
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        else:
            raise ValueError('Shape not supported.')
    else:
        raise ValueError('Shape not supported.')
    return r_img


def world2cam(r_world, Mexts):
    '''Transform a batch of 3D points from world to camera coordinates.'''
    if len(r_world.shape) == 1:
        D = r_world.shape
        if len(Mexts.shape) == 2:
            r_world = concat(r_world, (D,))
            r_cam = eo.einsum(Mexts, r_world, 'i j, j -> i')
            r_cam = r_cam[:3] / r_cam[3]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_world.shape) == 2:
        T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, 'b i j, b j -> b i')
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, 'i j, b j -> b i')
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_world.shape) == 3:
        B, T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, 'b i j, b t j -> b t i')
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, 'i j, b t j -> b t i')
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        else:
            raise ValueError('Shape not supported.')
    else:
        raise ValueError('Shape not supported.')
    return r_cam



if __name__ == "__main__":
    # Example usage
    heatmap = torch.zeros(2, 2, 5, 5)  # Example heatmap
    heatmap[0, 0, 2, 2] = 1.0  # Example ball position in the heatmap
    heatmap[0, 0, 3, 2] = 0.8  # Example ball position in the heatmap
    heatmap[1, 0, 4, 4] = 1.0  # Example ball position in the heatmap
    heatmap[0, 1, 2, 2] = 0.8  # Example ball position in the heatmap
    heatmap[0, 1, 3, 2] = 0.8  # Example ball position in the heatmap
    image_width = 5
    image_height = 5
    positions_gaussian = extract_position_torch_gaussian(heatmap, image_width, image_height)
    print(positions_gaussian)
    #positions_dark = extract_position_torch_dark(heatmap, image_width, image_height)
    #print(positions_dark)

    print('-------------')

    # Example usage of PCK calculation
    predictions = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 15], [300, 12]]])  # Example predictions
    ground_truths = np.array([[[1, 2], [2, 5]], [[5, 6], [6.5, 8]], [[9, 10], [11, 12]]])  # Example ground truths
    tolerance_pixels = 1.0  # Example tolerance
    average_pck, std_pck = calculate_pck_fixed_tolerance(predictions, ground_truths, tolerance_pixels)
    print(f"Average PCK: {average_pck}, Std PCK: {std_pck}")

    # Example usage of average distance calculation
    average_avg_dist, std_avg_dist = average_distance(predictions, ground_truths)
    print(f"Average Average Distance: {average_avg_dist}, Std Average Distance: {std_avg_dist}")