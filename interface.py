import os
import torch
import torch.hub
import zipfile
import numpy as np
import cv2
import einops as eo

from inference.utils import HEIGHT, WIDTH
from inference.utils import process_trajectory_ball, calibrate_camera, filter_trajectory_ball
from inference.utils import extract_position_table
from inference.utils import BALL_VISIBLE, KEYPOINT_VISIBLE, world2cam, cam2img
from inference.utils import process_trajectory_table, calibrate_camera, filter_trajectory_table
from inference.utils import process_trajectory_uplifting, _uplifting_transform

# ball detection
from inference.inference_balldetection import load_model as load_ball_model
# table detection
from inference.inference_tabledetection import load_model as load_table_model
# uplifting
from inference.inference_uplifting import load_model as load_uplifting_model
from uplifting.helper import transform_rotationaxes

import paths



# --- CONFIGURATION ---
WEIGHTS_ZIP_URL = "https://mediastore.rz.uni-augsburg.de/get/TL7oQRStHG/"
ZIP_FILENAME = "tt_uplifting_weights.zip"
EXTRACTED_FOLDER_NAME = "weights"  # The zip contains a folder named 'weights'


def _get_weights_path(relative_path):
    """
    Downloads the full weights zip if not present, extracts it,
    and returns the local path to the specific requested weight file.
    """
    hub_dir = torch.hub.get_dir()
    download_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(download_dir, exist_ok=True)

    zip_path = os.path.join(download_dir, ZIP_FILENAME)
    extract_path = os.path.join(download_dir, "tt_uplifting_extracted")

    # The full local path to the specific model file
    target_file = os.path.join(extract_path, EXTRACTED_FOLDER_NAME, relative_path)

    # 1. Check if the target file already exists
    if os.path.exists(target_file):
        return target_file

    print(f"Weights not found at {target_file}.")

    # 2. Check if zip exists, if not download it
    if not os.path.exists(zip_path):
        print(f"Downloading weights from {WEIGHTS_ZIP_URL}...")
        try:
            torch.hub.download_url_to_file(WEIGHTS_ZIP_URL, zip_path, progress=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download weights: {e}")

    # 3. Extract zip if we haven't already (or if target file is missing)
    if not os.path.exists(extract_path) or not os.path.exists(target_file):
        print("Extracting weights... this may take a moment.")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Extraction complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to extract weights: {e}")

    return target_file



# set paths in paths.py
paths.weights_path = _get_weights_path('')
print(paths.weights_path)



class BallDetector:
    def __init__(self, model_name='segformerpp_b2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resolution = (WIDTH, HEIGHT)

        # Initialize Architecture
        self.model, self.transform = load_ball_model(model_path=_get_weights_path(f"inference_balldetection/{model_name}/model.pt"))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images):
        """
        Input:
            - images: List (length B) with each entry being 3 numpy images (BGR format): [[prev, curr, next], ...]
        Returns:
            - pred_pos: predicted ball positions in pixel coordinates: [[x, y, confidence], ...] with shape (B, 3)
            - preds: predicted heatmaps as np arrays with shape (B, 1, H, W)
        """
        pred_pos, preds = [], []
        for imgs in images:
            # Apply transforms
            data = {
                'image': imgs[1],
                'prev_image': imgs[0],
                'next_image': imgs[2]
            }
            data = self.transform(data)
            element = np.concatenate([data['prev_image'], data['image'], data['next_image']], axis=2)
            element = eo.rearrange(element, 'h w c -> c h w').astype(np.float32)
            input_tensor = torch.tensor(element).to(self.device)
            with torch.no_grad():
                # apply model
                preds_tmp, _ = self.model(input_tensor.unsqueeze(0))
                pred_pos_tmp = extract_position_table(preds_tmp, self.resolution[0], self.resolution[1])
                preds_tmp = preds_tmp.squeeze(0).cpu().numpy()
            pred_pos.append(pred_pos_tmp.squeeze(0))
            preds.append(preds_tmp)
        return np.concatenate(pred_pos, axis=0), np.array(preds)

    def filter_trajectory(self, ball_positions, ball_positions_aux, fps):
        '''
        Filter the ball trajectory using auxiliary detections.
        Input:
            - ball_positions: (N, 3) array for the detected ball positions in pixel coordinates
            - ball_positions_aux: (N, 3) array for the auxiliary detected ball positions in pixel coordinates
            - fps: float number representing the framerate of the video
        Returns:
            - filtered_ball_positions: (M, 3) array for the filtered ball positions in pixel coordinates
            - valid_indices: list of valid indices after filtering
            - times: list of time stamps corresponding to the filtered positions
        '''
        return filter_trajectory_ball(ball_positions, ball_positions_aux, fps)


class TableDetector:
    def __init__(self, model_name='segformerpp_b2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resolution = (WIDTH, HEIGHT)
        self.KEYPOINT_VISIBLE = KEYPOINT_VISIBLE

        # Initialize Architecture
        self.model, self.transform = load_table_model(model_path=_get_weights_path(f"inference_tabledetection/{model_name}/model.pt"))
        self.model.to(self.device)
        self.model.eval()


    def predict(self, images):
        """
        Input:
            - images: list of images (BGR format) of length B
        Returns:
            - pred_pos (B, 13, 3) array for the 13 table keypoints in pixel coordinates
            - preds: predicted heatmaps as torch tensor with shape (B, 13, H, W)
        """
        pred_pos, preds = [], []
        for img in images:
            # Apply transforms
            data = {
                'image': img
            }
            data = self.transform(data)
            element = eo.rearrange(data['image'], 'h w c -> c h w').astype(np.float32)
            input_tensor = torch.tensor(element).to(self.device)
            with torch.no_grad():
                # apply model
                preds_tmp = self.model(input_tensor.unsqueeze(0))
                pred_pos_tmp = extract_position_table(preds_tmp, self.resolution[0], self.resolution[1])
            pred_pos.append(pred_pos_tmp)
            preds.append(preds_tmp.cpu())
        return np.concatenate(pred_pos, axis=0), np.array(preds)

    def calibrate_camera(self, keypoints):
        return calibrate_camera(keypoints)

    def filter_trajectory(self, table_keypoints, table_keypoints_aux):
        '''
        Filter the table keypoint trajectory using auxiliary detections.
        Input:
            - table_keypoints: (N, 13, 3) array for the detected table keypoints in pixel coordinates
            - table_keypoints_aux: (N, 13, 3) array for the auxiliary detected table keypoints in pixel coordinates
        Returns:
            - filtered_table_keypoints: (M, 13, 3) array for the filtered table keypoints in pixel coordinates
        '''
        return filter_trajectory_table(table_keypoints, table_keypoints_aux)


class UpliftingModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Architecture
        self.model, self.transform, self.transform_mode = load_uplifting_model(model_path=_get_weights_path(f"inference_uplifting/ours/model.pt"))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, ball_coords, table_coords, times):
        '''
        Input:
            - ball_coords: torch tensor (N, 2) of 2D ball coordinates in pixel space
            - table_coords: torch tensor (13, 3) of 2D table keypoints in pixel space -> (x, y, visibility)
            - times: torch tensor (N,) of time stamps in seconds -> they have to match the ball coordinates
        Returns:
            - pred_spin: torch tensor (3,) of predicted spin vector in local coordinate system
            - pred_pos_3d: torch tensor (N, 3) of predicted 3D ball positions in world coordinates
        '''
        # Prepare inputs
        data = {
            'r_img': ball_coords,
            'table_img': table_coords,
        }
        data = self.transform(data)
        ball_coords, table_coords = data['r_img'], data['table_img']

        mask = np.zeros((ball_coords.shape[0] + 1,), dtype=np.float32)
        mask[:-1] = 1.0  # True for all ball points

        return self.predict_without_normalization(ball_coords, table_coords, torch.tensor(mask).to(self.device), times)

    def predict_without_normalization(self, ball_coords, table_coords, mask, times):
        ''' Assume coords are already normalized
        Input:
            - ball_coords: torch tensor (N, 2) of 2D ball coordinates in pixel space
            - table_coords: torch tensor (13, 3) of 2D table keypoints in pixel space -> (x, y, visibility)
            - mask: torch tensor (N+1,) of mask for ball coordinates
            - times: torch tensor (N,) of time stamps in seconds -> they have to match the ball coordinates
        Returns:
            - pred_spin: torch tensor (3,) of predicted spin vector in local coordinate system
            - pred_pos: torch tensor (N, 3) of predicted 3D ball positions in world coordinates
        '''
        ball_coords, table_coords, mask, times = ball_coords.to(self.device), table_coords.to(self.device), mask.to(self.device), times.to(self.device)

        with torch.no_grad():
            pred_rotation, pred_position = self.model(ball_coords, table_coords, mask, times)

        # transform prediction into local coordinate system
        if self.transform_mode == 'global':
            pred_rotation_local = transform_rotationaxes(pred_rotation, pred_position.clone())
        else:
            pred_rotation_local = pred_rotation

        # remove padding
        T_prime = int(mask.sum().item())
        pred_position = pred_position[:, :T_prime, :].cpu().numpy()

        return pred_rotation_local.squeeze(0), pred_position.squeeze(0)



class TableTennisPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 1. Front-end
        self.ball_detector = BallDetector(model_name='segformerpp_b2')
        self.ball_detector_aux = BallDetector(model_name='wasb')
        self.table_detector = TableDetector(model_name='segformerpp_b2')
        self.table_detector_aux = TableDetector(model_name='hrnet')
        # 2. Back-end Configuration
        self.uplifting_model = UpliftingModel()

        self.KEYPOINT_VISIBLE = self.table_detector.KEYPOINT_VISIBLE


    def predict(self, images, fps):
        '''
        Input:
            - images: List (length B) with each entry being a numpy images (BGR format) -> Exactly one shot of a table tennis rally
            - fps: float number representing the framerate of the video -> used to create time stamps
        Returns:
            - pred_spin: torch tensor (3,) of predicted spin vector in local coordinate system
            - pred_pos_3d: torch tensor (N, 3) of predicted 3D ball positions in world coordinates
        '''
        # 1. Ball Detection
        image_triples = [(images[i-1].copy(), images[i].copy(), images[i+1].copy()) for i in range(1, len(images)-1)]
        ball_positions, _ = self.ball_detector.predict(image_triples)
        ball_positions_aux, _ = self.ball_detector_aux.predict(image_triples)
        filtered_ball_positions, valid_indices_ball, times_ball = self.ball_detector.filter_trajectory(ball_positions, ball_positions_aux, fps)

        # 2. Table Detection
        table_keypoints, _ = self.table_detector.predict(images)
        table_keypoints_aux, _ = self.table_detector_aux.predict(images)
        filtered_table_keypoints = self.table_detector_aux.filter_trajectory(table_keypoints, table_keypoints_aux)

        # 3. Uplifting
        ball_coords, table_coords, times, mask = _uplifting_transform(filtered_ball_positions, filtered_table_keypoints, times_ball)
        pred_spin, pred_pos_3d = self.uplifting_model.predict_without_normalization(ball_coords, table_coords, mask, times)

        return pred_spin, pred_pos_3d

    def calibrate_camera(self, keypoints):
        '''
        Input:
            - keypoints: numpy array with shape (13, 3) for the 13 table keypoints in pixel coordinates (x, y, visibility)
        Returns:
            - Mint: intrinsic camera matrix of shape (3, 3)
            - Mext: extrinsic camera matrix of shape (3, 4)
        '''
        return calibrate_camera(keypoints)

    def reproject(self, positions_3d, Mint, Mext):
        '''
        Input:
            - positions_3d: numpy array of shape (N, 3) of 3D ball positions in world coordinates
            - Mint: intrinsic camera matrix of shape (3, 3)
            - Mext: extrinsic camera matrix of shape (3, 4)
        Returns:
            - positions_2d: numpy array of shape (N, 2) of 2D ball positions in pixel coordinates
        '''
        positions_cam = world2cam(positions_3d, Mext)
        positions_2d = cam2img(positions_cam, Mint)
        return positions_2d


if __name__ == "__main__":
    # Simple test to check if weights can be loaded
    ball_model = BallDetector(model_name='segformerpp_b2')
    ball_model_aux = BallDetector(model_name='wasb')
    table_model = TableDetector(model_name='segformerpp_b2')
    table_model_aux = TableDetector(model_name='hrnet')
    uplifting_model = UpliftingModel()
    print("All models loaded successfully.")

    # Load sample images
    frames_folder = os.path.join('tutorials', 'example_imgs')
    images = [cv2.imread(os.path.join(frames_folder, f'{i:02d}.png')) for i in range(0, 35)]
    # Set the framerate for the loaded data
    fps = 60.0

    # Ball Detection Only
    image_triples = [(images[i-1].copy(), images[i].copy(), images[i+1].copy()) for i in range(1, len(images)-1)]
    ball_positions, _ = ball_model.predict(image_triples)
    print("Ball Detection executed successfully.")
    print("Predicted Ball Positions:", ball_positions.shape)

    # Filtering of ball trajectory
    ball_positions_aux, _ = ball_model_aux.predict(image_triples)
    filtered_ball_positions, valid_indices, times = ball_model.filter_trajectory(ball_positions, ball_positions_aux, fps)
    print("Filtered Ball Positions:", filtered_ball_positions.shape)
    print("Number of removed detections:", len(ball_positions) - len(filtered_ball_positions))

    # Table Detection Only
    table_keypoints, _ = table_model.predict(images)
    print("Table Detection executed successfully.")
    print("Predicted Table Keypoints:", table_keypoints.shape)

    # Filtering of table trajectory
    table_keypoints_aux, _ = table_model_aux.predict(images)
    filtered_table_keypoints = table_model.filter_trajectory(table_keypoints, table_keypoints_aux)
    print("Filtered Table Keypoints:", filtered_table_keypoints.shape)
    print("Number of removed keypoints:", np.where(filtered_table_keypoints[:, 2] != KEYPOINT_VISIBLE, 1, 0).sum())

    # Full Pipeline
    pipeline = TableTennisPipeline()
    pred_spin, pred_pos_3d = pipeline.predict(images, fps)
    pred_spin_class_str = 'Topspin' if pred_spin[1] > 0 else 'Backspin'
    print("Pipeline executed successfully.")
    print("Predicted Spin:", pred_spin)
    print("Predicted 3D Positions:", pred_pos_3d.shape)

    # plot
    import matplotlib.pyplot as plt
    img = images[10].copy()

    # plot the detected 2D ball trajectory on one image
    for pos in filtered_ball_positions:
        x, y = pos
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Ball Trajectory")
    plt.axis('off')
    plt.show()

    # plot the filtered table keypoints on one image
    img = images[10].copy()
    for kp in table_keypoints[10]:
        x, y, v = kp
        if v == table_model.KEYPOINT_VISIBLE:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Table Keypoints")
    plt.axis('off')
    plt.show()

    # plot the reprojected ball positions on one image
    img = images[10].copy()
    Mint, Mext = pipeline.calibrate_camera(filtered_table_keypoints)
    reprojected_trajectory = pipeline.reproject(pred_pos_3d, Mint, Mext)
    for pos, pos_det in zip(reprojected_trajectory, filtered_ball_positions):
        x, y = pos
        x_det, y_det = pos_det
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)  # reprojected position in blue
        cv2.circle(img, (int(x_det), int(y_det)), 5, (0, 255, 0), -1)  # 2D detection for comparison in green
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Ball Positions")
    plt.axis('off')
    plt.show()