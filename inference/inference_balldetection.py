import os
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
import numpy as np
from tqdm import tqdm
import cv2
import einops as eo
import matplotlib.pyplot as plt

import paths

from inference.utils import HEIGHT, WIDTH
from inference.utils import process_trajectory_ball, calibrate_camera, filter_trajectory_ball
from inference.utils import get_ball_model, get_transforms_ball, extract_position_table, plot_transforms, table_points, get_transforms_table
from inference.utils import BALL_VISIBLE, world2cam, cam2img

from inference.dataset import TTHQ as TTHQ_trajectory

from balldetection.dataset import TableTennisBall, TTHQ
from balldetection.helper_balldetection import calculate_pck_fixed_tolerance
from balldetection.config import TrainConfig as BallConfig

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 4

model_paths = [
    os.path.join(paths.weights_path, 'inference_balldetection', 'segformerpp_b0', 'model.pt'),
    os.path.join(paths.weights_path, 'inference_balldetection', 'segformerpp_b2', 'model.pt'),
    os.path.join(paths.weights_path, 'inference_balldetection', 'wasb', 'model.pt'),
    os.path.join(paths.weights_path, 'inference_balldetection', 'vitpose', 'model.pt'),
]


def load_model(model_path):
    '''
    Load the ball detection model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        ball_model (torch.nn.Module): Loaded ball detection model.
        transform_ball (callable): Transformation function for input images.
    '''
    load_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_name = load_dict['additional_info']['model_name']
    resolution = load_dict['additional_info']['image_resolution']
    in_frames = load_dict['additional_info']['in_frames']
    lr = load_dict['additional_info']['lr']
    ball_model = get_ball_model(model_name, in_frames=in_frames, resolution=resolution, pretraining=False)
    ball_model.load_state_dict(load_dict['model_state_dict'])
    ball_model.eval()
    print(f'Loaded BallDetection model: {model_name} with resolution {resolution}')
    print(f' - in_frames: {in_frames}, lr: {lr}')
    transform_ball = get_transforms_ball('test', resolution)

    return ball_model, transform_ball


def inference(model_path, title=''):
    '''
    Run inference on the dataset. Calculate metrics compared to manually annotated ground truth.
    Args:
        model_path (str): Path to the pre-trained model.
        title (str): Title for the inference run, used for logging.
    '''
    print('----- Starting inference for tabledetection. -----')
    print(f'Inference title: {title}')

    # load the model
    ball_model, transform_ball = load_model(model_path)
    ball_model.to(device)

    # load the dataset
    mode, in_frames = 'test', 3
    heatmap_sigma = 6  # not important for inference
    dataset = TTHQ(mode=mode, in_frames=3, heatmap_sigma=heatmap_sigma, transform=transform_ball, use_invisible=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=min(BATCH_SIZE, 4))

    gt_pos, pred_pos = [], []
    with torch.no_grad():
        for i, (image, heatmap, ball_coords, __, __, visibility) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            ball_coords = ball_coords

            # run the model
            pred, __ = ball_model(image)

            # calculate predicted positions scaled to evaluation resolution
            pred_positions = extract_position_table(pred, WIDTH, HEIGHT)
            for b in range(image.shape[0]):
                pred_pos.append(pred_positions[b])
                gt_pos.append(ball_coords[b].cpu().numpy())

    pred_pos = np.asarray(pred_pos).squeeze(1)  # (T, 1, 3) -> (T, 3)
    gt_pos = np.asarray(gt_pos)

    # TODO: Zur Sicherheit eine Implementierung ohne Blur Streaks verwenden
    pck2 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, gt_pos, gt_pos, tolerance_pixels=2)
    pck5 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, gt_pos, gt_pos, tolerance_pixels=5)
    pck10 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, gt_pos, gt_pos, tolerance_pixels=10)
    pck20 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, gt_pos, gt_pos, tolerance_pixels=20)

    print(f'PCK @ 2px: {pck2:.4f}')
    print(f'PCK @ 5px: {pck5:.4f}')
    print(f'PCK @ 10px: {pck10:.4f}')
    print(f'PCK @ 20px: {pck20:.4f}')

    print('----- Finished inference for tabledetection. -----')


def main():
    for model_path in model_paths:
        inference(model_path, title='tabledetection on TTHQ')


def evaluate_filter(model_path1, model_path2):
    '''
    Filter trajectories. We calculate how many points are filtered and visualize.
    Args:
        model_path1 (str): Path to the first model.
        model_path2 (str): Path to the second model.
    '''
    ball_model1, transform_ball1 = load_model(model_path1)
    ball_model2, transform_ball2 = load_model(model_path2)
    ball_model1.to(device)
    ball_model2.to(device)

    transform_table = get_transforms_table('test', (WIDTH, HEIGHT))  # Only used for plotting
    dataset1 = TTHQ_trajectory(transform_ball=transform_ball1, transform_table=transform_table)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=1)  # batch size 1 is important!
    dataset2 = TTHQ_trajectory(transform_ball=transform_ball2, transform_table=None)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)  # batch size 1 is important!
    print(f'Loaded dataset with {len(dataset1)} samples.')

    for (images_ball1, images_table, fps, __), (images_ball2, __, __, __) in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
        B, T, C, H, W = images_ball1.shape
        # Process table keypoints
        raw_ball_keypoints1 = process_trajectory_ball(ball_model1, images_ball1)
        raw_ball_keypoints2 = process_trajectory_ball(ball_model2, images_ball2)
        pred_ball_keypoints, valid_indices, times = filter_trajectory_ball(raw_ball_keypoints1, raw_ball_keypoints2, fps)

        T_prime = pred_ball_keypoints.shape[0]
        num_invis = T - T_prime
        if num_invis < 2: continue

        # plot filtered if available, else plot raw
        plot_image = images_table[0, T // 2].clone()
        plot_image = plot_transforms({'image': plot_image.numpy()})['image']
        plot_image = eo.rearrange(plot_image, 'c h w -> h w c')
        plot_image = cv2.resize(plot_image, (WIDTH, HEIGHT))
        for i, coord in enumerate(raw_ball_keypoints1):
            if i in valid_indices:
                x, y = pred_ball_keypoints[np.where(valid_indices == i)[0][0]]
                cv2.circle(plot_image, (int(x), int(y)), 7, (0, 255, 0), -1)
            else:
                x, y, __ = coord
                cv2.circle(plot_image, (int(x), int(y)), 7, (220, 20, 60), -1)  # This color is called crimson
                x, y, __ = raw_ball_keypoints2[i]
                cv2.circle(plot_image, (int(x), int(y)), 7, (255, 165, 0), -1)  # This color is called orange
        plt.figure(figsize=(30, 15))
        plt.imshow(plot_image)
        plt.axis('off')
        plt.show()



if __name__ == "__main__":
    main()
    # evaluate_filter(model_paths[0], model_paths[2])

