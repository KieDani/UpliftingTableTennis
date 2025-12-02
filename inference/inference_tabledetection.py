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
from inference.utils import process_trajectory_table, calibrate_camera, filter_trajectory_table
from inference.utils import get_table_model, get_transforms_table, extract_position_table, plot_transforms, table_points
from inference.utils import KEYPOINT_VISIBLE, world2cam, cam2img

from inference.dataset import TTHQ as TTHQ_trajectory

from tabledetection.dataset import TableTennisTable, TTHQ
from tabledetection.helper_tabledetection import calculate_pck_fixed_tolerance


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 4

model_paths = [
        os.path.join(paths.weights_path, 'inference_tabledetection', 'segformerpp_b0', 'model.pt'),
        os.path.join(paths.weights_path, 'inference_tabledetection', 'segformerpp_b2', 'model.pt'),
        os.path.join(paths.weights_path, 'inference_tabledetection', 'hrnet', 'model.pt'),
        os.path.join(paths.weights_path, 'inference_tabledetection', 'vitpose', 'model.pt'),
    ]


def load_model(model_path):
    '''
    Load the table detection model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        table_model (torch.nn.Module): Loaded table detection model.
        transform_table (callable): Transformation function for input images.
    '''
    load_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_name = load_dict['additional_info']['model_name']
    resolution = load_dict['additional_info']['image_resolution']
    table_model = get_table_model(model_name, resolution=resolution, pretraining=False)
    table_model.load_state_dict(load_dict['model_state_dict'])
    table_model.eval()
    print(f'Loaded tabledetection model: {model_name} with resolution {resolution}')
    transform_table = get_transforms_table('test', resolution)
    return table_model, transform_table


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
    table_model, transform_table = load_model(model_path)
    table_model.to(device)

    # load the dataset
    mode = 'test'
    heatmap_sigma = 12  # not important for inference
    dataset = TTHQ(mode=mode, heatmap_sigma=heatmap_sigma, transform=transform_table)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=min(BATCH_SIZE, 4))

    gt_pos, pred_pos = [], []
    with torch.no_grad():
        for i, (image, heatmap, ball_coords) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            ball_coords = ball_coords.to(device)

            # run the model
            pred = table_model(image)

            # calculate predicted positions scaled to evaluation resolution
            pred_positions = extract_position_table(pred, WIDTH, HEIGHT)
            # pred_positions = extract_position_torch_dark(pred, WIDTH, HEIGHT)
            for b in range(image.shape[0]):
                pred_pos.append(pred_positions[b])
                gt_pos.append(ball_coords[b].cpu().numpy())

    # calculate pck
    pck2 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=2)
    pck5 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=5)
    pck10 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=10)
    pck20 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=20)

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
    table_model1, transform_table1 = load_model(model_path1)
    table_model2, transform_table2 = load_model(model_path2)
    table_model1.to(device)
    table_model2.to(device)

    dataset1 = TTHQ_trajectory(transform_ball=None, transform_table=transform_table1)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=1)  # batch size 1 is important!
    dataset2 = TTHQ_trajectory(transform_ball=None, transform_table=transform_table2)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)  # batch size 1 is important!
    print(f'Loaded dataset with {len(dataset1)} samples.')

    for (__, images_table1, fps, __), (__, images_table2, __, __) in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
        B, T, C, H, W = images_table1.shape
        # Process table keypoints
        raw_table_keypoints1 = process_trajectory_table(table_model1, images_table1)
        raw_table_keypoints2 = process_trajectory_table(table_model2, images_table2)
        pred_table_keypoints = filter_trajectory_table(raw_table_keypoints1, raw_table_keypoints2)

        num_invis = np.sum(pred_table_keypoints[:, 2] != KEYPOINT_VISIBLE)
        if num_invis < 1: continue  # only show problematic examples

        # Calibrate camera
        M_int, M_ext = calibrate_camera(pred_table_keypoints)
        # calculate reprojection error
        points_3D = np.array(table_points)
        reprojected_points3D = cam2img(world2cam(points_3D, M_ext), M_int)
        points_2D = pred_table_keypoints[:, :2]
        visible_indices = pred_table_keypoints[:, 2] == KEYPOINT_VISIBLE
        points_2D = points_2D[visible_indices]
        reprojected_points3D = reprojected_points3D[visible_indices]
        error = np.mean(np.linalg.norm(points_2D - reprojected_points3D, axis=1))
        print(f'Camera Calibration reprojection error: {error:.2f} pixels')

        # print filtered prediction if available, else raw predictions
        plot_image = images_table1[0, T // 2].clone()
        plot_image = plot_transforms({'image': plot_image.numpy()})['image']
        plot_image = eo.rearrange(plot_image, 'c h w -> h w c')
        plot_image = cv2.resize(plot_image, (WIDTH, HEIGHT))

        table_points_3D = np.array(table_points)
        reprojected_points = cam2img(world2cam(table_points_3D, M_ext), M_int)
        t_worst1, error_worst = 0, 0
        for t in range(T):
            error = np.sum(np.linalg.norm(raw_table_keypoints1[t, :, :2] - reprojected_points, axis=1))
            if error > error_worst:
                t_worst1, error_worst = t, error
        t_worst2, error_worst = 0, 0
        for t in range(T):
            error = np.sum(np.linalg.norm(raw_table_keypoints2[t, :, :2] - reprojected_points, axis=1))
            if error > error_worst:
                t_worst2, error_worst = t, error

        for i, (filtered_point, raw_point1, raw_point2) in enumerate(zip(pred_table_keypoints, raw_table_keypoints1[t_worst1], raw_table_keypoints2[t_worst2])):
            x, y, v = filtered_point
            if v == KEYPOINT_VISIBLE:
                cv2.circle(plot_image, (int(x), int(y)), 7, (0, 255, 0), -1)
                cv2.putText(plot_image, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                x, y, __ = raw_point1
                cv2.circle(plot_image, (int(x), int(y)), 7, (220, 20, 60), -1)
                cv2.putText(plot_image, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 20, 60), 1)
                x, y, __ = raw_point2
                cv2.circle(plot_image, (int(x), int(y)), 6, (255, 165, 0), -1)
                cv2.putText(plot_image, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        plt.figure(figsize=(30, 15))
        plt.imshow(plot_image)
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    main()
    # evaluate_filter(model_paths[0], model_paths[1])


