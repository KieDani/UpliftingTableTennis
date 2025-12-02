import os

from sympy.printing.pretty.pretty_symbology import line_width

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='tthq', choices=['tthq', 'ttst'])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
import einops as eo
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import paths

from tabledetection.helper_tabledetection import KEYPOINT_VISIBLE, KEYPOINT_INVISIBLE
from uplifting.helper import table_points, cam2img, world2cam, table_connections
from uplifting.transformations import UnNormalizeImgCoords

from inference.dataset import TTHQ, TTST
from inference.dataset import TOPSPIN_CLASS, BACKSPIN_CLASS
from inference.inference_balldetection import load_model as load_ball_model
from inference.inference_tabledetection import load_model as load_table_model
from inference.inference_uplifting import load_model as load_uplifting_model
from inference.utils import process_trajectory_ball, filter_trajectory_ball, process_trajectory_uplifting, _uplifting_transform, process_trajectory_table, filter_trajectory_table
from inference.utils import calibrate_camera
from inference.utils import plot_transforms
from inference.utils import HEIGHT, WIDTH

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # Get the properties of the first GPU (index 0)
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory_bytes = gpu_properties.total_memory
    total_memory_gb = total_memory_bytes / (1024**3)
else:
    total_memory_gb = 0

move_weights = True if total_memory_gb < 60 else False


model_paths = [
    (
        # balldetection
        (
            os.path.join(paths.weights_path, 'inference_balldetection', 'segformerpp_b2', 'model.pt'),  # main model
            os.path.join(paths.weights_path, 'inference_balldetection', 'wasb', 'model.pt'),  # auxiliary model
        ),
        # tabledetection
        (
            os.path.join(paths.weights_path, 'inference_tabledetection', 'segformerpp_b2', 'model.pt'),  # main model
            os.path.join(paths.weights_path, 'inference_tabledetection', 'hrnet', 'model.pt'),  # auxiliary model
        ),
        # uplifting
        os.path.join(paths.weights_path, 'inference_uplifting', 'ours', 'model.pt'),
    ),
]


def inference_tthq(ball_model_paths, table_model_paths, uplifting_model_path):
    '''
    Run inference on the TTHQ dataset using combined balldetection, tabledetection, and Uplifting models.
    First extract ball positions and filter, next extract table positions and filter, then use the uplifting model to refine the ball positions.
    Calculate camera matrices from table detections, calculate reprojection errors for
    table points vs reprojected table points & detected 2D ball positions vs reprojected 3D ball positions.
    Args:
        ball_model_paths (tuple): Paths to the 2 balldetection models.
        table_model_paths (tuple): Paths to the 2 tabledetection models.
        uplifting_model_path (str): Path to the Uplifting model.
    '''
    print('----- Starting combined inference for TTHQ. -----')

    # load ball models
    assert len(ball_model_paths) == 2, 'Please provide exactly 2 balldetection model paths.'
    ball_model1, transform_ball1 = load_ball_model(ball_model_paths[0])
    ball_model2, transform_ball2 = load_ball_model(ball_model_paths[1])

    # load table models
    assert len(table_model_paths) == 2, 'Please provide exactly 2 tabledetection model paths.'
    table_model1, transform_table1 = load_table_model(table_model_paths[0])
    table_model2, transform_table2 = load_table_model(table_model_paths[1])

    # load uplifting model
    uplifting_model, __, transform_mode = load_uplifting_model(uplifting_model_path)  # Don't use the standard transform because of image size mismatch
    transform_uplifting = _uplifting_transform

    # Load dataset
    dataset1 = TTHQ(transform_ball=transform_ball1, transform_table=transform_table1)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=1)
    dataset2 = TTHQ(transform_ball=transform_ball2, transform_table=transform_table2)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)
    print(f'Loaded dataset with {len(dataset1)} samples.')

    # inference
    errors_table = []
    errors_ball = []
    TP, TN, FP, FN = 0, 0, 0, 0
    with tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)) as pbar:
        for (images_ball1, images_table1, fps, spin_class), (images_ball2, images_table2, __, __) in pbar:
            B, T, C, H, W = images_ball1.shape

            # Process ball trajectory
            raw_positions_ball1 = process_trajectory_ball(ball_model1, images_ball1)
            raw_positions_ball2 = process_trajectory_ball(ball_model2, images_ball2)
            pred_positions_ball, valid_indices_ball, times_ball = filter_trajectory_ball(raw_positions_ball1, raw_positions_ball2, fps)
            new_T = pred_positions_ball.shape[0]
            # Process table keypoints
            raw_table_keypoints1 = process_trajectory_table(table_model1, images_table1)
            raw_table_keypoints2 = process_trajectory_table(table_model2, images_table2)
            pred_table_keypoints = filter_trajectory_table(raw_table_keypoints1, raw_table_keypoints2)

            # Process uplifting
            ball_coords, table_coords, times, mask = transform_uplifting(pred_positions_ball, pred_table_keypoints, times_ball)
            pred_spin_local, pred_positions_3d = process_trajectory_uplifting(uplifting_model, ball_coords, table_coords, times, mask, transform_mode)

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            if spin_class == TOPSPIN_CLASS:  # Frontspin
                if pred_spin_local[1] > 0:
                    TP += 1
                else:
                    FN += 1
            elif spin_class == BACKSPIN_CLASS:  # Backspin
                if pred_spin_local[1] < 0:
                    TN += 1
                else:
                    FP += 1
            # else: spin annotation was forgotten -> do not include in spin metrics

            # Calibrate camera
            M_int, M_ext = calibrate_camera(pred_table_keypoints)

            # calculate reprojection error for table keypoints
            points_3D = np.array(table_points)
            reprojected_points3D = cam2img(world2cam(points_3D, M_ext), M_int)
            points_2D = pred_table_keypoints[:, :2]
            visible_indices = pred_table_keypoints[:, 2] == KEYPOINT_VISIBLE
            points_2D = points_2D[visible_indices]
            reprojected_points3D = reprojected_points3D[visible_indices]
            error_table = np.mean(np.linalg.norm(points_2D - reprojected_points3D, axis=1))
            errors_table.append(error_table)

            # calculate reprojection error for ball positions
            reprojected_ball_2D = cam2img(world2cam(pred_positions_3d, M_ext), M_int)
            detected_ball_2D = pred_positions_ball[:, :2]
            error_ball = np.mean(np.linalg.norm(detected_ball_2D - reprojected_ball_2D, axis=1))
            errors_ball.append(error_ball)

            # Update the progress bar with the current metrics (the inference takes ages, and I am already curious)
            f1_plus = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            f1_minus = 2 * TN / (2 * TN + FN + FP) if (2 * TN + FN + FP) > 0 else 0
            pbar.set_postfix({
                'tbl': f'{np.mean(errors_table):.2f}',
                'ball': f'{np.mean(errors_ball):.2f}',
                'f1': f'{(f1_plus + f1_minus) / 2:.4f}',
                'acc': f'{(TP + TN) / (TP + TN + FP + FN):.4f}',
                'f1+': f'{f1_plus:.4f}',
                'f1-': f'{f1_minus:.4f}',
            })

    # Spin metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_plus = 2 * TP / (2 * TP + FP + FN)
    f1_minus = 2 * TN / (2 * TN + FN + FP)
    macro_f1 = (f1_plus + f1_minus) / 2
    print(f'Spin classification accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}')

    print(f'Table Keypoint error: {np.mean(errors_table):.2f} ± {np.std(errors_table):.2f} pixels')
    print(f'Ball Position error: {np.mean(errors_ball):.2f} ± {np.std(errors_ball):.2f} pixels')

    print('----- Finished combined inference for TTHQ. -----')


def inference_ttst(ball_model_paths, table_model_paths, uplifting_model_path):
    '''
    Run inference on the TTST dataset using combined balldetection, tabledetection, and Uplifting models.
    First extract ball positions and filter, next extract table positions and filter, then use the uplifting model to refine the ball positions.
    Calculate reprojection errors for table points vs reprojected table points
    & detected 2D ball positions vs annotated 2D ball positions
    & reprojected predicted 3D ball positions vs annotated 2D ball positions.
    Args:
        ball_model_paths (tuple): Paths to the 2 balldetection models.
        table_model_paths (tuple): Paths to the 2 tabledetection models.
        uplifting_model_path (str): Path to the Uplifting model.
    '''
    print('----- Starting combined inference for TTST. -----')

    # load ball models
    assert len(ball_model_paths) == 2, 'Please provide exactly 2 balldetection model paths.'
    ball_model1, transform_ball1 = load_ball_model(ball_model_paths[0])
    ball_model2, transform_ball2 = load_ball_model(ball_model_paths[1])

    # load table models
    assert len(table_model_paths) == 2, 'Please provide exactly 2 tabledetection model paths.'
    table_model1, transform_table1 = load_table_model(table_model_paths[0])
    table_model2, transform_table2 = load_table_model(table_model_paths[1])

    # load uplifting model
    uplifting_model, original_transform_uplifting, transform_mode = load_uplifting_model(uplifting_model_path)  # Need orginal transforms for annotated TTST data
    transform_uplifting = _uplifting_transform  # Don't use the standard transform for detections because of image size mismatch

    # Load dataset
    dataset1 = TTST(transform_ball=transform_ball1, transform_table=transform_table1, transforms_uplifting=original_transform_uplifting)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=1)
    dataset2 = TTST(transform_ball=transform_ball2, transform_table=transform_table2, transforms_uplifting=original_transform_uplifting)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)
    print(f'Loaded dataset with {len(dataset1)} samples.')

    denorm_coords = UnNormalizeImgCoords()

    # inference
    errors_table = []
    errors_ball_detection = []
    errors_ball_uplifting = []
    TP, TN, FP, FN = 0, 0, 0, 0
    with tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)) as pbar:
        for i, (stuff1, stuff2) in enumerate(pbar):
            images_ball1, images_table1, r_img, table_img, times, mask, Mint, Mext, spin_class = stuff1
            images_ball2, images_table2, __, __, __, __, __, __, __ = stuff2

            # annotations are only used for evaluation
            # unnormalize img coordinates to pixel coordinates
            r_img = r_img[..., :2].numpy() * np.array([WIDTH, HEIGHT])
            table_img = table_img[..., :2].numpy() * np.array([WIDTH, HEIGHT])  # all coordinates are annotated and visible
            # remove padding and batch dimension
            length = int(mask.squeeze(0).sum().item())
            table_img = table_img.squeeze(0)
            r_img = r_img.squeeze(0)[:length, :]

            # Process ball trajectory
            raw_positions_ball1 = process_trajectory_ball(ball_model1, images_ball1)
            raw_positions_ball2 = process_trajectory_ball(ball_model2, images_ball2)
            fps = 1 / float(times.squeeze(0)[1] - times.squeeze(0)[0])  # I wrote the functions such that I cannot directly use times --> each frame is annotated in ttst, so I can do this
            pred_positions_ball, valid_indices_ball, times_ball = filter_trajectory_ball(raw_positions_ball1, raw_positions_ball2, fps)
            new_T = pred_positions_ball.shape[0]
            # Process table keypoints
            raw_table_keypoints1 = process_trajectory_table(table_model1, images_table1)
            raw_table_keypoints2 = process_trajectory_table(table_model2, images_table2)
            pred_table_keypoints = filter_trajectory_table(raw_table_keypoints1, raw_table_keypoints2)

            # Process uplifting
            ball_coords, table_coords, times, mask = transform_uplifting(pred_positions_ball, pred_table_keypoints, times_ball)
            pred_spin_local, pred_positions_3d = process_trajectory_uplifting(uplifting_model, ball_coords, table_coords, times, mask, transform_mode)

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            if spin_class == TOPSPIN_CLASS:  # Frontspin
                if pred_spin_local[1] > 0:
                    TP += 1
                else:
                    FN += 1
            elif spin_class == BACKSPIN_CLASS:  # Backspin
                if pred_spin_local[1] < 0:
                    TN += 1
                else:
                    FP += 1
            # else: spin annotation was forgotten -> do not include in spin metrics

            # calculate reprojection error for table keypoints
            points_2D = pred_table_keypoints[:, :2]
            visible_indices = pred_table_keypoints[:, 2] == KEYPOINT_VISIBLE  # gt annotations are always visible, no need to check
            points_2D = points_2D[visible_indices]
            table_img = table_img[visible_indices]
            error_table = np.mean(np.linalg.norm(points_2D - table_img, axis=1))
            errors_table.append(error_table)

            # calculate error for ball detections
            detected_ball_2D = pred_positions_ball[:, :2]
            r_img = r_img[valid_indices_ball]
            error_ball = np.mean(np.linalg.norm(detected_ball_2D - r_img, axis=1))
            errors_ball_detection.append(error_ball)

            # calculate reprojection error for ball positions after uplifting
            reprojected_ball_2D = cam2img(world2cam(pred_positions_3d, Mext.squeeze(0).numpy()), Mint.squeeze(0).numpy())
            error_ball_uplifting = np.mean(np.linalg.norm(reprojected_ball_2D - r_img, axis=1))
            errors_ball_uplifting.append(error_ball_uplifting)

            # Update the progress bar with the current metrics (the inference takes ages, and I am already curious)
            f1_plus = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            f1_minus = 2 * TN / (2 * TN + FN + FP) if (2 * TN + FN + FP) > 0 else 0
            pbar.set_postfix({
                'tbl': f'{np.mean(errors_table):.2f}',
                'det': f'{np.mean(errors_ball_detection):.2f}',
                'upl': f'{np.mean(errors_ball_uplifting):.2f}',
                'f1': f'{(f1_plus + f1_minus) / 2:.4f}',
                'acc': f'{(TP + TN) / (TP + TN + FP + FN):.4f}',
                'f1+': f'{f1_plus:.4f}',
                'f1-': f'{f1_minus:.4f}',
            })

            # Spin metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_plus = 2 * TP / (2 * TP + FP + FN)
        f1_minus = 2 * TN / (2 * TN + FN + FP)
        macro_f1 = (f1_plus + f1_minus) / 2
        print(f'Spin classification accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}')

        print(f'Table Keypoint error: {np.mean(errors_table):.2f} ± {np.std(errors_table):.2f} pixels')
        print(f'Ball Detection error: {np.mean(errors_ball_detection):.2f} ± {np.std(errors_ball_detection):.2f} pixels')
        print(f'Ball Uplifting error: {np.mean(errors_ball_uplifting):.2f} ± {np.std(errors_ball_uplifting):.2f} pixels')

        print('----- Finished combined inference for TTST. -----')




def visualize_tthq(ball_model_paths, table_model_paths, uplifting_model_path, plot_table_gt=False):
    '''
    Visualize the predictions of the combined models on a random sample from the TTHQ dataset.
    Args:
        ball_model_paths (tuple): Paths to the 2 balldetection models.
        table_model_paths (tuple): Paths to the 2 tabledetection models.
        uplifting_model_path (str): Path to the Uplifting model.
        plot_table_gt (bool): Whether to print the reprojected 3D table points.
    '''
    # load ball models
    assert len(ball_model_paths) == 2, 'Please provide exactly 2 balldetection model paths.'
    ball_model1, transform_ball1 = load_ball_model(ball_model_paths[0])
    ball_model2, transform_ball2 = load_ball_model(ball_model_paths[1])

    # load table models
    assert len(table_model_paths) == 2, 'Please provide exactly 2 tabledetection model paths.'
    table_model1, transform_table1 = load_table_model(table_model_paths[0])
    table_model2, transform_table2 = load_table_model(table_model_paths[1])

    # load uplifting model
    uplifting_model, __, transform_mode = load_uplifting_model(
        uplifting_model_path)  # Don't use the standard transform because of image size mismatch
    transform_uplifting = _uplifting_transform

    # Load dataset
    dataset1 = TTHQ(transform_ball=transform_ball1, transform_table=transform_table1)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=1)
    dataset2 = TTHQ(transform_ball=transform_ball2, transform_table=transform_table2)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)
    print(f'Loaded dataset with {len(dataset1)} samples.')

    # inference
    with tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)) as pbar:
        for (images_ball1, images_table1, fps, spin_class), (images_ball2, images_table2, __, __) in pbar:
            B, T, C, H, W = images_ball1.shape

            # Process ball trajectory
            raw_positions_ball1 = process_trajectory_ball(ball_model1, images_ball1, move_weights=move_weights)
            raw_positions_ball2 = process_trajectory_ball(ball_model2, images_ball2, move_weights=move_weights)
            pred_positions_ball, valid_indices_ball, times_ball = filter_trajectory_ball(raw_positions_ball1, raw_positions_ball2, fps)
            # Process table keypoints
            raw_table_keypoints1 = process_trajectory_table(table_model1, images_table1, move_weights=move_weights)
            raw_table_keypoints2 = process_trajectory_table(table_model2, images_table2, move_weights=move_weights)
            pred_table_keypoints = filter_trajectory_table(raw_table_keypoints1, raw_table_keypoints2)
            # Process uplifting
            ball_coords, table_coords, times, mask = transform_uplifting(pred_positions_ball, pred_table_keypoints, times_ball)
            pred_spin_local, pred_positions_3d = process_trajectory_uplifting(uplifting_model, ball_coords, table_coords, times, mask, transform_mode, move_weights=move_weights)

            pred_spin_class_str = 'Topspin' if pred_spin_local[1] > 0 else 'Backspin'
            gt_spin_class_str = 'Topspin' if spin_class == TOPSPIN_CLASS else 'Backspin' if spin_class == BACKSPIN_CLASS else 'Unknown'

            # Calibrate camera
            M_int, M_ext = calibrate_camera(pred_table_keypoints)

            # 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the 3D trajectory
            ax.plot(pred_positions_3d[:, 0], pred_positions_3d[:, 1], pred_positions_3d[:, 2], label='Predicted Ball Trajectory', linewidth=3.5, color='orange')
            #ax.scatter(pred_positions_3d[:, 0], pred_positions_3d[:, 1], pred_positions_3d[:, 2], c='red', marker='o')

            # Plot the table
            table_points_3D = np.array(table_points)
            ax.scatter(table_points_3D[:, 0], table_points_3D[:, 1], table_points_3D[:, 2], c='green', marker='o', s=80, label='Table Keypoints')

            # Draw the table's lines
            table_3d = np.array(table_points)
            for conn in table_connections:
                ax.plot([table_3d[conn[0], 0], table_3d[conn[1], 0]],
                        [table_3d[conn[0], 1], table_3d[conn[1], 1]],
                        [table_3d[conn[0], 2], table_3d[conn[1], 2]], c='black')


            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            #ax.set_title('3D Trajectory Visualization')
            #ax.legend()

            # Set aspect ratio to be equal for a more accurate representation
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)

            # Adjust the viewing angle for a better perspective
            ax.view_init(elev=20., azim=130)  # TODO: Adjust it for each video automatically by using the camera matrices from the real video

            plot_radius = 0.5 * max([x_range, y_range, z_range])
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

            plt.show()

            if plot_table_gt:
                # calculate reprojection for table keypoints
                table_points_3D = np.array(table_points)
                reprojected_points = cam2img(world2cam(table_points_3D, M_ext), M_int)

            # plot detected and reprojected keypoints
            plot_image = images_table1[0, T // 2]
            plot_image = plot_transforms({'image': plot_image.numpy()})['image']
            plot_image = eo.rearrange(plot_image, 'c h w -> h w c')
            plot_image = cv2.resize(plot_image, (WIDTH, HEIGHT))

            if plot_table_gt:
                for i, point in enumerate(reprojected_points):  # reprojected gt 3D positions
                    x, y = point
                    cv2.circle(plot_image, (int(x), int(y)), 9, (255, 0, 0), -1)
                    cv2.putText(plot_image, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            for i, coord in enumerate(pred_table_keypoints):  # detected 2D positions
                x, y, v = coord
                if v == KEYPOINT_VISIBLE:
                    cv2.circle(plot_image, (int(x), int(y)), 8, (0, 255, 0), -1)
                    cv2.putText(plot_image, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # plot detected and reprojected trajectory
            reprojected_trajectory = cam2img(world2cam(pred_positions_3d, M_ext), M_int)
            for i, point in enumerate(pred_positions_ball):  # detected 2D positions
                x, y = point
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    cv2.circle(plot_image, (int(x), int(y)), 7, (0, 255, 0), -1)
            for i, point in enumerate(reprojected_trajectory):  # reprojected 3D positions
                x, y = point
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    cv2.circle(plot_image, (int(x), int(y)), 5, (255, 165, 0), -1)
            # write spin class in image with cv2
            # offset = (140, 410)  # where to print the text
            # offset = (0, 0)  # where to print the text
            # cv2.putText(plot_image, f'Predicted Spin: {pred_spin_class_str} {pred_spin_local[1] / (2 * np.pi):.1f}Hz', (offset[0], offset[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(plot_image, f'GT Spin Class: {gt_spin_class_str}', (offset[0], offset[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


            plt.figure(figsize=(60, 30))
            plt.title(f'Predicted Spin: {pred_spin_class_str} {pred_spin_local[1] / (2 * np.pi):.1f}Hz , GT Spin Class: {gt_spin_class_str}',
                      fontsize=36)

            plt.imshow(plot_image)
            #plt.axis('off')
            plt.show()






def main():
    for (ball_model_paths, table_model_paths, uplifting_model_path) in model_paths:
        if args.dataset == 'tthq':
            inference_tthq(ball_model_paths, table_model_paths, uplifting_model_path)
            # visualize_tthq(ball_model_paths, table_model_paths, uplifting_model_path, plot_table_gt=False)
        elif args.dataset == 'ttst':
            inference_ttst(ball_model_paths, table_model_paths, uplifting_model_path)
        else:
            raise ValueError(f'Unknown dataset {args.dataset}. Please choose either "tthq" or "ttst".')

if __name__ == '__main__':
    main()