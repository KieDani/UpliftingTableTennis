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

import paths

from inference.utils import HEIGHT, WIDTH, KEYPOINT_VISIBLE, KEYPOINT_INVISIBLE
from inference.utils import get_uplifting_model
from inference.dataset import TOPSPIN_CLASS, BACKSPIN_CLASS

from uplifting.data import RealInferenceDataset as TTSTDataset
from uplifting.data import TT3DDataset
from uplifting.transformations import get_transforms, UnNormalizeImgCoords
from uplifting.helper import transform_rotationaxes, cam2img, world2cam
from uplifting.helper import HEIGHT as SYNTHETIC_HEIGHT, WIDTH as SYNTHETIC_WIDTH


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
model_paths = [
    os.path.join(paths.weights_path, 'inference_uplifting', 'kienzleetal', 'model.pt'),
    os.path.join(paths.weights_path, 'inference_uplifting', 'mixed', 'model.pt'),
    os.path.join(paths.weights_path, 'inference_uplifting', 'ours', 'model.pt'),
]

def load_model(model_path):
    '''
    Load the uplifting model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        model (torch.nn.Module): Loaded uplifting model.
        transform (callable): Transformation function for input images.
        transform_mode (str): Spin transformation mode used during training ('global' or 'local').
    '''
    loaded_dict = torch.load(model_path, weights_only=False)
    model_name = loaded_dict['additional_info']['name']
    model_size = loaded_dict['additional_info']['size']
    identifier = loaded_dict['identifier']
    tabletoken_mode = loaded_dict['additional_info']['tabletoken_mode']
    time_rotation = loaded_dict['additional_info']['time_rotation']
    transform_mode = loaded_dict['additional_info']['transform_mode']
    randdet_prob, randmiss_prob, tablemiss_prob =loaded_dict['additional_info']['randdet_prob'], loaded_dict['additional_info']['randmiss_prob'], loaded_dict['additional_info']['tablemiss_prob']
    uplifting_model = get_uplifting_model(model_name, size=model_size, mode=tabletoken_mode, time_rotation=time_rotation)
    uplifting_model.load_state_dict(loaded_dict['model_state_dict'])
    uplifting_model.eval()
    print(f'Loaded Uplifting model: {model_name} with size {model_size}, tabletoken_mode: {tabletoken_mode}, time_rotation: {time_rotation}, transform_mode: {transform_mode}')
    print(f'Noise settings during training - randdet_prob: {randdet_prob}, randmiss_prob: {randmiss_prob}, tablemiss_prob: {tablemiss_prob}')
    # Use the standard transforms for evaluation because data comes from the dataloader, not the detection models
    transform = get_transforms(config=None, mode='test')  # config=None only works because mode=test
    return uplifting_model, transform, transform_mode


def inference_tt3d(model_path):
    '''
    Run inference on the TT3D dataset. Calculate 3D trajectory metrics because we have 3D gt for this dataset.
    Three views are possible for TT3D. Calculate metrics for each view and all views together.
    There is also the noise option for the detections. Evaluate both with and without noise.
    Args:
        model_path (str): Path to the saved model.
    '''
    views = ['back', 'side', 'oblique']
    # initialize metrics functions
    metric_fn = lambda x, y, mask: np.sum(np.sqrt(np.sum((x - y) ** 2, axis=-1)) * mask, axis=1) / np.sum(mask, axis=1)

    # load the model
    model, __, transform_mode = load_model(model_path)
    model.to(device)

    # run inference
    print('Running inference on TT3D dataset...')
    for noise in [True, False]:
        metrics = {view: None for view in views}
        for view in views:
            dataset = TT3DDataset(view=view, noise=noise)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=min(8, BATCH_SIZE), pin_memory=True)
            for i, stuff in enumerate(tqdm(dataloader)):
                r_img, table_img, mask, r_world, times, hits, Mint, Mext, framerate = stuff
                r_img = r_img.to(device)  # (B, T, 2)
                table_img = table_img.to(device)  # (B, 13, 2)
                mask = mask.to(device)  # (B, T)
                r_world = r_world.to(device)  # (B, T, 3)
                times = times.to(device)  # (B, T)

                with torch.no_grad():
                    # forward pass
                    B, T, D = r_img.shape
                    pred_rotation, pred_position = model(r_img, table_img, mask, times)

                    # not needed here because we cannot calculate a spin metric. But better to have it than to forget it
                    if transform_mode == 'global':
                        pred_rotation = transform_rotationaxes(pred_rotation, r_world)

                    # calculate metrics
                    m = metric_fn(pred_position.cpu().numpy(), r_world.cpu().numpy(), mask.cpu().numpy())
                    if metrics[view] is None:
                        metrics[view] = m
                    else:
                        metrics[view] = np.concatenate((metrics[view], m), axis=0)

        for view in views:
            print(f'Noise: {noise}, View: {view}, Mean 3D Position Error: {np.mean(metrics[view]) * 100:.2f} cm, Std: {np.std(metrics[view]) * 100:.2f} cm')
        all_metrics = np.concatenate([metrics[view] for view in views])
        print(f'Noise: {noise}, All Views, Mean 3D Position Error: {np.mean(all_metrics) * 100:.2f} cm, Std: {np.std(all_metrics) * 100:.2f} cm')
        #print(metrics)

    print('Finished inference on TT3D dataset.')



def inference_ttst(model_path, special_transform):
    '''
    Run inference on the TTSD dataset. Calculate 2D trajectory metrics because we have only 2D gt for this dataset. Also calculate spin metrics.
    Args:
        model_path (str): Path to the saved model.
        special_transform (callable): Special transform to apply to the input (e.g. changing fps, adding noise, dropping detections).

    '''
    # initialize metrics functions
    metric_fn = lambda x, y, mask: np.sum(np.sqrt(np.sum((x - y) ** 2, axis=-1)) * mask, axis=1) / np.sum(mask, axis=1)

    # load the model
    model, transform, transform_mode = load_model(model_path)
    model.to(device)
    denorm = UnNormalizeImgCoords()

    print(f'Applying special transform: {special_transform.__class__.__name__}')

    # run inference
    print('Running inference on TTSD dataset...')
    dataset = TTSTDataset(mode='test', transforms=transform)
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(8, batch_size), pin_memory=True)
    metrics = None
    TP, TN, FP, FN = 0, 0, 0, 0  # For frontspin vs backspin classification
    number = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            r_img, table_img, mask, times, hits, Mint, Mext, spin_class = data
            r_img, table_img, mask, times = r_img.to(device), table_img.to(device), mask.to(device), times.to(device)
            Mint, Mext = Mint.to(device), Mext.to(device)
            B, T, D = r_img.shape

            # apply special transform (e.g. changing fps, adding noise, dropping detections)
            data_dict = {'r_img': r_img, 'table_img': table_img, 'mask': mask, 'times': times}
            data_dict = special_transform(data_dict)  # Apply special transform (e.g. changing fps, adding noise, dropping detections)
            r_img, table_img, mask, times = data_dict['r_img'], data_dict['table_img'], data_dict['mask'], data_dict['times']

            # forward pass
            pred_rotation, pred_position = model(r_img, table_img, mask, times)
            # transform prediction into local coordinate system
            if transform_mode == 'global':
                pred_rotation_local = transform_rotationaxes(pred_rotation, pred_position.clone())
            else:
                pred_rotation_local = pred_rotation

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            for b in range(B):
                # binary metrics
                if spin_class[b] == TOPSPIN_CLASS:  # Frontspin
                    number += 1
                    if pred_rotation_local[b, 1] > 0:
                        TP += 1
                    else:
                        FN += 1
                elif spin_class[b] == BACKSPIN_CLASS:  # Backspin
                    number += 1
                    if pred_rotation_local[b, 1] < 0:
                        TN += 1
                    else:
                        FP += 1
                # else: spin annotation was forgotten -> do not include in spin metrics

            # denormalization of ground truth image coordinates to calculate the 2D metric
            data_gt = denorm({'r_img': r_img.cpu().numpy(), 'table_img': table_img.cpu().numpy()})
            r_img, table_img = data_gt['r_img'], data_gt['table_img']

            # reproject predicted 3D positions to 2D image coordinates
            pred_pos_2D = cam2img(world2cam(pred_position, Mext), Mint)  # (B, T, 2)

            # resize the coordinates to correct resolution for evaluation
            if (HEIGHT, WIDTH) != (SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH):
                scale_x, scale_y = WIDTH / SYNTHETIC_WIDTH, HEIGHT / SYNTHETIC_HEIGHT
                r_img[0] = (r_img[0] + 0.5) * scale_x - 0.5
                r_img[1] = (r_img[1] + 0.5) * scale_y - 0.5
                pred_pos_2D[0] = (pred_pos_2D[0] + 0.5) * scale_x - 0.5
                pred_pos_2D[1] = (pred_pos_2D[1] + 0.5) * scale_y - 0.5

            # calculate metrics
            metric = metric_fn(pred_pos_2D.cpu().numpy(), r_img, mask.cpu().numpy())
            if metrics is None:
                metrics = metric
            else:
                metrics = np.concatenate((metrics, metric), axis=0)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_plus = 2 * TP / (2 * TP + FP + FN)
    f1_minus = 2 * TN / (2 * TN + FN + FP)
    macro_f1 = (f1_plus + f1_minus) / 2

    print(f'Mean 2D Position Error: {np.mean(metrics):.2f} pixels, Std: {np.std(metrics):.2f} pixels')
    print(f'Mean normed 2D Position Error: {np.mean(metrics) / np.sqrt(HEIGHT**2 + WIDTH**2):.4f}, Std: {np.std(metrics) / np.sqrt(HEIGHT**2 + WIDTH**2):.4f}')
    print(f'Spin Classification - Accuracy: {accuracy:.4f}')
    print(f'Spin Classification - Macro F1 Score: {macro_f1:.4f}')

    print('Finished inference on TTSD dataset.')




class HalfFPS_transform:
    '''Simlate a 50% drop in frame rate by dropping every second detection.'''
    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the data with keys 'r_img', 'table_img', 'mask', 'times'
        Returns:
            data (dict): Transformed data with half the frame rate.
        '''
        r_img, mask, times = data['r_img'], data['mask'], data['times']
        B, T, D = r_img.shape
        new_r_img = torch.zeros((B, T, D), device=r_img.device)
        new_mask = torch.zeros_like(mask, dtype=mask.dtype, device=mask.device)  # 0 == False
        new_times = torch.zeros((B, T), device=times.device)
        for b in range(B):
            curr_index = 0
            length = int(torch.sum(mask[b]).item())
            for t in range(length):
                if t % 2 == 0:
                    new_r_img[b, curr_index] = r_img[b, t]
                    new_mask[b, curr_index] = mask[b, t]
                    new_times[b, curr_index] = times[b, t]
                    curr_index += 1
        data['r_img'], data['mask'], data['times'] = new_r_img, new_mask, new_times
        return data



class DropBall_transform:
    '''Simulate some missed ball detections by randomly dropping some detections.'''
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(42)  # for reproducibility

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the data with keys 'r_img', 'table_img', 'mask', 'times'
        Returns:
            data (dict): Transformed data with some detections dropped.
        '''
        r_img, mask, times = data['r_img'], data['mask'], data['times']
        B, T, D = r_img.shape

        new_r_img = torch.zeros((B, T, D), device=r_img.device)
        new_mask = torch.zeros_like(mask, dtype=mask.dtype, device=mask.device)  # 0 == False
        new_times = torch.zeros((B, T), device=times.device)

        for b in range(B):
            curr_index = 0
            length = int(torch.sum(mask[b]).item())
            for t in range(length):
                if not (self.rng.random() < self.drop_prob):  # If we do not drop the detection
                    new_r_img[b, curr_index] = r_img[b, t]
                    new_mask[b, curr_index] = mask[b, t]
                    new_times[b, curr_index] = times[b, t]
                    curr_index += 1

        data['r_img'], data['mask'], data['times'] = new_r_img, new_mask, new_times
        return data


class DropTable_transform:
    '''Simulate some missed table detections by randomly dropping some detections.'''
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(42)  # for reproducibility

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the data with keys 'r_img', 'table_img', 'mask', 'times'
        Returns:
            data (dict): Transformed data with some table detections dropped.
        '''
        table_img = data['table_img'].clone()
        B, N, D = table_img.shape

        for b in range(B):
            for n in range(N):
                if self.rng.random() < self.drop_prob:  # If we do drop the detection
                    table_img[b, n, 2] = KEYPOINT_INVISIBLE  # Set visibility to invisible
                    # set random coordinates
                    table_img[b, n, :2] = torch.tensor(self.rng.uniform(low=-0.99, high=0.99, size=(2,)), device=table_img.device, dtype=table_img.dtype)

        data['table_img'] = table_img
        return data


class Drop_transform:
    '''Simulate some missed ball and table detections by randomly dropping some detections.'''
    def __init__(self, drop_ball_prob=0.1, drop_table_prob=0.1):
        self.drop_ball_transform = DropBall_transform(drop_prob=drop_ball_prob)
        self.drop_table_transform = DropTable_transform(drop_prob=drop_table_prob)

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the data with keys 'r_img', 'table_img', 'mask', 'times'
        '''
        data = self.drop_ball_transform(data)
        data = self.drop_table_transform(data)
        return data


class Identity_transform:
    '''Identity transform that does nothing.'''
    def __call__(self, data):
        return data


class Combine_transform:
    '''Combine Drop and HalfFPS transforms.'''
    def __init__(self, drop_ball_prob=0.1, drop_table_prob=0.1):
        self.drop_transform = Drop_transform(drop_ball_prob=drop_ball_prob, drop_table_prob=drop_table_prob)
        self.half_fps_transform = HalfFPS_transform()
    def __call__(self, data):
        data = self.half_fps_transform(data)
        data = self.drop_transform(data)
        return data




def main():
    for model_path in model_paths:
        print('========================================')
        # inference_tt3d(model_path)
        for special_transform in [Identity_transform(), HalfFPS_transform(), Drop_transform(0.1, 0.1), Combine_transform(0.1, 0.1)]:
            inference_ttst(model_path, special_transform)
        print('========================================')


if __name__ == '__main__':
    main()
    pass



