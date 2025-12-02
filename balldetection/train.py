import os
import sys

if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='segformerpp_b2')
    parser.add_argument('--data', type=str, default='tthq')
    parser.add_argument('--in_frames', type=int, default=3)
    parser.add_argument('--heatmap_sigma', type=float, default=6.0)
    parser.add_argument('--not_use_invis', action='store_true')
    parser.add_argument('--pretraining', action='store_true')
    parser.add_argument('--folder', type=str, default='debug')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exp_id', type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import einops as eo
import random
from copy import deepcopy

from balldetection.dataset import TableTennisBall, BlurBall, TTHQ
from balldetection.helper_balldetection import extract_position_torch_gaussian, acc_visible_invisible_keypoints, BALL_VISIBLE
from balldetection.helper_balldetection import calculate_pck_fixed_tolerance, average_distance, distance_to_streak, ratio_visible_detected
from balldetection.helper_balldetection import update_ema, seed_worker
from balldetection.helper_balldetection import weighted_mse_loss, save_model
from balldetection.helper_balldetection import WIDTH, HEIGHT
from balldetection.transforms import get_transform, plot_transforms, resize_transform
from balldetection.config import TrainConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(config):
    '''
    Main function to run the training and validation of the model.
    Args:
        config (TrainConfig): Config object containing hyperparameters and settings.
    '''
    # set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    writer = SummaryWriter(log_dir=config.logs_path)

    # load model, keep track of model weights as ema
    model = get_model(config.model_name, config.in_frames, config.image_resolution, config.model_pretraining).to(device)
    ema_model = get_model(config.model_name, config.in_frames, config.image_resolution, config.model_pretraining).to(device)
    ema_model = update_ema(model, ema_model, alpha=0.0) # copy weights from model to ema_model

    # initialize the datasets and optimizer
    train_transforms, val_transforms = get_transform('train', config.image_resolution), get_transform('val', config.image_resolution)
    if args.data == "blurball":
        trainset = BlurBall(mode='train', heatmap_sigma=config.heatmap_sigma, in_frames=config.in_frames, transform=train_transforms, use_invisible=config.use_invis)
        valset = BlurBall(mode='val', heatmap_sigma=config.heatmap_sigma, in_frames=config.in_frames, transform=val_transforms, use_invisible=config.use_invis)
    elif args.data == "tthq":
        trainset = TTHQ(mode='train', heatmap_sigma=config.heatmap_sigma, in_frames=config.in_frames, transform=train_transforms, use_invisible=config.use_invis)
        valset = TTHQ(mode='val', heatmap_sigma=config.heatmap_sigma, in_frames=config.in_frames, transform=val_transforms, use_invisible=config.use_invis)
    elif args.data == "tabletennis":
        trainset = TableTennisBall(mode='train', heatmap_sigma=config.heatmap_sigma, in_frames=config.in_frames, transform=train_transforms, use_invisible=config.use_invis)
        valset = TableTennisBall(mode='val', heatmap_sigma=config.heatmap_sigma, in_frames=config.in_frames, transform=val_transforms, use_invisible=config.use_invis)
    else:
        raise RuntimeError(f"Dataset not supported: {args.data}. Possible options: blurball, tthq")
    # always 0 workers in debug mode
    if sys.gettrace() is None:
        num_workers = int(config.BATCH_SIZE)
    else:
        num_workers = 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=min(num_workers, 8),
                                              worker_init_fn=seed_worker, generator=g, persistent_workers=True if num_workers > 0 else False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2*config.BATCH_SIZE, shuffle=False, num_workers=min(num_workers, 16),
                                            worker_init_fn=seed_worker, generator=g, persistent_workers=True if num_workers > 0 else False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # start training
    loss, best_pck, avg_dist = val(ema_model, valloader, writer, epoch=0)
    save_model(ema_model, config, epoch=0)
    for epoch in range(1, config.NUM_EPOCHS+1):
        model, ema_model, best_pck = train(model, ema_model, trainloader, optimizer, writer, epoch, config, best_pck, valloader)


def train(model, ema_model, trainloader, optimizer, writer, epoch, config, best_pck, valloader):
    loss_seg = weighted_mse_loss
    loss_cls = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.asarray([10., 1.])).float().to(device))
    weight_seg = 1 # min(epoch / 5, 1)
    weight_cls = 0 # 2 - weight_seg
    model.train()
    iterations = (epoch - 1) * len(trainloader)
    for i, stuff in enumerate(tqdm(trainloader)):
        image, heatmap, ball_coords, _, _, vis = deepcopy(stuff)
        del stuff
        image = image.to(device)
        heatmap = heatmap.to(device)
        ball_coords = ball_coords.to(device)
        vis = vis.to(device)
        B, __, heat_H, heat_W = heatmap.shape  # H and W are the shape of the heatmap

        optimizer.zero_grad()
        pred_seg, pred_cls = model(image)
        # scale the model output to the size of the heatmap
        pred_seg = torch.nn.functional.interpolate(pred_seg, size=(heat_H, heat_W), mode='bilinear')
        loss_s = loss_seg(pred_seg, heatmap)
        if pred_cls is not None:
            loss_c = loss_cls(pred_cls, vis)
        else:
            loss_c = torch.tensor(0)
        loss = weight_seg * loss_s + weight_cls * loss_c
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Magic Numbers suck^^
        optimizer.step()

        # update ema model
        ema_model = update_ema(model, ema_model, alpha=config.ema_alpha)

        iterations += 1

        writer.add_scalar('Training/Loss', weight_seg * loss_s.item(), iterations)
        writer.add_scalar('Training/LossCls', weight_cls * loss_c.item(), iterations)
        writer.add_scalar('Training/LossTotal', loss.item(), iterations)

        if iterations % config.VAL_ITERATIONS == 0:
            loss, pck, avg_dist = val(ema_model, valloader, writer, iterations)

            # save the model with the highest pck
            if pck > best_pck:
                best_pck = pck
                save_model(ema_model, config, epoch=iterations)

    return model, ema_model, best_pck


def val(model, valloader, writer, epoch):
    loss_seg = torch.nn.MSELoss()
    loss_cls = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_s, loss_c = 0, 0
    gt_pos, pred_pos = [], []
    gt_max, gt_min = [], []
    gt_vis, pred_vis = [], []
    print('Do validation')
    with torch.no_grad():
        for i, stuff in enumerate(valloader):
            image, heatmap, ball_coords, max_coords, min_coords, vis = deepcopy(stuff)
            del stuff
            image = image.to(device)
            heatmap = heatmap.to(device)
            ball_coords = ball_coords.to(device)
            vis = vis.to(device)
            pred_seg, pred_cls = model(image)

            # calculate predicted positions scaled to evaluation resolution
            pred_positions = extract_position_torch_gaussian(pred_seg, WIDTH, HEIGHT)
            for b in range(image.shape[0]):
                pred_pos.append(pred_positions[b])
                gt_pos.append(ball_coords[b].cpu().numpy().copy())
                gt_max.append(max_coords[b].cpu().numpy().copy())
                gt_min.append(min_coords[b].cpu().numpy().copy())
                gt_vis.append(vis[b].cpu().numpy().copy())
                if pred_cls is not None:
                    pred_vis.append(pred_cls[b].cpu().numpy().copy())

            # calculate loss
            # for loss in evaluation, we both scale the predicted and ground truth heatmap to the evaluation resolution
            # Note: The ball coordinates are extracted before this resizing
            pred_seg = torch.nn.functional.interpolate(pred_seg, size=(HEIGHT, WIDTH), mode='bilinear')
            heatmap = torch.nn.functional.interpolate(heatmap, size=(HEIGHT, WIDTH), mode='bilinear')
            loss_s += loss_seg(pred_seg, heatmap).item() / len(valloader)
            if pred_cls is not None:
                loss_c += loss_cls(pred_cls, vis).item() / len(valloader)

        # resize image to the evaluation resolution (only for plotting)
        image = torch.nn.functional.interpolate(image, size=(HEIGHT, WIDTH), mode='bilinear')

    # plot image overlayed with heatmap
    fig, ax = plt.subplots()
    if config.in_frames == 3:
        im = image[0, 3:6].cpu().numpy()
    else:
        im = image[0].cpu().numpy()
    tmp = plot_transforms({'image': im})['image']  # Unnormalize the image
    ax.imshow(tmp.transpose(1, 2, 0))
    tmp2 = np.clip(heatmap[0].cpu().numpy().squeeze() * 255.0, 0, 255).astype(np.uint8)
    ax.imshow(tmp2, alpha=0.5)
    ax.set_title('Image with GT Heatmap')
    writer.add_figure('Validation/GT_Heatmap', fig, epoch)
    fig, ax = plt.subplots()
    tmp = plot_transforms({'image': im})['image']
    ax.imshow(tmp.transpose(1, 2, 0))
    tmp2 = np.clip(pred_seg[0].cpu().numpy().squeeze() * 255.0, 0, 255).astype(np.uint8)
    ax.imshow(tmp2, alpha=0.5)
    ax.set_title('Image with Predicted Heatmap')
    writer.add_figure('Validation/Pred_Heatmap', fig, epoch)

    # visible balls in gt
    gt_vis = np.asarray(gt_vis)
    gt_vis_mask = (gt_vis == BALL_VISIBLE)
    # calculate PCK
    pred_pos = np.array(pred_pos, dtype=np.float32)
    gt_pos = np.array(gt_pos, dtype=np.float32)
    gt_min = np.array(gt_min, dtype=np.float32)
    gt_max = np.array(gt_max, dtype=np.float32)
    pck2 = calculate_pck_fixed_tolerance(pred_pos[gt_vis_mask], gt_pos[gt_vis_mask], gt_min[gt_vis_mask], gt_max[gt_vis_mask], tolerance_pixels=2)
    pck5 = calculate_pck_fixed_tolerance(pred_pos[gt_vis_mask], gt_pos[gt_vis_mask], gt_min[gt_vis_mask], gt_max[gt_vis_mask], tolerance_pixels=5)
    pck10 = calculate_pck_fixed_tolerance(pred_pos[gt_vis_mask], gt_pos[gt_vis_mask], gt_min[gt_vis_mask], gt_max[gt_vis_mask], tolerance_pixels=10)
    # ratio = ratio_visible_detected(pred_pos[gt_vis_mask])
    # ratios_vis, ratios_invis, num_vis, num_invis, thresholds = acc_visible_invisible_keypoints(pred_pos, gt_vis)
    avg_dist = average_distance(pred_pos[gt_vis_mask][:, :2], gt_pos[gt_vis_mask])
    dist_streak = distance_to_streak(pred_pos[gt_vis_mask][:, :2], gt_pos[gt_vis_mask], gt_min[gt_vis_mask], gt_max[gt_vis_mask])
    print(f'PCK@5: {pck5:.2f}')
    print(f'Average distance: {avg_dist:.2f} pixels')
    # for ratio_vis, ratio_invis, threshold in zip(ratios_vis, ratios_invis, thresholds):
    #     print(f'Correctly visible Keypoints: {ratio_vis:.2f}% of {num_vis} for threshold {threshold}')
    #     print(f'Correctly invisible Keypoints: {ratio_invis:.2f}% of {num_invis} for threshold {threshold}')
    #     writer.add_scalar(f'ValidationVisInvis/CorrVisKeypoints@{threshold}', ratio_vis, epoch)
    #     writer.add_scalar(f'ValidationVisInvis/CorrInvisKeypoints@{threshold}', ratio_invis, epoch)
    if len(pred_vis) > 0:
        acc_vis, acc_invis = acc_visible_invisible_keypoints(np.asarray(pred_vis), gt_vis)
        writer.add_scalar(f'Validation/CorrVisKeypoints', acc_vis, epoch)
        writer.add_scalar(f'Validation/CorrInvisKeypoints', acc_invis, epoch)
    print(f'Distance to streak: {dist_streak:.2f} pixels')
    print(loss_s)
    print('-----------')

    writer.add_scalar('Validation/Loss', loss_s, epoch)
    writer.add_scalar('Validation/LossCls', loss_c, epoch)
    writer.add_scalar('Validation/PCK@2px', pck2, epoch)
    writer.add_scalar('Validation/PCK@5px', pck5, epoch)
    writer.add_scalar('Validation/PCK@10px', pck10, epoch)
    writer.add_scalar('Validation/AverageDistance', avg_dist, epoch)
    writer.add_scalar('Validation/DistanceToStreak', dist_streak, epoch)


    return loss_s, pck5, avg_dist



def get_model(model_name, in_frames, resolution, pretraining):
    '''
    Get the model based on the model name.
    Args:
        model_name: Name of the model.
        in_frames: Number of input frames.
        resolution: Resolution of the input images.
        pretraining: Whether to use pretraining. Boolean.
    '''
    if 'segformerpp' in model_name:
        from balldetection.models.segformer_pp import Segformer_pp
        assert len(model_name.split('_')) == 2, "Model name should be in the format 'segformerpp_<model_size>'"
        model_size = model_name.split('_')[1]
        model = Segformer_pp(in_frames=in_frames, model_size=model_size, pretraining=pretraining)
    elif model_name == 'vitpose':
        from balldetection.models.vitpose import VitPose
        model = VitPose(in_frames=in_frames, model_size='small', pretraining=pretraining, resolution=resolution)
    elif model_name == 'wasb':
        from balldetection.models.wasb import WASBNet
        model = WASBNet(in_frames=in_frames, resolution=resolution, pretraining=pretraining)
    else:
        raise ValueError(f"Model {model_name} not implemented")
    return model

if __name__ == "__main__":
    config = TrainConfig(lr=args.lr, model_name=args.model_name, in_frames=args.in_frames, heatmap_sigma=args.heatmap_sigma,
                         dataset_name=args.data, pretraining=args.pretraining, exp_id=args.exp_id,
                        folder=args.folder, debug=args.debug, use_invis=(not args.not_use_invis))
    run(config)