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
    parser.add_argument('--heatmap_sigma', type=float, default=6.0)
    parser.add_argument('--pretraining', action='store_true')
    parser.add_argument('--folder', type=str, default='debug')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exp_id', type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import einops as eo
import random

from tabledetection.dataset import TableTennisTable, BlurBall, TTHQ
from tabledetection.helper_tabledetection import extract_position_torch_gaussian, KEYPOINT_VISIBLE
from tabledetection.helper_tabledetection import calculate_pck_fixed_tolerance, average_distance, ratio_detected
from tabledetection.helper_tabledetection import update_ema, seed_worker
from tabledetection.helper_tabledetection import weighted_mse_loss, save_model
from tabledetection.helper_tabledetection import WIDTH, HEIGHT
from tabledetection.transforms import get_transform, plot_transforms, resize_transform
from tabledetection.config import TrainConfig

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
    model = get_model(config.model_name, config.image_resolution, config.model_pretraining).to(device)
    ema_model = get_model(config.model_name, config.image_resolution, config.model_pretraining).to(device)
    ema_model = update_ema(model, ema_model, alpha=0.0) # copy weights from model to ema_model

    # initialize the datasets and optimizer
    train_transforms, val_transforms = get_transform('train', config.image_resolution), get_transform('val', config.image_resolution)
    if args.data == "blurball":
        trainset = BlurBall(mode='train', heatmap_sigma=config.heatmap_sigma, transform=train_transforms)
        valset = BlurBall(mode='val', heatmap_sigma=config.heatmap_sigma, transform=val_transforms)
    elif args.data == "tthq":
        trainset = TTHQ(mode='train', heatmap_sigma=config.heatmap_sigma, transform=train_transforms)
        valset = TTHQ(mode='val', heatmap_sigma=config.heatmap_sigma, transform=val_transforms)
    elif args.data == "tabletennis":
        trainset = TableTennisTable(mode='train', heatmap_sigma=config.heatmap_sigma, transform=train_transforms)
        valset = TableTennisTable(mode='val', heatmap_sigma=config.heatmap_sigma, transform=val_transforms)
    else:
        raise RuntimeError(f"Dataset not supported: {args.data}. Possible options: blurball, tthq, tabletennis")

    # always 0 workers in debug mode
    if sys.gettrace() is None:
        num_workers = config.BATCH_SIZE
    else:
        num_workers = 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=min(num_workers, 8),
                                              worker_init_fn=seed_worker, generator=g)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2*config.BATCH_SIZE, shuffle=False, num_workers=min(num_workers, 16),
                                            worker_init_fn=seed_worker, generator=g)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # start training
    loss, best_pck, avg_dist = val(ema_model, valloader, writer, epoch=0)
    # best_pck = 0
    save_model(ema_model, config, epoch=0)
    for epoch in range(1, config.NUM_EPOCHS+1):
        model, ema_model, best_pck = train(model, ema_model, trainloader, optimizer, writer, epoch, config, best_pck, valloader)



def train(model, ema_model, trainloader, optimizer, writer, epoch, config, best_pck, valloader):
    loss_fn = weighted_mse_loss
    model.train()
    iterations = (epoch - 1) * len(trainloader)
    for i, (image, heatmap, table_coords) in enumerate(tqdm(trainloader)):
        image = image.to(device)
        heatmap = heatmap.to(device)
        table_coords = table_coords.to(device)
        visibilities = table_coords[:, :, 2]
        B, __, heat_H, heat_W = heatmap.shape  # H and W are the shape of the heatmap

        optimizer.zero_grad()
        pred = model(image)
        # scale the model output to the size of the heatmap
        pred = torch.nn.functional.interpolate(pred, size=(heat_H, heat_W), mode='bilinear')
        loss = loss_fn(pred, heatmap, visibilities)
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Magic Numbers suck^^
        optimizer.step()

        # update ema model
        ema_model = update_ema(model, ema_model, alpha=config.ema_alpha)

        iterations += 1

        writer.add_scalar('Training/Loss', loss.item(), iterations)

        if iterations % config.VAL_ITERATIONS == 0:  # because train set is pretty small
            loss, pck, avg_dist = val(ema_model, valloader, writer, iterations)

            # save the model with the highest pck
            if pck > best_pck:
                best_pck = pck
                save_model(ema_model, config, epoch=iterations)

    return model, ema_model, best_pck


def val(model, valloader, writer, epoch):
    loss_fn = weighted_mse_loss
    model.eval()
    loss = 0
    gt_pos, pred_pos = [], []
    print('Do validation')
    with torch.no_grad():
        for i, (image, heatmap, table_coords) in enumerate(valloader):
            image = image.to(device)
            heatmap = heatmap.to(device)
            table_coords = table_coords.to(device)
            visibilities = table_coords[:, :, 2]
            pred = model(image)

            # calculate predicted positions scaled to evaluation resolution
            pred_positions = extract_position_torch_gaussian(pred, WIDTH, HEIGHT)
            for b in range(image.shape[0]):
                pred_pos.append(pred_positions[b].copy())
                gt_pos.append(table_coords[b].cpu().numpy().copy())

            # calculate loss
            # for loss in evaluation, we both scale the predicted and ground truth heatmap to the evaluation resolution
            # Note: The ball coordinates are extracted before this resizing
            pred = torch.nn.functional.interpolate(pred, size=(HEIGHT, WIDTH), mode='bilinear')
            heatmap = torch.nn.functional.interpolate(heatmap, size=(HEIGHT, WIDTH), mode='bilinear')
            loss += loss_fn(pred, heatmap, visibilities).item() / len(valloader)

        # resize image to the evaluation resolution (only for plotting)
        image = torch.nn.functional.interpolate(image, size=(HEIGHT, WIDTH), mode='bilinear')

    # plot image overlayed with heatmap
    fig, ax = plt.subplots()
    im = image[0].cpu().numpy()
    tmp = plot_transforms({'image': im})['image']  # Unnormalize the image
    ax.imshow(tmp.transpose(1, 2, 0))
    heat = torch.max(heatmap[0], dim=0)[0]
    tmp2 = np.clip(heat.cpu().numpy().squeeze() * 255.0, 0, 255).astype(np.uint8)
    ax.imshow(tmp2, alpha=0.5)
    ax.set_title('Image with GT Heatmap')
    writer.add_figure('Validation/GT_Heatmap', fig, epoch)
    fig, ax = plt.subplots()
    tmp = plot_transforms({'image': im})['image']
    ax.imshow(tmp.transpose(1, 2, 0))
    heat = torch.max(pred[0], dim=0)[0]
    tmp2 = np.clip(heat.cpu().numpy().squeeze() * 255.0, 0, 255).astype(np.uint8)
    ax.imshow(tmp2, alpha=0.5)
    ax.set_title('Image with Predicted Heatmap')
    writer.add_figure('Validation/Pred_Heatmap', fig, epoch)

    gt_pos = np.asarray(gt_pos)
    pred_pos = np.asarray(pred_pos)
    # calculate PCK
    pck2 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=2)
    pck5 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=5)
    pck10 = calculate_pck_fixed_tolerance(pred_pos, gt_pos, tolerance_pixels=10)
    avg_dist = average_distance(pred_pos, gt_pos)
    ratio = ratio_detected(pred_pos)
    print(f'PCK@5: {pck5:.2f}%')
    print(f'Average distance: {avg_dist:.2f} pixels')
    print(f'Ratio of detected Keypoints: {ratio:.2f}')
    print(loss)
    print('-----------')

    writer.add_scalar('Validation/Loss', loss, epoch)
    writer.add_scalar('Validation/PCK@2px', pck2, epoch)
    writer.add_scalar('Validation/PCK@5px', pck5, epoch)
    writer.add_scalar('Validation/PCK@10px', pck10, epoch)
    writer.add_scalar('Validation/AverageDistance', avg_dist, epoch)
    writer.add_scalar('Validation/RatioDetectedKeypoints', ratio, epoch)

    return loss, pck5, avg_dist



def get_model(model_name, resolution, pretraining):
    '''
    Get the model based on the model name.
    Args:
        model_name: Name of the model.
        resolution: Resolution of the input images.
        pretraining: Whether to use pretraining. Boolean.
    '''
    if 'segformerpp' in model_name:
        from tabledetection.models.segformer_pp import Segformer_pp
        assert len(model_name.split('_')) == 2, "Model name should be in the format 'segformerpp_<model_size>'"
        model_size = model_name.split('_')[1]
        model = Segformer_pp(model_size=model_size, pretraining=pretraining)
    elif model_name == 'vitpose':
        from tabledetection.models.vitpose import VitPose
        model = VitPose(model_size='small', pretraining=pretraining, resolution=resolution)
    elif model_name == 'hrnet':
        from tabledetection.models.hrnet import MyHRNet
        model = MyHRNet(pretraining=pretraining, resolution=resolution)
    else:
        raise ValueError(f"Model {model_name} not implemented")
    return model

if __name__ == "__main__":
    config = TrainConfig(lr=args.lr, model_name=args.model_name, heatmap_sigma=args.heatmap_sigma,
                         pretraining=args.pretraining, dataset_name=args.data, exp_id=args.exp_id,
                         folder=args.folder, debug=args.debug)
    run(config)