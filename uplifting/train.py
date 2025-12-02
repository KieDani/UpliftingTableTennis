import os
if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--blur_strength', type=float, default=0.4)
    parser.add_argument('--stop_prob', type=float, default=0.5)
    parser.add_argument('--randdet_prob', type=float, default=0.00)
    parser.add_argument('--randmiss_prob', type=float, default=0.05)
    parser.add_argument('--tablemiss_prob', type=float, default=0.05)
    parser.add_argument('--randomize_std', type=float, default=2)
    parser.add_argument('--model_name', type=str, default='connectstage')
    parser.add_argument('--model_size', type=str, default='large')
    parser.add_argument('--token_mode', type=str, default='dynamic')
    parser.add_argument('--transform_mode', type=str, default='global')
    parser.add_argument('--time_rotation', type=str, default='new')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
if __name__ == '__main__' and args.debug:
    torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import roc_auc_score

from uplifting.helper import SummaryWriter, seed_worker
from uplifting.helper import binary_metrics
from uplifting.helper import update_ema
from uplifting.helper import create_confusion_matrix, transform_rotationaxes
from uplifting.helper import save_model, WIDTH, HEIGHT
from uplifting.helper import world2cam, cam2img
from uplifting.data import TableTennisDataset, RealInferenceDataset, TOPSPIN_CLASS, BACKSPIN_CLASS
from uplifting.transformations import get_transforms, UnNormalizeImgCoords
from uplifting.model import get_model
from uplifting.config import TrainConfig

device = 'cuda:0'
debug = False

def run(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    logs_path = config.get_logs_path(debug)
    writer = SummaryWriter(logs_path)

    model = get_model(config.name, config.size, config.tabletoken_mode, config.time_rotation).to(device)
    model_ema = get_model(config.name, config.size, config.tabletoken_mode, config.time_rotation).to(device)
    model_ema = update_ema(model, model_ema, 0) # copies the model to model_ema completely

    num_workers = 0 if debug else min(config.BATCH_SIZE, 16)
    train_transforms = get_transforms(config, 'train')
    val_transforms = get_transforms(config, 'val')
    trainset = TableTennisDataset('train', transforms=train_transforms)
    valset = TableTennisDataset('val', transforms=val_transforms)
    valset_real = RealInferenceDataset(transforms=val_transforms, mode='val')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True,
                                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config.BATCH_SIZE, shuffle=False,
                                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    valloader_real = torch.utils.data.DataLoader(valset_real, batch_size=4, shuffle=False,
                                                    num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_metric_trajectory, best_metric_spin, best_metric_synthetic = 1e8, 0, 1e8
    threshold_trajectory_metric = 0.007
    best_metric_spin_mixed, best_metric_trajectory_mixed = 0, 1e8
    val(model_ema, valloader, writer, -1, device, config)
    val_real(model_ema, valloader_real, writer, -1, device, config)
    for epoch in range(config.NUM_EPOCHS):
        model_ema = train(model, model_ema, trainloader, optimizer, writer, epoch, device, config)
        metric_synthetic = val(model_ema, valloader, writer, epoch, device, config)
        metric_trajectory, metric_spin = val_real(model_ema, valloader_real, writer, epoch, device, config)
        # save model if metric improved
        if metric_trajectory < best_metric_trajectory:
            best_metric_trajectory = metric_trajectory
            save_model(model_ema, config, epoch, debug, name='model_trajectory.pt')
        if metric_spin >= best_metric_spin:  # larger F1 is better
            best_metric_spin = metric_spin
            save_model(model_ema, config, epoch, debug, name='model_spin.pt')
        if metric_synthetic < best_metric_synthetic:
            best_metric_synthetic = metric_synthetic
            save_model(model_ema, config, epoch, debug, name='model_synthetic.pt')
        if metric_trajectory <= threshold_trajectory_metric:
            if metric_spin > best_metric_spin_mixed:
                best_metric_spin_mixed = metric_spin
                save_model(model_ema, config, epoch, debug, name='model.pt')
                best_metric_trajectory_mixed = metric_trajectory  # update with value of saved model
            elif metric_spin == best_metric_spin_mixed: # important case, since 100% is realistic in our setting
                if metric_trajectory < best_metric_trajectory_mixed:  # If spin metric is already maxed out, decide using trajectory metric
                    best_metric_trajectory_mixed = metric_trajectory
                    save_model(model_ema, config, epoch, debug, name='model.pt')


def train(model, model_ema, trainloader, optimizer, writer, epoch, device, config):
    # L2 loss for the whole vector
    loss_fn = lambda angle, pred_angle: torch.sum(torch.sqrt(torch.sum((angle-pred_angle)**2, dim=1)))

    # If we are using the multistage model, we want to prevent that the rotation influences the learning of the positions
    if config.name == 'multistage':
        model.full_backprop = False

    model.train()
    iteration = epoch * len(trainloader)
    for i, data in enumerate(tqdm(trainloader)):
        r_img, table_img, mask, r_world, rotation, times, hits, __, __ = data
        r_img, table_img, mask, rotation = r_img.to(device), table_img.to(device), mask.to(device), rotation.to(device)
        r_world, times = r_world.to(device), times.to(device)

        optimizer.zero_grad()
        pred_rotation, pred_position = model(r_img, table_img, mask, times)
        # If network output is in ball's coordinate system, transform the ground truth to the local coordinate system
        if config.transform_mode == 'local':
            rotation = transform_rotationaxes(rotation, r_world)
        loss_rot = loss_fn(pred_rotation, rotation)
        loss_pos = torch.sum(torch.nn.functional.mse_loss(pred_position, r_world, reduction='none') * mask.unsqueeze(-1)) / torch.sum(mask)
        loss = loss_rot + loss_pos
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # gradient clipping
        optimizer.step()

        model_ema = update_ema(model, model_ema, config.ema_decay)

        writer.add_scalar('train/loss', loss.item(), iteration+i)
        writer.add_scalar('train/loss rotation', loss_rot.item(), iteration+i)
        writer.add_scalar('train/loss position', loss_pos.item(), iteration+i)

    return model_ema


def val(model, valloader, writer, epoch, device, config):
    loss_fn = torch.nn.MSELoss()
    metric_fn = lambda x, y: torch.sum(torch.sqrt(torch.sum((x-y)**2, dim=1)))
    metricx_fn = lambda x, y: torch.sum(torch.abs(x[:, 0] - y[:, 0]))
    metricy_fn = lambda x, y: torch.sum(torch.abs(x[:, 1] - y[:, 1]))
    metricz_fn = lambda x, y: torch.sum(torch.abs(x[:, 2] - y[:, 2]))
    metricabs_fn = lambda x, y: torch.sum(torch.abs(torch.norm(x, dim=1) - torch.norm(y, dim=1)))
    metricangle_fn = lambda x, y: torch.sum(torch.rad2deg(torch.acos(torch.einsum('bi, bi -> b', x, y) / (torch.norm(x, dim=1) * torch.norm(y, dim=1)))))
    metric_pos_fn = lambda x, y, mask: torch.sum(torch.sum(torch.sqrt(torch.sum((x-y)**2, dim=-1)) * mask, dim=1) / torch.sum(mask, dim=1))

    model.eval()

    for cam_num in range(valloader.dataset.num_cameras):
        valloader.dataset.cam_num = cam_num
        loss, metric, relmetric, metricx, metricy, metricz, metricabs, metricangle = 0, 0, 0, 0, 0, 0, 0, 0
        metric_trajectory, metric_trajectory_before, metric_trajectory_after = 0, 0, 0
        metric_position = 0
        TPs, TNs, FPs, FNs = 0, 0, 0, 0
        number = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader)):
                r_img, table_img, mask, r_world, rotation, times, hits, __, __ = data
                r_img, table_img, mask, rotation = r_img.to(device), table_img.to(device), mask.to(device), rotation.to(device)
                r_world, times = r_world.to(device), times.to(device)
                B, T, D = r_img.shape
                pred_rotation, pred_position = model(r_img, table_img, mask, times)

                # All metrics are calculated in the ball's coordinate system -> transform (predicted and) gt rotations accordingly
                rotation = transform_rotationaxes(rotation, r_world)
                if config.transform_mode == 'global':
                    pred_rotation = transform_rotationaxes(pred_rotation, r_world)

                loss = loss_fn(pred_rotation, rotation)
                metric += metric_fn(pred_rotation, rotation)
                metricx += metricx_fn(pred_rotation, rotation)
                metricy += metricy_fn(pred_rotation, rotation)
                metricz += metricz_fn(pred_rotation, rotation)
                metricabs += metricabs_fn(pred_rotation, rotation)
                metricangle += metricangle_fn(pred_rotation, rotation)
                metric_position += metric_pos_fn(pred_position, r_world, mask)
                tmp = binary_metrics(pred_rotation, rotation)
                TPs, TNs, FPs, FNs = TPs + tmp[0], TNs + tmp[1], FPs + tmp[2], FNs + tmp[3]
                number += B
            loss /= number
            metric /= number
            metricx /= number
            metricy /= number
            metricz /= number
            metricabs /= number
            metricangle /= number
            metric_trajectory /= number
            metric_trajectory_before /= number
            metric_trajectory_after /= number
            metric_position /= number
            accuracy = (TPs + TNs) / (TPs + TNs + FPs + FNs)

        writer.add_scalar(f'val{cam_num}/loss', loss.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric', metric.item(), epoch)
        # writer.add_scalar(f'val{cam_num}/metric x', metricx.item(), epoch)
        # writer.add_scalar(f'val{cam_num}/metric y', metricy.item(), epoch)
        # writer.add_scalar(f'val{cam_num}/metric z', metricz.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric abs', metricabs.item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric angle', metricangle.item(), epoch)
        # Calculating the trajectory metric takes a lot of time...
        writer.add_scalar(f'val{cam_num}/accuracy x', accuracy[0].item(), epoch)
        writer.add_scalar(f'val{cam_num}/accuracy y', accuracy[1].item(), epoch)
        writer.add_scalar(f'val{cam_num}/accuracy z', accuracy[2].item(), epoch)
        writer.add_scalar(f'val{cam_num}/metric position', metric_position.item(), epoch)
        # Writing the confusion matrix only every tenth time to save space
        if epoch % 10 == 0 or epoch == -1:
            TPs, TNs, FPs, FNs = TPs.cpu().numpy(), TNs.cpu().numpy(), FPs.cpu().numpy(), FNs.cpu().numpy()
            writer.add_image(f'val{cam_num}/confusion matrix x', create_confusion_matrix(TPs[0], TNs[0], FPs[0], FNs[0]), epoch, dataformats='HWC')
            writer.add_image(f'val{cam_num}/confusion matrix y', create_confusion_matrix(TPs[1], TNs[1], FPs[1], FNs[1]), epoch, dataformats='HWC')
            writer.add_image(f'val{cam_num}/confusion matrix z', create_confusion_matrix(TPs[2], TNs[2], FPs[2], FNs[2]), epoch, dataformats='HWC')
        if cam_num == 0:
            # Add predicted and gt rotations as text to tensorboard
            writer.add_text(f'Predicted Rotations', str(np.round(pred_rotation[:4].cpu().numpy(), 1)), epoch)
            writer.add_text(f'GT Rotations', str(np.round(rotation[:4].cpu().numpy(), 1)), epoch)
            #Add hparams to tensorboard
            writer.add_hparams2(config.get_hparams(), {'metric': metric.item()})

    valloader.dataset.cam_num = 0
    model.train()

    return metric.item()


def val_real(model, valloader, writer, epoch, device, config):
    def metric_pos2D_fn(pred_2D, gt_2D, mask):
        B, T, __ = pred_2D.shape
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    pred_2D[b, t], gt_2D[b, t] = np.array([0, 0]), np.array([0, 0]) # set to same value such that difference is 0
        return np.sum(np.sum(np.sqrt(np.sum((pred_2D - gt_2D) ** 2, axis=-1)) * mask, axis=1) / np.sum(mask, axis=1))

    denorm = UnNormalizeImgCoords()
    model.eval()

    metric_2D = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    scores = []  # needed for ROC-AUC and number of missortings
    labels = []  # needed for ROC-AUC and number of missortings
    number = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            r_img, table_img, mask, times, hits, Mint, Mext, spin_class = data
            r_img, table_img, mask, times = r_img.to(device), table_img.to(device), mask.to(device), times.to(device)
            B, T, D = r_img.shape
            pred_rotation, pred_position = model(r_img, table_img, mask, times)
            # If network output is in ball's coordinate system, transform the ground truth to the local coordinate system
            if config.transform_mode == 'global':
                pred_rotation = transform_rotationaxes(pred_rotation, pred_position)

            # denormalization of ground truth image coordinates to calculate the 2D metric
            data_gt = denorm({'r_img': r_img.cpu().numpy(), 'table_img': table_img.cpu().numpy()})
            r_img, table_img = data_gt['r_img'], data_gt['table_img']

            # calculate metrics
            pred_pos_2D = cam2img(world2cam(pred_position, Mext.to(device)), Mint.to(device))
            m = metric_pos2D_fn(pred_pos_2D.cpu().numpy(), r_img, mask.cpu().numpy())
            metric_2D += m

            # binary metrics: Front- vs Backspin ; ROC-AUC ; Number of missortings
            for b in range(B):
                # binary metrics
                if spin_class[b] == TOPSPIN_CLASS:  # Frontspin
                    if pred_rotation[b, 1] > 0:
                        TP += 1
                    else:
                        FN += 1
                elif spin_class[b] == BACKSPIN_CLASS:  # Backspin
                    if pred_rotation[b, 1] < 0:
                        TN += 1
                    else:
                        FP += 1
                # ROC-AUC and missortings
                if spin_class[b] in [BACKSPIN_CLASS, TOPSPIN_CLASS]:  # only consider if spin class is annotated
                    scores.append(pred_rotation[b, 1].item())
                    labels.append(1 if spin_class[b] == 1 else 0)  # The methods use Frontspin=1 and Backspin=0


            number += B

        metric_2D /= number
        normed_metric_2D = metric_2D / (WIDTH ** 2 + HEIGHT ** 2) ** 0.5  # normalize by image size
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_plus = 2 * TP / (2 * TP + FP + FN)
        f1_minus = 2 * TN / (2 * TN + FN + FP)
        macro_f1 = (f1_plus + f1_minus) / 2
        roc_auc = roc_auc_score(labels, scores)

    writer.add_scalar('val real/metric 2D', metric_2D, epoch)
    writer.add_scalar('val real/metric 2D normed', normed_metric_2D, epoch)
    writer.add_scalar('val real/accuracy', accuracy, epoch)
    writer.add_scalar('val real/macro f1', macro_f1, epoch)
    writer.add_scalar('val real/ROC AUC', roc_auc, epoch)

    return normed_metric_2D, macro_f1






def main():
    global debug
    debug = args.debug
    config = TrainConfig(args.lr, args.model_name, args.model_size, debug, args.folder, args.exp_id)
    config.blur_strength = args.blur_strength
    config.stop_prob = args.stop_prob
    config.randomize_std = args.randomize_std
    config.randdet_prob = args.randdet_prob
    config.randmiss_prob = args.randmiss_prob
    config.tablemiss_prob = args.tablemiss_prob
    config.tabletoken_mode = args.token_mode
    config.transform_mode = args.transform_mode
    config.time_rotation = args.time_rotation
    run(config)


if __name__ == '__main__':
    main()