import torch
import torch.nn as nn
import numpy as np
import os

from vit_pose import ViTPoseModel
import paths

def get_config(size='small'):
    assert size in ['small', 'base'], f"Model size {size} not supported"
    channel_cfg = dict(
        num_output_channels=1,
        )

    model = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='ViT',
            img_size=(256, 192),  # Is adjusted later
            patch_size=16,
            embed_dim=384 if size == 'small' else 768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels= 384 if size == 'small' else 768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=channel_cfg['num_output_channels'],),
        train_cfg=dict(),
        )

    return model



class VitPose(nn.Module):
    def __init__(self, in_frames=3, model_size='base', pretraining=False, resolution=((512, 384)), classify_invisible=False):
        super(VitPose, self).__init__()
        assert model_size in ['small', 'base', 'large'], f"Model size {model_size} not supported"
        model_config = get_config()
        model_config['backbone']['img_size'] = resolution
        self.model = ViTPoseModel(model_config)
        self.pretraining = pretraining
        self.classify_invisible = classify_invisible

        # MAE initialization
        file_path = os.path.join(paths.weights_path, 'initialization', 'vitpose', f'mae_pretrain_vit_{model_size}.pth')
        sd = torch.load(file_path)['model']
        number_successfully_loaded = 0
        number_total = 0
        for k, v in self.model.backbone.state_dict().items():
            if k in sd:
                if v.shape == sd[k].shape:
                    number_successfully_loaded += 1
                else:
                    del sd[k]
            number_total += 1
        print(f"Loaded {number_successfully_loaded} out of {number_total} MAE weights successfully")
        self.model.backbone.load_state_dict(sd, strict=False)

        # adjust the first layer to accept multiple frames
        first_conv = self.model.backbone.patch_embed.proj
        new_conv = nn.Conv2d(in_channels=in_frames*3, out_channels=first_conv.out_channels, kernel_size=first_conv.kernel_size, stride=first_conv.stride, padding=first_conv.padding)
        # init weights as in_frames times the original weights
        with torch.no_grad():
            new_conv.weight = torch.nn.Parameter(torch.cat([first_conv.weight for _ in range(in_frames)], dim=1) / in_frames)
            new_conv.bias = torch.nn.Parameter(first_conv.bias)
        self.model.backbone.patch_embed.proj = new_conv

        # classification output: ball visible or invisible
        if self.classify_invisible:
            self.visible_classification = torch.nn.Linear(in_features=self.model.decode_head.in_channels[-1], out_features=2)

        if self.pretraining:
            if model_size == 'small':
                file_path = os.path.join(paths.weights_path, 'pretraining_blurball', 'vitpose', 'model.pt')
                self.load_state_dict(torch.load(file_path)['model_state_dict'])
                print(f"Loaded pretrained VitPose Small model trained on BlurBall dataset")
            else:
                raise NotImplementedError("Pretraining only implemented for small model")

    def forward(self, x):
        features = self.model.forward_features(x)
        seg_out = self.model.keypoint_head(features)
        if self.classify_invisible:
            used_features = features
            _, c, h, w = used_features.shape
            pooled = torch.nn.functional.avg_pool2d(used_features, (h, w))
            pooled = torch.flatten(pooled, start_dim=1)
            class_out = self.visible_classification(pooled)
            return seg_out, class_out
        else:
            return seg_out, None


if __name__ == "__main__":
    resolution = (512, 384)
    model = VitPose(model_size='small', pretraining=False, resolution=resolution)
    x = torch.randn(1, 9, resolution[0], resolution[1])
    y = model(x)
    print(y.shape)  # Should be (1, 17, 64, 48) for base model

    # number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")



