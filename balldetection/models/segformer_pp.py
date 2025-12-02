import torch.nn as nn
import torch
import os

from paths import weights_path


class Segformer_pp(nn.Module):
    def __init__(self, in_frames=3, model_size='b5', pretraining=False, classify_invisible=False):
        super(Segformer_pp, self).__init__()
        assert model_size in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], f"Model size {model_size} not supported"
        self.model = torch.hub.load(
            'KieDani/SegformerPlusPlus',
            'segformer_plusplus',
            pretrained=True,  # Always load ImageNet pretraining
            backbone=model_size,
            tome_strategy='bsm_hq',
            out_channels=19,
        )
        self.pretraining = pretraining
        self.classify_invisible = classify_invisible

        # adjust the first layer to accept multiple frames
        first_conv = self.model.backbone.layers[0][0].projection
        kernel_size = first_conv.kernel_size
        stride = first_conv.stride
        padding = first_conv.padding
        in_channels = first_conv.in_channels
        out_channels = first_conv.out_channels

        # change the first layer to accept 3 frames
        self.model.backbone.layers[0][0].projection = nn.Conv2d(in_channels*in_frames, out_channels, kernel_size, stride, padding)

        # init weights as in_frames times the original weights
        with torch.no_grad():
            self.model.backbone.layers[0][0].projection.weight = torch.nn.Parameter(torch.cat([first_conv.weight for _ in range(in_frames)], dim=1) / in_frames)
            self.model.backbone.layers[0][0].projection.bias = torch.nn.Parameter(first_conv.bias)

        # adjust number of output channels in the last layer -> one output channel (as heatmap)
        last_conv = self.model.decode_head.conv_seg
        in_channels = last_conv.in_channels
        kernel_size = last_conv.kernel_size
        stride = last_conv.stride
        padding = last_conv.padding
        avg_weight = last_conv.weight.mean(dim=0).unsqueeze(0)
        avg_bias = last_conv.bias.mean().unsqueeze(0)
        self.model.decode_head.conv_seg = nn.Conv2d(in_channels, 1, kernel_size, stride, padding)
        with torch.no_grad():
            self.model.decode_head.conv_seg.weight = torch.nn.Parameter(avg_weight)
            self.model.decode_head.conv_seg.bias = torch.nn.Parameter(avg_bias)

        # classification output: ball visible or invisible
        if self.classify_invisible:
            self.visible_classification = torch.nn.Linear(in_features=self.model.decode_head.in_channels[-1], out_features=2)


        if self.pretraining:
            if model_size == "b2":
                file_path = os.path.join(weights_path, 'pretraining_blurball', 'segformerpp_b2', 'model.pt')
                self.load_state_dict(torch.load(file_path)['model_state_dict'])
                print(f"Loaded pretrained BlurBall model")
            elif model_size == 'b0':
                file_path = os.path.join(weights_path, 'pretraining_blurball', 'segformerpp_b0', 'model.pt')
                self.load_state_dict(torch.load(file_path)['model_state_dict'])
                print(f"Loaded pretrained SegFormer B0 model trained on BlurBall dataset")
            else:
                raise NotImplementedError(f"Pretrained weights for model size {model_size} not implemented")



    def forward(self, x):
        features = self.model.backbone(x)
        seg_out = self.model.decode_head(features)
        if self.classify_invisible:
            used_features = features[-1]
            _, c, h, w = used_features.shape
            pooled = torch.nn.functional.avg_pool2d(used_features, (h, w))
            pooled = torch.flatten(pooled, start_dim=1)
            class_out = self.visible_classification(pooled)
            return seg_out, class_out
        else:
            return seg_out, None