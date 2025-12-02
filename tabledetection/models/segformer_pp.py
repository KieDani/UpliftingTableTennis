import torch.nn as nn
import torch
import os
import einops as eo


class Segformer_pp(nn.Module):

    def __init__(self, model_size='b5', pretraining=False):
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

        self.number_output_channels = 13

        if self.pretraining:
            # load the pretrained model
            # self.model.load_state_dict(torch.load(file_path)['state_dict'])
            # print(f"Loaded pretrained city scapes model")
            raise NotImplementedError("Pretraining not yet supported for SegFormer++ in tabledetection")

        # adjust number of output channels in the last layer -> one output channel (as heatmap)
        last_conv = self.model.decode_head.conv_seg
        in_channels = last_conv.in_channels
        kernel_size = last_conv.kernel_size
        stride = last_conv.stride
        padding = last_conv.padding
        self.model.decode_head.conv_seg = nn.Conv2d(in_channels, self.number_output_channels, kernel_size, stride, padding)
        with torch.no_grad():
            avg_weight = last_conv.weight.mean(dim=0)
            avg_bias = last_conv.bias.mean()
            new_weight = eo.repeat(avg_weight, 'c ... -> n c ...', n=self.number_output_channels).clone()
            new_bias = eo.repeat(avg_bias, '  -> n', n=self.number_output_channels).clone()
            self.model.decode_head.conv_seg.weight = torch.nn.Parameter(new_weight)
            self.model.decode_head.conv_seg.bias = torch.nn.Parameter(new_bias)

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = Segformer_pp(model_size='b0', pretraining=False)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)  # Should be (1, 13, 128, 128) for b5 model

    # number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")