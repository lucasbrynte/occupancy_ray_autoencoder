"""
Here an outer autoencoder is defined, encoding each of the occlusion rays, regarded as one-dimensional functions.
The result is dense latent codes (e.g. on the cylinder / sphere), which in turn will be encoded as a whole by the inner autoencoder.
"""

import torch
from torch import nn

class OccRayEncoder(nn.Module):
    def __init__(
        self,
        cnn_channel_list = [2, 1024],
        ksize_list = [45],
        stride_list = [1],
        fc_channel_list = [1024, 1024, 1024, 1024, 1024, 16],
        batch_norm = False,
    ):
        super().__init__()
        n_conv_layers = len(cnn_channel_list) - 1
        # n_fc_layers = len(fc_channel_list) - 1
        assert len(ksize_list) == n_conv_layers
        assert len(stride_list) == n_conv_layers
        assert fc_channel_list[0] == cnn_channel_list[-1], 'The number of output channels of the CNN part needs to equal the number of input channels for the FC head.'

        self.batch_norm = batch_norm

        # CNN
        self.conv_layers = nn.ModuleList()
        if self.batch_norm:
            self.conv_bn_layers = nn.ModuleList()
        for in_ch, out_ch, ksize, stride in zip(cnn_channel_list[:-1], cnn_channel_list[1:], ksize_list, stride_list):
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, ksize, stride=stride, padding=0))
            if self.batch_norm:
                self.conv_bn_layers.append(nn.BatchNorm1d(out_ch))

        # FC head
        self.fc_layers = nn.ModuleList()
        if self.batch_norm:
            self.fc_bn_layers = nn.ModuleList()
        for j, (in_ch, out_ch) in enumerate(zip(fc_channel_list[:-1], fc_channel_list[1:])):
            self.fc_layers.append(nn.Linear(in_ch, out_ch))
            if self.batch_norm and j + 1 < len(fc_channel_list[:-1]):
                self.fc_bn_layers.append(nn.BatchNorm1d(out_ch))

    def forward(self, occ_ray_rasterized, grid):
        bs = occ_ray_rasterized.shape[0]
        x = torch.cat((occ_ray_rasterized, grid), dim=1)

        # CNN
        for j, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = nn.functional.relu(x)
            x = self.conv_bn_layers[j](x)
            # BN after activation:
            # https://github.com/cvjena/cnn-models/issues/3#issuecomment-266782416
            # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        assert x.shape[2] == 1, 'The CNN part has not managed to reduce the spatial dimension to 1 pixel, as was expected. Shape: {}'.format(x.shape)

        # FC head
        x = x.squeeze(2) # Squeeze spatial dimension
        for j, layer in enumerate(self.fc_layers):
            x = layer(x)
            if j + 1 < len(self.fc_layers):
                x = nn.functional.relu(x)
                x = self.fc_bn_layers[j](x)

        return x


class OccRayDecoder(nn.Module):
    def __init__(
        self,
        fc_channel_list = [17, 1024, 1024, 1024, 1024, 1],
        batch_norm = False,
    ):
        super().__init__()
        self.batch_norm = batch_norm

        self.fc_layers = nn.ModuleList()
        if self.batch_norm:
            self.fc_bn_layers = nn.ModuleList()
        for j, (in_ch, out_ch) in enumerate(zip(fc_channel_list[:-1], fc_channel_list[1:])):
            self.fc_layers.append(nn.Linear(in_ch, out_ch))
            if self.batch_norm and j + 1 < len(fc_channel_list[:-1]):
                self.fc_bn_layers.append(nn.BatchNorm1d(out_ch))

    def forward(self, z, radial_samples):
        bs = z.shape[0]
        x = torch.cat((z, radial_samples), dim=1)

        for j, layer in enumerate(self.fc_layers):
            x = layer(x)
            if j + 1 < len(self.fc_layers):
                x = nn.functional.relu(x)
                x = self.fc_bn_layers[j](x)

        return x
