"""
Here an outer autoencoder is defined, encoding each of the occlusion rays, regarded as one-dimensional functions.
The result is dense latent codes (e.g. on the cylinder / sphere), which in turn will be encoded as a whole by the inner autoencoder.
"""

from torch import nn
import torch.nn.functional.relu as relu

class OccRayEncoder(nn.Module):
    def __init__(
        self,
        cnn_channel_list = [1, 1024],
        ksize_list = [45],
        stride_list = [1],
        fc_channel_list = [1024, 1024, 1024, 1024, 1024, 16],
    ):
        super().__init__()
        n_conv_layers = len(cnn_channel_list) - 1
        # n_fc_layers = len(fc_channel_list) - 1
        assert len(ksize_list) == n_conv_layers
        assert len(stride_list) == n_conv_layers
        assert fc_channel_list[0] == cnn_channel_list[-1], 'The number of output channels of the CNN part needs to equal the number of input channels for the FC head.'

        # CNN
        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, stride in zip(cnn_channel_list[:-1], cnn_channel_list[1:], stride_list):
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=0))

        # FC head
        self.fc_layers = nn.ModuleList()
        for in_ch, out_ch in zip(fc_channel_list[:-1], fc_channel_list[1:]):
            self.fc_layers.append(nn.Linear(in_ch, out_ch))

    def forward(self, occ_ray_rasterized, grid):
        bs = occ_ray_rasterized.shape[0]
        x = torch.cat((occ_ray_rasterized, grid), dim=1)

        # CNN
        for conv in self.conv_layers:
            x = conv(x)
            x = relu(x)
        assert x.shape[2] == 1, 'The CNN part has not managed to reduce the spatial dimension to 1 pixel, as was expected.'

        # FC head
        x = x.squeeze(2) # Squeeze spatial dimension
        for j, layer in enumerate(self.fc_layers):
            x = layer(x)
            if j + 1 < len(self.fc_layers):
                x = relu(x)

        return x


class OccRayDecoder(nn.Module):
    def __init__(
        self,
        fc_channel_list = [17, 1024, 1024, 1024, 1024, 1],
    ):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        for in_ch, out_ch in zip(fc_channel_list[:-1], fc_channel_list[1:]):
            self.fc_layers.append(nn.Linear(in_ch, out_ch))

    def forward(self, z, radial_samples):
        bs = z.shape[0]
        x = torch.cat((z, radial_samples), dim=1)

        for j, layer in enumerate(self.fc_layers):
            x = layer(x)
            if j + 1 < len(self.fc_layers):
                x = relu(x)

        return x
