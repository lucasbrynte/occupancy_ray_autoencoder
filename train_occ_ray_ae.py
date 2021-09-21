import os
import argparse
import torch
from torch.utils.data import DataLoader
from lib.datasets.occl_ray_dataset import OcclRayDataset
from lib.models.occl_ray_ae import OcclRayEncoder, OcclRayDecoder
import torch.nn.functional.mse_loss as mse_loss

BS = 16
LR = 1e-3
RAY_RANGE = 1
TRAINSET_LEN = 1024
OCC_RAY_RESOLUTION = 45
N_OCC_FCN_SAMPLES = 4
OCC_RAY_LATENT_DIM = 16

def main(args):
    exp_path = os.path.join(args.exp_root, args.exp_name)
    tb_path = os.path.join(exp_path, 'tb')

    train_dataset = OcclRayDataset(
        range = RAY_RANGE,
        resolution = OCC_RAY_RESOLUTION,
        len = TRAINSET_LEN,
        n_occ_fcn_samples = N_OCC_FCN_SAMPLES,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BS,
        shuffle = True,
        pin_memory = True,
        drop_last = False,
    )

    occl_ray_encoder = OccRayEncoder(
        cnn_channel_list = [1, 1024],
        ksize_list = [45],
        stride_list = [1],
        fc_channel_list = [1024, 1024, 1024, 1024, 1024, OCC_RAY_LATENT_DIM],
    ).cuda()
    occl_ray_decoder = OccRayDecoder(
        fc_channel_list = [OCC_RAY_LATENT_DIM+1, 1024, 1024, 1024, 1024, 1],
    ).cuda()

    optimizer = torch.optim.Adam(
        {'occl_ray_encoder': occl_ray_encoder.parameters(), 'occl_ray_decoder': occl_ray_decoder.parameters()},
        lr = LR,
    )

    for epoch in range(N_EPOCHS):
        for batch in train_dataloader:
            z = occl_ray_encoder(batch['occ_ray_rasterized'].reshape((BS, 1, OCC_RAY_RESOLUTION)).cuda(), batch['grid'].cuda())
            occ_fcn_vals_pred = occl_ray_decoder(
                z.reshape((BS, 1, OCC_RAY_LATENT_DIM)).expand((1, N_OCC_FCN_SAMPLES, 1)).reshape((BS*N_OCC_FCN_SAMPLES, OCC_RAY_LATENT_DIM)),
                batch['radial_samples'].reshape((BS*N_OCC_FCN_SAMPLES, 1)).cuda(),
            ).reshape((BS, N_OCC_FCN_SAMPLES, 1))
            occ_fcn_vals_target = batch['occ_fcn_vals'].cuda()
            loss = mse_loss(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='mean') / RAY_RANGE**2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--exp-root', help='Root path in which to experiments.', default='out')
    parser.add_argument('--exp-name', help='Experiment name.', required=True)
    args = parser.parse_args()

    main(args)
