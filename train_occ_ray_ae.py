import os
import shutil
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.config.config import config
from lib.logging.logging import log
from lib.logging.tb import get_tb_writer, initialize_tensorboard
from lib.datasets.occ_ray_dataset import OccRayDataset
from lib.models.occ_ray_ae import OccRayEncoder, OccRayDecoder


BS = 16
LR = 1e-4
N_EPOCHS = 10000
TRAINSET_LEN = 1024#*16
RAY_RANGE = 1
N_BATCHES_LOG_INTERVAL = TRAINSET_LEN//BS
OCC_RAY_RESOLUTION = 45
N_OCC_FCN_SAMPLES = 16
OCC_RAY_LATENT_DIM = 16

def main():
    if os.path.exists(config.exp_path):
        shutil.rmtree(config.exp_path)
    initialize_tensorboard()
    tb_writer = get_tb_writer()

    train_dataset = OccRayDataset(
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

    occ_ray_encoder = OccRayEncoder(
        cnn_channel_list = [2, 1024],
        ksize_list = [45],
        stride_list = [1],
        fc_channel_list = [1024, 1024, 1024, 1024, 1024, OCC_RAY_LATENT_DIM],
        # cnn_channel_list = [1, 16, 32, 64, 128, 256, 512, 1024],
        # ksize_list = [3, 3, 3, 3, 3, 3, 3],
        # stride_list = [1, 2, 1, 2, 1, 2, 1],
        # fc_channel_list = [1024, 1024, 1024, OCC_RAY_LATENT_DIM],
    ).cuda()
    occ_ray_decoder = OccRayDecoder(
        fc_channel_list = [OCC_RAY_LATENT_DIM+1, 1024, 1024, 1024, 1024, 1],
    ).cuda()

    optimizer = torch.optim.Adam([
        {'name': 'occ_ray_encoder', 'params': occ_ray_encoder.parameters()},
        {'name': 'occ_ray_decoder', 'params': occ_ray_decoder.parameters()},
    ], lr = LR)

    global_batch_cnt = 0
    for epoch in range(N_EPOCHS):
        for batch in tqdm(train_dataloader, 'Epoch #{}/{}'.format(epoch+1, N_EPOCHS)):
            z = occ_ray_encoder(batch['occ_ray_rasterized'].reshape((BS, 1, OCC_RAY_RESOLUTION)).cuda(), batch['grid'].reshape((BS, 1, OCC_RAY_RESOLUTION)).cuda())
            occ_fcn_vals_pred = occ_ray_decoder(
                z.reshape((BS, 1, OCC_RAY_LATENT_DIM)).expand((-1, N_OCC_FCN_SAMPLES, -1)).reshape((BS*N_OCC_FCN_SAMPLES, OCC_RAY_LATENT_DIM)),
                batch['radial_samples'].reshape((BS*N_OCC_FCN_SAMPLES, 1)).cuda(),
            ).reshape((BS, N_OCC_FCN_SAMPLES))
            occ_fcn_vals_target = batch['occ_fcn_vals'].cuda()
            loss = nn.functional.mse_loss(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='mean') / RAY_RANGE**2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if global_batch_cnt % N_BATCHES_LOG_INTERVAL == 0:
                log.info('Train loss: {:.8f}'.format(loss.item()))
                tb_writer.add_scalar("loss/train", loss, global_batch_cnt)
                # tb_writer.flush()
            global_batch_cnt += 1

if __name__ == '__main__':
    main()
