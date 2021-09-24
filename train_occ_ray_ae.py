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
from lib.visualization.visualization import visualize_train_batch



def main():
    if os.path.exists(config.EXP_PATH):
        shutil.rmtree(config.EXP_PATH)
    initialize_tensorboard()
    tb_writer = get_tb_writer()

    train_dataset = OccRayDataset(
        range = 1,
        resolution = config.OCC_RAY_AE.OCC_RAY_RESOLUTION,
        len = 1024,#*16
        n_occ_fcn_samples = config.OCC_RAY_AE.N_OCC_FCN_SAMPLES,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.OCC_RAY_AE.BS,
        shuffle = True,
        pin_memory = True,
        drop_last = False,
    )

    occ_ray_encoder = OccRayEncoder(
        cnn_channel_list = [2, 1024],
        ksize_list = [45],
        stride_list = [1],
        fc_channel_list = [1024, 1024, 1024, 1024, 1024, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM],
        # cnn_channel_list = [2, 16, 32, 64, 128, 256, 512, 1024],
        # ksize_list = [3, 3, 3, 3, 3, 3, 3],
        # stride_list = [1, 2, 1, 2, 1, 2, 1],
        # fc_channel_list = [1024, 1024, 1024, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM],
    ).cuda()
    occ_ray_decoder = OccRayDecoder(
        fc_channel_list = [config.OCC_RAY_AE.OCC_RAY_LATENT_DIM+1, 1024, 1024, 1024, 1024, 1],
    ).cuda()

    optimizer = torch.optim.Adam([
        {'name': 'occ_ray_encoder', 'params': occ_ray_encoder.parameters()},
        {'name': 'occ_ray_decoder', 'params': occ_ray_decoder.parameters()},
    ], lr = config.OCC_RAY_AE.LR)

    global_batch_cnt = 0
    for epoch in range(config.OCC_RAY_AE.N_EPOCHS):
        for batch_idx, batch in tqdm(enumerate(train_dataloader), 'Epoch #{}/{}'.format(epoch+1, config.OCC_RAY_AE.N_EPOCHS)):
            is_last_batch = not (batch_idx+1) < len(train_dataloader)
            z = occ_ray_encoder(batch['occ_ray_rasterized'].reshape((config.OCC_RAY_AE.BS, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)).cuda(), batch['grid'].reshape((config.OCC_RAY_AE.BS, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)).cuda())
            occ_fcn_vals_pred = occ_ray_decoder(
                z.reshape((config.OCC_RAY_AE.BS, 1, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)).expand((-1, config.OCC_RAY_AE.N_OCC_FCN_SAMPLES, -1)).reshape((config.OCC_RAY_AE.BS*config.OCC_RAY_AE.N_OCC_FCN_SAMPLES, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)),
                batch['radial_samples'].reshape((config.OCC_RAY_AE.BS*config.OCC_RAY_AE.N_OCC_FCN_SAMPLES, 1)).cuda(),
            ).reshape((config.OCC_RAY_AE.BS, config.OCC_RAY_AE.N_OCC_FCN_SAMPLES))
            occ_fcn_vals_target = batch['occ_fcn_vals'].cuda()
            if config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'mse':
                loss = nn.functional.mse_loss(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='mean')
            elif config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'bce':
                occ_fcn_vals_pred = torch.sigmoid(occ_fcn_vals_pred)
                loss = nn.functional.binary_cross_entropy(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='mean')
            else:
                assert False
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if is_last_batch or (config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL is not None and global_batch_cnt % config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL == 0):
                log.info('Train loss: {:.8f}'.format(loss.item()))
                tb_writer.add_scalar("loss/train", loss, global_batch_cnt)
                # tb_writer.flush()
            if is_last_batch or (config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL is not None and global_batch_cnt % config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL == 0):
                visualize_train_batch('figures/prediction/train', global_batch_cnt, batch['occ_ray_rasterized'][0].detach().cpu().numpy(), batch['radial_samples'][0].detach().cpu().numpy(), occ_fcn_vals_pred[0].detach().cpu().numpy(), occ_fcn_vals_target[0].detach().cpu().numpy())
            global_batch_cnt += 1

if __name__ == '__main__':
    main()
