import os
import json
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.config.config import config
from lib.logging.logging import log
from lib.logging.signal_manager import SignalManager
from lib.logging.checkpoint import save_checkpoint, load_checkpoint, serialize_checkpoint_metadata
from lib.logging.version_dump import version_dump
from lib.datasets.occ_ray_dataset import OccRayDataset
from lib.models.occ_ray_ae import OccRayEncoder, OccRayDecoder



def main():
    with open(os.path.join(config.EXP_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    version_dump()

    signal_manager = SignalManager()

    train_dataset = OccRayDataset(
        config.OCC_RAY_AE.TRAIN.DATA.N_SAMPLES,
        config.OCC_RAY_AE.TRAIN.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.OCC_RAY_AE.BS,
        shuffle = True,
        pin_memory = True,
        drop_last = False,
    )
    val_dataset = OccRayDataset(
        config.OCC_RAY_AE.VAL.DATA.N_SAMPLES,
        config.OCC_RAY_AE.VAL.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = config.OCC_RAY_AE.BS,
        shuffle = True,
        pin_memory = True,
        drop_last = False,
    )

    occ_ray_encoder = OccRayEncoder(
        cnn_channel_list = config.OCC_RAY_AE.ARCH.ENCODER.CNN_CHANNEL_LIST,
        ksize_list = config.OCC_RAY_AE.ARCH.ENCODER.KSIZE_LIST,
        stride_list = config.OCC_RAY_AE.ARCH.ENCODER.STRIDE_LIST,
        fc_channel_list = list(config.OCC_RAY_AE.ARCH.ENCODER.FC_CHANNEL_LIST) + [config.OCC_RAY_AE.OCC_RAY_LATENT_DIM],
        # cnn_channel_list = [2, 16, 32, 64, 128, 256, 512, 1024],
        # ksize_list = [3, 3, 3, 3, 3, 3, 3],
        # stride_list = [1, 2, 1, 2, 1, 2, 1],
        # fc_channel_list = [1024, 1024, 1024, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM],
    ).cuda()
    occ_ray_decoder = OccRayDecoder(
        fc_channel_list = [config.OCC_RAY_AE.OCC_RAY_LATENT_DIM+1] + list(config.OCC_RAY_AE.ARCH.DECODER.FC_CHANNEL_LIST) + [1],
    ).cuda()

    optimizer = torch.optim.Adam([
        {'name': 'occ_ray_encoder', 'params': occ_ray_encoder.parameters()},
        {'name': 'occ_ray_decoder', 'params': occ_ray_decoder.parameters()},
    ], lr = config.OCC_RAY_AE.LR)

    for epoch in range(config.OCC_RAY_AE.N_EPOCHS):
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, '[TRAIN] Epoch #{}/{}'.format(epoch+1, config.OCC_RAY_AE.N_EPOCHS))):
            is_last_batch = not (batch_idx+1) < len(train_dataloader)
            batch_data = preprocess_batch(batch_data)
            batch_forward_out = batch_forward(
                occ_ray_encoder,
                occ_ray_decoder,
                batch_data,
            )
            loss = calc_loss(batch_data, batch_forward_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            signal_manager.record_train_batch(
                {
                    'n_surface_occ_fcn_samples': batch_data['n_surface_occ_fcn_samples'].numpy(),
                    'occ_ray_rasterized': batch_data['occ_ray_rasterized'].numpy(),
                    'anywhere_pts': batch_data['anywhere_pts'].numpy(),
                    # 'anywhere_pt_mask': anywhere_pt_mask.numpy(),
                    'surface_pts': batch_data['surface_pts'].numpy(),
                    'surface_pt_mask': batch_data['surface_pt_mask'].numpy(),
                    'anywhere_occ_fcn_vals_pred': batch_forward_out['anywhere_occ_fcn_vals_pred'].detach().cpu().numpy(),
                    'surface_occ_fcn_vals_pred': batch_forward_out['surface_occ_fcn_vals_pred'].detach().cpu().numpy(),
                    'anywhere_occ_fcn_vals_target': batch_data['anywhere_occ_fcn_vals'].numpy(),
                    'surface_occ_fcn_vals_target': batch_data['surface_occ_fcn_vals'].numpy(),
                    'loss': loss.detach().cpu().numpy(),
                },
                log_signals = is_last_batch or (config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL is not None and (batch_idx+1) % config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL == 0),
                log_signals_tb = is_last_batch or (config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL is not None and (batch_idx+1) % config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL == 0),
                visualize_pred = is_last_batch or (config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL is not None and (batch_idx+1) % config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL == 0),
            )
            if not is_last_batch and (config.OCC_RAY_AE.N_BATCHES_VAL_INTERVAL is not None and (batch_idx+1) % config.OCC_RAY_AE.N_BATCHES_VAL_INTERVAL == 0):
                with torch.no_grad():
                    validate(
                        val_dataloader,
                        occ_ray_encoder,
                        occ_ray_decoder,
                        signal_manager,
                    )
        with torch.no_grad():
            validate(
                val_dataloader,
                occ_ray_encoder,
                occ_ray_decoder,
                signal_manager,
            )
        if epoch == 0 or (epoch+1) == config.OCC_RAY_AE.N_EPOCHS or (epoch+1) % config.OCC_RAY_AE.N_EPOCHS_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                os.path.join(config.CHECKPOINT_DIR, serialize_checkpoint_metadata({'epoch': '{:08d}'.format(epoch+1)})),
                epoch+1,
                (epoch+1) * len(train_dataset),
                occ_ray_encoder,
                occ_ray_decoder,
                optimizer,
            )

def validate(
    val_dataloader,
    occ_ray_encoder,
    occ_ray_decoder,
    signal_manager,
):
    for batch_idx, batch_data in enumerate(tqdm(val_dataloader, '[VAL]')):
        batch_data = preprocess_batch(batch_data)
        batch_forward_out = batch_forward(
            occ_ray_encoder,
            occ_ray_decoder,
            batch_data,
        )
        loss = calc_loss(batch_data, batch_forward_out)
        signal_manager.record_val_batch(
            {
                'n_surface_occ_fcn_samples': batch_data['n_surface_occ_fcn_samples'].numpy(),
                'occ_ray_rasterized': batch_data['occ_ray_rasterized'].numpy(),
                'anywhere_pts': batch_data['anywhere_pts'].numpy(),
                # 'anywhere_pt_mask': anywhere_pt_mask.numpy(),
                'surface_pts': batch_data['surface_pts'].numpy(),
                'surface_pt_mask': batch_data['surface_pt_mask'].numpy(),
                'anywhere_occ_fcn_vals_pred': batch_forward_out['anywhere_occ_fcn_vals_pred'].detach().cpu().numpy(),
                'surface_occ_fcn_vals_pred': batch_forward_out['surface_occ_fcn_vals_pred'].detach().cpu().numpy(),
                'anywhere_occ_fcn_vals_target': batch_data['anywhere_occ_fcn_vals'].numpy(),
                'surface_occ_fcn_vals_target': batch_data['surface_occ_fcn_vals'].numpy(),
                'loss': loss.detach().cpu().numpy(),
            },
            visualize_pred = batch_idx == 0,
        )
    signal_manager.calc_avg_val_metrics(
        log_signals = True,
        log_signals_tb = True,
    )

def preprocess_batch(batch_data):
    assert torch.all(batch_data['anywhere_pt_weights'] > 0) # Simplifies things if only surface points are variable-length
    # batch_data['anywhere_pt_mask'] = batch_data['anywhere_pt_weights'] > 0
    batch_data['surface_pt_mask'] = batch_data['surface_pt_weights'] > 0
    return batch_data

def batch_forward(
    occ_ray_encoder,
    occ_ray_decoder,
    batch_data,
):
    z = occ_ray_encoder(batch_data['occ_ray_rasterized'].reshape((config.OCC_RAY_AE.BS, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)).cuda(), batch_data['grid'].reshape((config.OCC_RAY_AE.BS, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)).cuda())
    occ_fcn_vals_pred = occ_ray_decoder(
        z.reshape((config.OCC_RAY_AE.BS, 1, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)).expand((-1, config.OCC_RAY_AE.N_OCC_FCN_SAMPLES, -1)).reshape((config.OCC_RAY_AE.BS*config.OCC_RAY_AE.N_OCC_FCN_SAMPLES, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)),
        torch.cat([
            batch_data['anywhere_pts'],
            batch_data['surface_pts'],
        ], dim=1).reshape((config.OCC_RAY_AE.BS*config.OCC_RAY_AE.N_OCC_FCN_SAMPLES, 1)).cuda(),
    ).reshape((config.OCC_RAY_AE.BS, config.OCC_RAY_AE.N_OCC_FCN_SAMPLES))
    if config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'bce':
        occ_fcn_vals_pred = torch.sigmoid(occ_fcn_vals_pred)
    anywhere_occ_fcn_vals_pred = occ_fcn_vals_pred[:, :config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES]
    surface_occ_fcn_vals_pred = occ_fcn_vals_pred[:, -config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES:]
    return {
        'z': z,
        'anywhere_occ_fcn_vals_pred': anywhere_occ_fcn_vals_pred,
        'surface_occ_fcn_vals_pred': surface_occ_fcn_vals_pred,
    }

def calc_loss(batch_data, batch_forward_out):
    occ_fcn_vals_pred = torch.cat([
        batch_forward_out['anywhere_occ_fcn_vals_pred'],
        batch_forward_out['surface_occ_fcn_vals_pred'],
    ], dim=1).cuda()
    occ_fcn_vals_target = torch.cat([
        batch_data['anywhere_occ_fcn_vals'],
        batch_data['surface_occ_fcn_vals'],
    ], dim=1).cuda()
    point_weights = torch.cat([
        batch_data['anywhere_pt_weights'],
        batch_data['surface_pt_weights'],
    ], dim=1).cuda()
    if config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'mse':
        loss = nn.functional.mse_loss(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='none')
        loss = torch.mean((point_weights * loss)[point_weights > 0])
    elif config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'bce':
        loss = nn.functional.binary_cross_entropy(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='none')
        loss = torch.mean((point_weights * loss)[point_weights > 0])
    else:
        assert False
    return loss
