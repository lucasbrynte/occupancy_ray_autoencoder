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
from lib.run.run import preprocess_batch, occ_ray_ae_forward
from lib.loss.loss import calc_loss_anywhere_surface, calc_pairwise_sinkhorn_regularization_loss



def main():
    with open(os.path.join(config.EXP_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    version_dump()

    signal_manager = SignalManager()

    train_dataset = OccRayDataset(
        config.OCC_RAY_AE.TRAIN.DATA.N_SAMPLES,
        config.OCC_RAY_AE.TRAIN.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS,
        random_seed = 23443419977523935600099423643672937524,
        reset_seed_on_epoch_start = False,
        anywhere_samples = True,
        surface_samples = True,
        dense_samples = False,
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
        random_seed = 165569474884930433373555743278446600479,
        reset_seed_on_epoch_start = True, # requires shuffle = False
        anywhere_samples = True,
        surface_samples = True,
        dense_samples = False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = config.OCC_RAY_AE.BS,
        shuffle = False,
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
        norm_config = config.OCC_RAY_AE.ARCH.NORMALIZATION,
    ).cuda()
    occ_ray_decoder = OccRayDecoder(
        fc_channel_list = [config.OCC_RAY_AE.OCC_RAY_LATENT_DIM+1] + list(config.OCC_RAY_AE.ARCH.DECODER.FC_CHANNEL_LIST) + [1],
        norm_config = config.OCC_RAY_AE.ARCH.NORMALIZATION,
    ).cuda()

    optimizer = torch.optim.Adam([
        {'name': 'occ_ray_encoder', 'params': occ_ray_encoder.parameters()},
        {'name': 'occ_ray_decoder', 'params': occ_ray_decoder.parameters()},
    ], lr = config.OCC_RAY_AE.LR)

    for epoch in range(config.OCC_RAY_AE.N_EPOCHS):
        occ_ray_encoder.train()
        occ_ray_decoder.train()
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, '[TRAIN] Epoch #{}/{}'.format(epoch+1, config.OCC_RAY_AE.N_EPOCHS))):
            is_last_batch = not (batch_idx+1) < len(train_dataloader)
            batch_data = preprocess_batch(batch_data)
            ae_out = occ_ray_ae_forward(
                occ_ray_encoder,
                occ_ray_decoder,
                batch_data,
            )
            main_loss = calc_loss_anywhere_surface(batch_data, ae_out)
            if config.OCC_RAY_AE.SINKHORN_REG.enabled:
                sinkhorn_reg_loss = calc_pairwise_sinkhorn_regularization_loss(batch_data, ae_out)
                loss = main_loss + config.OCC_RAY_AE.SINKHORN_REG.loss_coefficient * sinkhorn_reg_loss
            else:
                sinkhorn_reg_loss = torch.zeros_like(main_loss)
                loss = main_loss
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
                    'anywhere_occ_fcn_vals_pred': ae_out['anywhere_occ_fcn_vals_pred'].detach().cpu().numpy(),
                    'surface_occ_fcn_vals_pred': ae_out['surface_occ_fcn_vals_pred'].detach().cpu().numpy(),
                    'anywhere_occ_fcn_vals_target': batch_data['anywhere_occ_fcn_vals'].numpy(),
                    'surface_occ_fcn_vals_target': batch_data['surface_occ_fcn_vals'].numpy(),
                    'main_loss': main_loss.detach().cpu().numpy(),
                    'sinkhorn_reg_loss': sinkhorn_reg_loss.detach().cpu().numpy(),
                    'loss': loss.detach().cpu().numpy(),
                },
                log_signals = is_last_batch or (config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL is not None and (batch_idx+1) % config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL == 0),
                log_signals_tb = is_last_batch or (config.OCC_RAY_AE.N_BATCHES_TB_INTERVAL is not None and (batch_idx+1) % config.OCC_RAY_AE.N_BATCHES_TB_INTERVAL == 0),
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
        occ_ray_encoder.eval()
        occ_ray_decoder.eval()
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
        ae_out = occ_ray_ae_forward(
            occ_ray_encoder,
            occ_ray_decoder,
            batch_data,
        )
        main_loss = calc_loss_anywhere_surface(batch_data, ae_out)
        if config.OCC_RAY_AE.SINKHORN_REG.enabled:
            sinkhorn_reg_loss = calc_pairwise_sinkhorn_regularization_loss(batch_data, ae_out)
            loss = main_loss + config.OCC_RAY_AE.SINKHORN_REG.loss_coefficient * sinkhorn_reg_loss
        else:
            sinkhorn_reg_loss = torch.zeros_like(main_loss)
            loss = main_loss
        signal_manager.record_val_batch(
            {
                'n_surface_occ_fcn_samples': batch_data['n_surface_occ_fcn_samples'].numpy(),
                'occ_ray_rasterized': batch_data['occ_ray_rasterized'].numpy(),
                'anywhere_pts': batch_data['anywhere_pts'].numpy(),
                # 'anywhere_pt_mask': anywhere_pt_mask.numpy(),
                'surface_pts': batch_data['surface_pts'].numpy(),
                'surface_pt_mask': batch_data['surface_pt_mask'].numpy(),
                'anywhere_occ_fcn_vals_pred': ae_out['anywhere_occ_fcn_vals_pred'].detach().cpu().numpy(),
                'surface_occ_fcn_vals_pred': ae_out['surface_occ_fcn_vals_pred'].detach().cpu().numpy(),
                'anywhere_occ_fcn_vals_target': batch_data['anywhere_occ_fcn_vals'].numpy(),
                'surface_occ_fcn_vals_target': batch_data['surface_occ_fcn_vals'].numpy(),
                'main_loss': main_loss.detach().cpu().numpy(),
                'sinkhorn_reg_loss': sinkhorn_reg_loss.detach().cpu().numpy(),
                'loss': loss.detach().cpu().numpy(),
            },
            visualize_pred = batch_idx == 0,
        )
    signal_manager.calc_avg_val_metrics(
        log_signals = True,
        log_signals_tb = True,
    )
