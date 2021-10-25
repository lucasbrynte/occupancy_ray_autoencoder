import os
import json
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.config.config import config
from lib.logging.logging import log
from lib.logging.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint_file
from lib.logging.version_dump import version_dump
from lib.datasets.occ_ray_dataset import OccRayDataset
from lib.datasets.random_latent_code_dataset import RandomLatentCodeDataset
from lib.models.occ_ray_ae import OccRayEncoder, OccRayDecoder
from lib.run.run import preprocess_batch, occ_ray_encoder_forward, occ_ray_decoder_forward, interpolate_latent_codes
from lib.loss.loss import calc_loss


def main():
    with open(os.path.join(config.EXP_DIR, 'config.json'), 'w') as f:
        json.dump(config, f)
    version_dump()

    test_dataset_encoded_interpolation_samples = OccRayDataset(
        2 * config.OCC_RAY_AE.TEST.DATA.N_ENCODED_INTERPOLATION_PAIRS,
        config.OCC_RAY_AE.TEST.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS,
        random_seed = 244346077244724968364822639758077361206,
        reset_seed_on_epoch_start = True, # requires shuffle = False
        anywhere_samples = False,
        surface_samples = False,
        dense_samples = True,
    )
    test_dataloader_encoded_interpolation_samples = DataLoader(
        test_dataset_encoded_interpolation_samples,
        batch_size = config.OCC_RAY_AE.BS,
        shuffle = False,
        pin_memory = True,
        drop_last = False,
    )

    test_dataset_random_interpolation_samples = RandomLatentCodeDataset(
        2 * config.OCC_RAY_AE.TEST.DATA.N_RANDOM_INTERPOLATION_PAIRS,
        random_seed = 244346077244724968364822639758077361206,
        reset_seed_on_epoch_start = True, # requires shuffle = False
    )
    test_dataloader_random_interpolation_samples = DataLoader(
        test_dataset_random_interpolation_samples,
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
    ).cuda()
    occ_ray_decoder = OccRayDecoder(
        fc_channel_list = [config.OCC_RAY_AE.OCC_RAY_LATENT_DIM+1] + list(config.OCC_RAY_AE.ARCH.DECODER.FC_CHANNEL_LIST) + [1],
    ).cuda()

    load_checkpoint(
        config.CHECKPOINT_LOAD_PATH,
        occ_ray_encoder,
        occ_ray_decoder,
    )

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader_encoded_interpolation_samples, '[TEST INTERP ENC]')):
            # batch_data = preprocess_batch(batch_data) # add 'surface_pt_mask'
            z = occ_ray_encoder_forward(
            occ_ray_encoder,
            batch_data,
            )
            z_bs = z.shape[0]
            assert z_bs % 2 == 0
            n_z_pairs = z_bs//2
            z_interpol = interpolate_latent_codes(z)
            occ_fcn_vals_pred = occ_ray_decoder_forward(
                occ_ray_decoder,
                batch_data['dense_pts'][::2, :].reshape((n_z_pairs, 1, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)).expand(-1, config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, -1).reshape((n_z_pairs * config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)),
                z_interpol.reshape((n_z_pairs * config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)),
            ).reshape((n_z_pairs, config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES))
            # loss = calc_loss(occ_fcn_vals_pred, batch_data['dense_occ_fcn_vals'])

        for batch_idx, batch_data in enumerate(tqdm(test_dataloader_random_interpolation_samples, '[TEST INTERP RND]')):
            # batch_data = preprocess_batch(batch_data) # add 'surface_pt_mask'
            z_bs = batch_data['z'].shape[0]
            assert z_bs % 2 == 0
            n_z_pairs = z_bs//2
            z_interpol = interpolate_latent_codes(batch_data['z'].cuda())
            occ_fcn_vals_pred = occ_ray_decoder_forward(
                occ_ray_decoder,
                batch_data['dense_pts'][::2, :].reshape((n_z_pairs, 1, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)).expand(-1, config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, -1).reshape((n_z_pairs * config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)),
                z_interpol.reshape((n_z_pairs * config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)),
            ).reshape((n_z_pairs, config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES))
