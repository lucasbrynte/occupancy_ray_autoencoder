import torch
from lib.config.config import config

def preprocess_batch(batch_data):
    assert torch.all(batch_data['anywhere_pt_weights'] > 0) # Simplifies things if only surface points are variable-length
    # batch_data['anywhere_pt_mask'] = batch_data['anywhere_pt_weights'] > 0
    batch_data['surface_pt_mask'] = batch_data['surface_pt_weights'] > 0
    return batch_data

def occ_ray_encoder_forward(
    occ_ray_encoder,
    batch_data,
):
    bs = batch_data['occ_ray_rasterized'].shape[0]
    z = occ_ray_encoder(batch_data['occ_ray_rasterized'].reshape((bs, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)).cuda(), batch_data['grid'].reshape((bs, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)).cuda())
    return z

def occ_ray_decoder_forward(
    occ_ray_decoder,
    point_samples,
    z,
):
    bs, n_samples = point_samples.shape
    occ_fcn_vals_pred = occ_ray_decoder(
        z.reshape((bs, 1, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)).expand((-1, n_samples, -1)).reshape((bs*n_samples, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)),
        point_samples.reshape((bs*n_samples, 1)).cuda(),
    ).reshape((bs, n_samples))
    if config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'bce':
        occ_fcn_vals_pred = torch.sigmoid(occ_fcn_vals_pred)
    return occ_fcn_vals_pred

def occ_ray_decoder_forward_anywhere_surface(
    occ_ray_decoder,
    batch_data,
    z,
):
    occ_fcn_vals_pred = occ_ray_decoder_forward(
        occ_ray_decoder,
        torch.cat([
            batch_data['anywhere_pts'],
            batch_data['surface_pts'],
        ], dim=1),
        z,
    )
    anywhere_occ_fcn_vals_pred = occ_fcn_vals_pred[:, :config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES]
    surface_occ_fcn_vals_pred = occ_fcn_vals_pred[:, -config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES:]
    return {
        'anywhere_occ_fcn_vals_pred': anywhere_occ_fcn_vals_pred,
        'surface_occ_fcn_vals_pred': surface_occ_fcn_vals_pred,
    }

def occ_ray_ae_forward(
    occ_ray_encoder,
    occ_ray_decoder,
    batch_data,
):
    z = occ_ray_encoder_forward(occ_ray_encoder, batch_data)
    decoder_out = occ_ray_decoder_forward_anywhere_surface(occ_ray_decoder, batch_data, z)
    return {
        'z': z,
        'anywhere_occ_fcn_vals_pred': decoder_out['anywhere_occ_fcn_vals_pred'],
        'surface_occ_fcn_vals_pred': decoder_out['surface_occ_fcn_vals_pred'],
    }
