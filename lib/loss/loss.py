import torch
from torch import nn
from lib.config.config import config

def calc_loss(occ_fcn_vals_pred, occ_fcn_vals_target, point_weights):
    if config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'mse':
        loss = nn.functional.mse_loss(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='none')
        loss = torch.mean((point_weights * loss)[point_weights > 0])
    elif config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'bce':
        loss = nn.functional.binary_cross_entropy(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='none')
        loss = torch.mean((point_weights * loss)[point_weights > 0])
    else:
        assert False
    return loss

def calc_loss_anywhere_surface(batch_data, decoder_out):
    # Note: Could be simplified significantly if first calculating loss on "anywhere" and "surface" separately, and then combining the losses.
    occ_fcn_vals_pred = torch.cat([
        decoder_out['anywhere_occ_fcn_vals_pred'],
        decoder_out['surface_occ_fcn_vals_pred'],
    ], dim=1).cuda()
    occ_fcn_vals_target = torch.cat([
        batch_data['anywhere_occ_fcn_vals'],
        batch_data['surface_occ_fcn_vals'],
    ], dim=1).cuda()
    point_weights = torch.cat([
        batch_data['anywhere_pt_weights'],
        batch_data['surface_pt_weights'],
    ], dim=1).cuda()
    loss = calc_loss(occ_fcn_vals_pred, occ_fcn_vals_target, point_weights)
    return loss
