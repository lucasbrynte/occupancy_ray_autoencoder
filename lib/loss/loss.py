import torch
from torch import nn
from scipy.special import comb
from lib.config.config import config
from lib.utils.misc import k_random_pairs
from lib.utils.ot import BatchVanillaSinkhorn, calc_S_eps_zero_robust_pytorch

def calc_loss(occ_fcn_vals_pred, occ_fcn_vals_target, point_weights=None):
    if config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'mse':
        loss = nn.functional.mse_loss(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='none')
        if point_weights is None:
            loss = torch.mean(loss)
        else:
            loss = torch.mean((point_weights * loss)[point_weights > 0])
    elif config.OCC_RAY_AE.RECONSTRUCTION_LOSS == 'bce':
        loss = nn.functional.binary_cross_entropy(occ_fcn_vals_pred, occ_fcn_vals_target, reduction='none')
        if point_weights is None:
            loss = torch.mean(loss)
        else:
            loss = torch.mean((point_weights * loss)[point_weights > 0])
    else:
        assert False
    return loss

def calc_loss_anywhere_surface(batch_data, ae_out):
    # Note: Could be simplified significantly if first calculating loss on "anywhere" and "surface" separately, and then combining the losses.
    occ_fcn_vals_pred = torch.cat([
        ae_out['anywhere_occ_fcn_vals_pred'],
        ae_out['surface_occ_fcn_vals_pred'],
    ], dim=1).cuda()
    occ_fcn_vals_target = torch.cat([
        batch_data['anywhere_occ_fcn_vals'],
        batch_data['surface_occ_fcn_vals'],
    ], dim=1).cuda()
    point_weights = torch.cat([
        batch_data['anywhere_pt_weights'],
        batch_data['surface_pt_weights'],
    ], dim=1).cuda()
    loss = calc_loss(occ_fcn_vals_pred, occ_fcn_vals_target, point_weights=point_weights)
    return loss

def calc_pairwise_sinkhorn_regularization_loss(batch_data, ae_out):
    assert batch_data['occ_ray_rasterized'].shape == (config.OCC_RAY_AE.BS, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)
    assert ae_out['z'].shape == (config.OCC_RAY_AE.BS, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)
    assert config.OCC_RAY_AE.BS >= 2, "Random pairs for Sinkhorn divergence regularization can not be sampled unless the batch size is at least 2."
    max_pairs_possible = comb(config.OCC_RAY_AE.BS, 2, exact=True) # BS choose 2 = BS*(BS-1)/2
    n_pairs = min(config.OCC_RAY_AE.SINKHORN_REG.max_n_pairs, max_pairs_possible)

    # Sample a number of pairs of latent codes from z and corresponding pairs of rays from inputs.
    index_pairs = k_random_pairs(config.OCC_RAY_AE.BS, n_pairs)

    z_a = ae_out['z'][index_pairs[:, 0], :]
    z_b = ae_out['z'][index_pairs[:, 1], :]
    assert z_a.shape == (n_pairs, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)
    assert z_b.shape == (n_pairs, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)
    latent_distances = torch.sqrt(torch.sum((z_b - z_a)**2, dim=1))

    # with torch.no_grad(): # Ineffective for this OT library, since set_grad_enabled is called, toggling gradient on / off according to parameters assume_convergence, nits, nits_grad.
    rays_a = batch_data['occ_ray_rasterized'][index_pairs[:, 0], :].cuda()
    rays_b = batch_data['occ_ray_rasterized'][index_pairs[:, 1], :].cuda()
    assert rays_a.shape == (n_pairs, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)
    assert rays_b.shape == (n_pairs, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)

    # solver = None
    # Greedy default settings:
    solver = BatchVanillaSinkhorn(
        nits = 100,
        # nits_grad = 5,
        nits_grad = 0, # We don't need any gradient iterations however.
        tol = 1e-3,
        assume_convergence = True,
    )
    # solver = BatchVanillaSinkhorn( # Override greedy default settings for higher precision
    #     nits = 5000,
    #     nits_grad = 15,
    #     tol = 1e-8,
    #     assume_convergence = True,
    # )

    x = torch.linspace(0, 1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION, dtype=torch.float32)[None, :, None].cuda().repeat(n_pairs, 1, 1)
    # x = torch.arange(config.OCC_RAY_AE.OCC_RAY_RESOLUTION, dtype=torch.float32)[None, :, None].cuda().repeat(n_pairs, 1, 1)
    true_sinkhorn_divergences = calc_S_eps_zero_robust_pytorch(
        rays_a, # shape (bs, n_dirac_masses)
        rays_b, # shape (bs, n_dirac_masses)
        x, # shape (bs, n_dirac_masses, n_space_dimensions)
        x, # shape (bs, n_dirac_masses, n_space_dimensions)
        solver = solver,
        epsilon = config.OCC_RAY_AE.SINKHORN_REG.epsilon, # = Entropy parameter
        rho = config.OCC_RAY_AE.SINKHORN_REG.rho, # = Unbalanced KL relaxation parameter
    )

    assert latent_distances.shape == (n_pairs,)
    assert true_sinkhorn_divergences.shape == (n_pairs,), "Some Sinkhorn divergences are NaN!\n{}".format(true_sinkhorn_divergences)

    assert not torch.any(torch.isnan(latent_distances))
    sinkhorn_nan_mask = torch.isnan(true_sinkhorn_divergences)
    if torch.any(sinkhorn_nan_mask):
        sinkhorn_nan_percentage = 100. * torch.sum(sinkhorn_nan_mask).item() / n_pairs
        log().warning("{}% NaN:s detected among sinkhorn divergence targets! Excluded from loss.".format(sinkhorn_nan_percentage))
        latent_distances = latent_distances[~sinkhorn_nan_mask]
        true_sinkhorn_divergences = true_sinkhorn_divergences[~sinkhorn_nan_mask]
    assert not torch.any(torch.isnan(latent_distances))
    assert not torch.any(sinkhorn_nan_mask)

    # MSE loss over all pairs of distance / divergence discrepancy.
    return torch.mean((latent_distances - true_sinkhorn_divergences)**2)
