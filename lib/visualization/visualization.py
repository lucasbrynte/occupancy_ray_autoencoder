import numpy as np
from matplotlib import pyplot as plt
from lib.logging.tb import get_tb_writer
from lib.config.config import config

def prediction_barplot(occ_ray_rasterized, radial_samples, occ_fcn_vals_pred, occ_fcn_vals_target):
    fig, ax = plt.subplots(figsize=(10,2))
    grid = np.linspace(0, config.OCC_RAY_RESOLUTION-1, config.OCC_RAY_RESOLUTION) * config.RAY_RANGE / config.OCC_RAY_RESOLUTION
    ax.bar(
        grid,
        occ_ray_rasterized,
        width=1.0*config.RAY_RANGE/config.OCC_RAY_RESOLUTION,
        align='edge',
        color='gray',
        edgecolor='black',
    )
    ax.plot(
        np.vstack(2*[radial_samples*config.RAY_RANGE/config.OCC_RAY_RESOLUTION]),
        np.vstack([occ_fcn_vals_pred, occ_fcn_vals_target]),
    'r--')
    # ax.plot(radial_samples*config.RAY_RANGE/config.OCC_RAY_RESOLUTION, occ_fcn_vals_target, marker='o', markeredgecolor='red', markerfacecolor='None', linestyle='None')
    ax.plot(radial_samples*config.RAY_RANGE/config.OCC_RAY_RESOLUTION, occ_fcn_vals_target, 'r.')
    ax.plot(radial_samples*config.RAY_RANGE/config.OCC_RAY_RESOLUTION, occ_fcn_vals_pred, 'bx')
    return fig

def visualize_train_batch(step, occ_ray_rasterized, radial_samples, occ_fcn_vals_pred, occ_fcn_vals_target, write_tb=True):
    fig = prediction_barplot(occ_ray_rasterized, radial_samples, occ_fcn_vals_pred, occ_fcn_vals_target)
    if write_tb:
        get_tb_writer().add_figure('figures/prediction/train', fig, global_step=step)
