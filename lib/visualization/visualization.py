import numpy as np
from matplotlib import pyplot as plt
from lib.logging.tb import get_tb_writer
from lib.config.config import config

def _prediction_barplot(occ_ray_rasterized, radial_samples, occ_fcn_vals_pred, occ_fcn_vals_target):
    fig, ax = plt.subplots(figsize=(10,2))
    grid = np.linspace(0, config.OCC_RAY_AE.OCC_RAY_RESOLUTION-1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION) * config.OCC_RAY_AE.RAY_RANGE / config.OCC_RAY_AE.OCC_RAY_RESOLUTION
    ax.bar(
        grid,
        occ_ray_rasterized,
        width=1.0*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION,
        align='edge',
        color='gray',
        edgecolor='black',
    )
    ax.plot(
        np.vstack(2*[radial_samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION]),
        np.vstack([occ_fcn_vals_pred, occ_fcn_vals_target]),
    'r--')
    # ax.plot(radial_samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION, occ_fcn_vals_target, marker='o', markeredgecolor='red', markerfacecolor='None', linestyle='None')
    ax.plot(radial_samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION, occ_fcn_vals_target, 'r.')
    ax.plot(radial_samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION, occ_fcn_vals_pred, 'bx')
    ax.set_xlim(0, config.OCC_RAY_AE.RAY_RANGE)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(0, ymin), max(1, ymax))
    return fig

def prediction_barplot(occ_ray_rasterized, radial_samples, occ_fcn_vals_pred, occ_fcn_vals_target, write_tb=False, tb_tag=None, tb_step=None):
    if write_tb:
        assert tb_tag is not None
        assert tb_step is not None
    else:
        assert tb_tag is None
        assert tb_step is None
    fig = _prediction_barplot(occ_ray_rasterized, radial_samples, occ_fcn_vals_pred, occ_fcn_vals_target)
    if write_tb:
        get_tb_writer().add_figure(tb_tag, fig, global_step=tb_step)
