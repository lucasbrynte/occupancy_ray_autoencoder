import numpy as np
from matplotlib import pyplot as plt
from lib.logging.tb import get_tb_writer
from lib.config.config import config

def _prediction_barplot(
    samples = None,
    occ_fcn_vals_pred = None,
    occ_fcn_vals_target = None,
    occ_ray_rasterized = None,
    line_plot = False,
):
    assert not all([
        samples is None,
        occ_fcn_vals_pred is None,
        occ_fcn_vals_target is None,
        occ_ray_rasterized is None,
    ])
    if occ_fcn_vals_pred is not None or occ_fcn_vals_target is not None:
        assert samples is not None
    fig, ax = plt.subplots(figsize=(10,2))
    if occ_ray_rasterized is not None:
        grid = np.linspace(0, config.OCC_RAY_AE.OCC_RAY_RESOLUTION-1, config.OCC_RAY_AE.OCC_RAY_RESOLUTION) * config.OCC_RAY_AE.RAY_RANGE / config.OCC_RAY_AE.OCC_RAY_RESOLUTION
        ax.bar(
            grid,
            occ_ray_rasterized,
            width=1.0*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION,
            align='edge',
            color='gray',
            edgecolor='black',
        )
    if occ_fcn_vals_pred is not None and occ_fcn_vals_target is not None:
        ax.plot(
            np.vstack(2*[samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION]),
            np.vstack([occ_fcn_vals_pred, occ_fcn_vals_target]),
        'r--')
    if occ_fcn_vals_target is not None:
        # ax.plot(samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION, occ_fcn_vals_target, marker='o', markeredgecolor='red', markerfacecolor='None', linestyle='None')
        ax.plot(samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION, occ_fcn_vals_target, 'r.-' if line_plot else 'r.')
    if occ_fcn_vals_pred is not None:
        ax.plot(samples*config.OCC_RAY_AE.RAY_RANGE/config.OCC_RAY_AE.OCC_RAY_RESOLUTION, occ_fcn_vals_pred, 'b.-' if line_plot else 'bx')
    ax.set_xlim(0, config.OCC_RAY_AE.RAY_RANGE)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(0, ymin), max(1, ymax))
    return fig

def prediction_barplot(
    samples = None,
    occ_fcn_vals_pred = None,
    occ_fcn_vals_target = None,
    occ_ray_rasterized = None,
    line_plot = False,
    write_tb = False,
    tb_tag = None,
    tb_step = None,
):
    if write_tb:
        assert tb_tag is not None
        assert tb_step is not None
    else:
        assert tb_tag is None
        assert tb_step is None
    fig = _prediction_barplot(
        samples = samples,
        occ_fcn_vals_pred = occ_fcn_vals_pred,
        occ_fcn_vals_target = occ_fcn_vals_target,
        occ_ray_rasterized = occ_ray_rasterized,
        line_plot = line_plot,
    )
    if write_tb:
        get_tb_writer().add_figure(tb_tag, fig, global_step=tb_step)
