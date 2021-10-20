import os
import shutil
import click
import dotsi
config = dotsi.Dict()
import train_occ_ray_ae


@click.group()
@click.option('--exp-root', help='Root path in which to experiments.', default='out', show_default=True)
@click.option('--exp-name', help='Experiment name.', required=True)
def cli(**kwargs):
    config.EXP_ROOT = kwargs['exp_root']
    config.EXP_NAME = kwargs['exp_name']
    config.EXP_DIR = os.path.join(config.EXP_ROOT, config.EXP_NAME)
    config.CONFIG_PATH = os.path.join(config.EXP_DIR, 'config.json')
    config.LOG_PATH = os.path.join(config.EXP_DIR, 'LOG')
    config.TB_DIR = os.path.join(config.EXP_DIR, 'tb')
    config.CHECKPOINT_DIR = os.path.join(config.EXP_DIR, 'checkpoints')
    config.VERSION_DUMP_DIR = os.path.join(config.EXP_DIR, 'version_dump')

    if os.path.exists(config.EXP_DIR):
        shutil.rmtree(config.EXP_DIR)
    os.makedirs(config.TB_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.VERSION_DUMP_DIR, exist_ok=True)

@cli.group()
def occ_ray_ae(**kwargs):
    config.OCC_RAY_AE = {}
    config.OCC_RAY_AE.OCC_RAY_RESOLUTION = 45
    config.OCC_RAY_AE.RAY_RANGE = 1
    config.OCC_RAY_AE.OCC_RAY_LATENT_DIM = 16
    config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION = 'occupancy_probability'
    config.OCC_RAY_AE.ARCH = {
        'ENCODER': {
            'CNN_CHANNEL_LIST': [2, 1024],
            'KSIZE_LIST': [45],
            'STRIDE_LIST': [1],
            'FC_CHANNEL_LIST': [1024, 1024, 1024, 1024, 1024],
        },
        'DECODER': {
            'FC_CHANNEL_LIST': [1024, 1024, 1024, 1024],
        },
    }

    assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'

@occ_ray_ae.command()
def train():
    config.OCC_RAY_AE.TRAIN = {}
    config.OCC_RAY_AE.VAL = {}
    config.OCC_RAY_AE.BS = 16
    config.OCC_RAY_AE.LR = 1e-4
    # config.OCC_RAY_AE.N_EPOCHS = 100000
    config.OCC_RAY_AE.N_EPOCHS = 15000
    config.OCC_RAY_AE.N_EPOCHS_CHECKPOINT_INTERVAL = 500
    config.OCC_RAY_AE.N_BATCHES_VAL_INTERVAL = None # Just every epoch
    # config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = None
    config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = 256
    # config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = 1024
    config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL = None # Just every epoch
    config.OCC_RAY_AE.TRAIN.DATA = {}
    config.OCC_RAY_AE.VAL.DATA = {}
    config.OCC_RAY_AE.TRAIN.DATA.N_SAMPLES = 1024*64
    config.OCC_RAY_AE.TRAIN.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS = {
        'prob_center_occluded': 0.75,
        'alpha_start': 1,
        'beta_start': 1/0.1,
        'alpha_stop': 1,
        'beta_stop': 1/0.05,
    }
    config.OCC_RAY_AE.VAL.DATA.N_SAMPLES = 1024
    config.OCC_RAY_AE.VAL.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS = {
        'prob_center_occluded': 0.25,
        'alpha_start': 1,
        'beta_start': 1/0.05,
        'alpha_stop': 1,
        'beta_stop': 1/0.1,
        # 'prob_center_occluded': 0.75,
        # 'alpha_start': 1,
        # 'beta_start': 1/0.1,
        # 'alpha_stop': 1,
        # 'beta_stop': 1/0.05,
    }
    config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES = 16
    # config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES = 8
    config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES = 16
    config.OCC_RAY_AE.N_OCC_FCN_SAMPLES = config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES + config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES
    config.OCC_RAY_AE.RECONSTRUCTION_LOSS = 'mse'
    # config.OCC_RAY_AE.RECONSTRUCTION_LOSS = 'bce'

    train_occ_ray_ae.main()
