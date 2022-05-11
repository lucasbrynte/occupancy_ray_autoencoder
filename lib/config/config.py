import os
import shutil
import click
import json
import dotsi
config = dotsi.Dict()
from deepdiff import DeepDiff
import textwrap
import train_occ_ray_ae
import test_occ_ray_ae
from lib.logging.logging import log
from lib.logging.checkpoint import find_latest_checkpoint_file


@click.group()
@click.option('--exp-root', help='Root path in which to experiments.', default='out', show_default=True)
@click.option('--exp-name', help='Experiment name.', required=True)
@click.option('--old-exp-name', default=None, help='Old experiment name.')
@click.option('--checkpoint-load-path', default=None, help='Path from which to load model checkpoint. Alternatively, if loading an old experiment, "latest" can be given as an option in order to locate the latest checkpoint available from the experiment.')
def cli(**kwargs):
    config.EXP_ROOT = kwargs['exp_root']
    config.EXP_NAME = kwargs['exp_name']
    config.EXP_DIR = os.path.join(config.EXP_ROOT, config.EXP_NAME)
    config.CHECKPOINT_LOAD_PATH = kwargs['checkpoint_load_path']
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

    config.OLD_EXP_NAME = kwargs['old_exp_name']
    if config.OLD_EXP_NAME is not None:
        assert config.CHECKPOINT_LOAD_PATH is not None
        config.OLD_EXP_DIR = os.path.join(config.EXP_ROOT, config.OLD_EXP_NAME)
        with open(os.path.join(config.OLD_EXP_DIR, 'config.json'), 'r') as f:
            config.OLD = dotsi.Dict(json.load(f))
        if config.CHECKPOINT_LOAD_PATH.lower() == 'latest':
            assert config.OLD_EXP_NAME is not None
            fname_latest, latest_epoch = find_latest_checkpoint_file(os.path.join(config.OLD_EXP_DIR, 'checkpoints'))
            config.CHECKPOINT_LOAD_PATH = os.path.join(config.OLD_EXP_DIR, 'checkpoints', fname_latest)
            assert os.path.exists(config.CHECKPOINT_LOAD_PATH)

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
        # 'NORMALIZATION': None,
        # 'NORMALIZATION': {'METHOD': 'bn'},
        'NORMALIZATION': {'METHOD': 'gn', 'CHANNELS_PER_GROUP': 64},
    }

    assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'

    check_old_new_consistency('config.OCC_RAY_AE.RAY_RANGE')
    check_old_new_consistency('config.OCC_RAY_AE.OCC_RAY_LATENT_DIM')
    check_old_new_consistency('config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION')
    check_old_new_consistency('config.OCC_RAY_AE.ARCH')
    check_old_new_consistency('config.OCC_RAY_AE.OCC_RAY_RESOLUTION')

def check_old_new_consistency(new_name):
    tmp = new_name.split('.')
    old_name = '.'.join([tmp[0]] + ['OLD'] + tmp[1:])
    new_val = eval(new_name)
    try:
        old_val = eval(old_name)
    except KeyError:
        log().warning('Old configuration not found: {}'.format(old_name))
        return
    complex_structure = isinstance(old_val, (dict, list))
    if new_val != old_val:
        warn_msg = 'Configuration mismatch for [{}].'.format(new_name)
        if not complex_structure:
            warn_msg += '\n' + textwrap.indent('Reverting to old value: {} -> {}'.format(new_val, old_val), 2*' ')
        else:
            warn_msg += '\n' + textwrap.indent('Reverting to old value. Diff below:', 2*' ')
            warn_msg += '\n' + textwrap.indent(json.dumps(DeepDiff(old_val, new_val), indent=2), 2*' ')
        log().warning(warn_msg)
        exec(new_name + ' = ' + old_name)

@occ_ray_ae.command()
def train():
    config.OCC_RAY_AE.TRAIN = {}
    config.OCC_RAY_AE.VAL = {}
    config.OCC_RAY_AE.BS = 16
    config.OCC_RAY_AE.LR = 1e-4
    # config.OCC_RAY_AE.N_EPOCHS = 100000
    config.OCC_RAY_AE.N_EPOCHS = 15000
    config.OCC_RAY_AE.N_EPOCHS_CHECKPOINT_INTERVAL = 500
    # config.OCC_RAY_AE.EXTRA_CHECKPOINTS_AT_EPOCHS = []
    config.OCC_RAY_AE.EXTRA_CHECKPOINTS_AT_EPOCHS = list(range(1, 10, 1)) + list(range(10, 30, 5)) + list(range(30, 100, 10)) + list(range(100, 200, 25)) + list(range(200, 500, 50)) + list(range(500, 1000, 100)) + list(range(1000, 3000, 250))
    config.OCC_RAY_AE.N_BATCHES_VAL_INTERVAL = None # Just every epoch
    # config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = None
    config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = 256
    config.OCC_RAY_AE.N_BATCHES_TB_INTERVAL = 256
    # config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = 1024
    config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL = None # Just every epoch
    config.OCC_RAY_AE.SINKHORN_REG = {'enabled': False}
    # config.OCC_RAY_AE.SINKHORN_REG = {
    #     'enabled': True,
    #     # 'loss_coefficient': 1e0,
    #     # 'loss_coefficient': 1e-1,
    #     # 'loss_coefficient': 1e-2,
    #     # 'loss_coefficient': 1e-3,
    #     'loss_coefficient': 1e-4,
    #     'max_n_pairs': 12, # For OCC_RAY_LATENT_DIM = 16: 16 choose 2 = 16! / (14! * 2!) = 120 is max.
    #     # 'max_n_pairs': 32, # For OCC_RAY_LATENT_DIM = 16: 16 choose 2 = 16! / (14! * 2!) = 120 is max.
    #     'epsilon': 0.1, # = Entropy parameter
    #     'rho': 1., # = Unbalanced KL relaxation parameter
    # }
    if config.OCC_RAY_AE.SINKHORN_REG.enabled:
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'
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

@occ_ray_ae.command()
def test():
    assert config.OLD_EXP_NAME is not None
    assert config.CHECKPOINT_LOAD_PATH is not None

    config.OCC_RAY_AE.TEST = {}
    config.OCC_RAY_AE.BS = 16
    config.OCC_RAY_AE.TEST.DATA = {}
    config.OCC_RAY_AE.TEST.DATA.N_ENCODED_INTERPOLATION_PAIRS = 16
    config.OCC_RAY_AE.TEST.DATA.N_RANDOM_INTERPOLATION_PAIRS = 16
    config.OCC_RAY_AE.TEST.DATA.INTERPOLATION_RESOLUTION = 20
    config.OCC_RAY_AE.TEST.DATA.SYNTH_OCC_RAY_GENERATION_PARAMETERS = {
        'prob_center_occluded': 0.5,
        'alpha_start': 1,
        'beta_start': 1/0.15,
        'alpha_stop': 1,
        'beta_stop': 1/0.15,
        # 'prob_center_occluded': 0.75,
        # 'alpha_start': 1,
        # 'beta_start': 1/0.1,
        # 'alpha_stop': 1,
        # 'beta_stop': 1/0.05,
    }
    config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES = 200
    config.OCC_RAY_AE.RECONSTRUCTION_LOSS = config.OLD.OCC_RAY_AE.RECONSTRUCTION_LOSS

    test_occ_ray_ae.main()
