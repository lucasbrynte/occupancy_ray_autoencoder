import os
import argparse
import dotsi

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--exp-root', dest='EXP_ROOT', help='Root path in which to experiments.', default='out')
parser.add_argument('--exp-name', dest='EXP_NAME', help='Experiment name.', required=True)
args = parser.parse_args()

config = dotsi.Dict()
config.EXP_ROOT = args.EXP_ROOT
config.EXP_NAME = args.EXP_NAME
config.EXP_PATH = os.path.join(config.EXP_ROOT, config.EXP_NAME)
config.LOG_PATH = os.path.join(config.EXP_PATH, 'LOG')
config.TB_PATH = os.path.join(config.EXP_PATH, 'tb')
config.CHECKPOINT_PATH = os.path.join(config.EXP_PATH, 'checkpoints')
config.VERSION_DUMP_PATH = os.path.join(config.EXP_PATH, 'version_dump')

config.OCC_RAY_AE = {}
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
config.OCC_RAY_AE.OCC_RAY_RESOLUTION = 45
config.OCC_RAY_AE.RAY_RANGE = 1
config.OCC_RAY_AE.TRAIN.SYNTH_OCC_RAY_GENERATION_PARAMETERS = {
    'prob_center_occluded': 0.75,
    'alpha_start': 1,
    'beta_start': 1/0.1,
    'alpha_stop': 1,
    'beta_stop': 1/0.05,
}
config.OCC_RAY_AE.VAL.SYNTH_OCC_RAY_GENERATION_PARAMETERS = {
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
config.OCC_RAY_AE.OCC_RAY_LATENT_DIM = 16
config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION = 'occupancy_probability'
config.OCC_RAY_AE.RECONSTRUCTION_LOSS = 'mse'
# config.OCC_RAY_AE.RECONSTRUCTION_LOSS = 'bce'
