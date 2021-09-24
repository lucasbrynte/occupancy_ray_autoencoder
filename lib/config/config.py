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
config.TB_PATH = os.path.join(config.EXP_PATH, 'tb')

config.OCC_RAY_AE = {}
config.OCC_RAY_AE.BS = 16
config.OCC_RAY_AE.LR = 1e-4
config.OCC_RAY_AE.N_EPOCHS = 100000
config.OCC_RAY_AE.N_BATCHES_LOG_INTERVAL = None
config.OCC_RAY_AE.N_BATCHES_VIZ_INTERVAL = None
config.OCC_RAY_AE.OCC_RAY_RESOLUTION = 45
config.OCC_RAY_AE.N_OCC_FCN_SAMPLES = 16
config.OCC_RAY_AE.OCC_RAY_LATENT_DIM = 16
config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION = 'occupancy_probability'
config.OCC_RAY_AE.RECONSTRUCTION_LOSS = 'mse'
# config.OCC_RAY_AE.RECONSTRUCTION_LOSS = 'bce'
