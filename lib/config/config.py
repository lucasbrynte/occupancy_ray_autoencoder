import os
import argparse
from attrdict import AttrDict

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--exp-root', dest='EXP_ROOT', help='Root path in which to experiments.', default='out')
parser.add_argument('--exp-name', dest='EXP_NAME', help='Experiment name.', required=True)
args = parser.parse_args()

config = AttrDict()
config.EXP_ROOT = args.EXP_ROOT
config.EXP_NAME = args.EXP_NAME
config.EXP_PATH = os.path.join(config.EXP_ROOT, config.EXP_NAME)
config.TB_PATH = os.path.join(config.EXP_PATH, 'tb')

config.BS = 16
config.LR = 1e-4
config.N_EPOCHS = 100000
config.N_BATCHES_LOG_INTERVAL = None
config.N_BATCHES_VIZ_INTERVAL = None
config.OCC_RAY_RESOLUTION = 45
config.N_OCC_FCN_SAMPLES = 16
config.OCC_RAY_LATENT_DIM = 16
