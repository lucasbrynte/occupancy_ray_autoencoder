import os
import argparse
from attrdict import AttrDict

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--exp-root', help='Root path in which to experiments.', default='out')
parser.add_argument('--exp-name', help='Experiment name.', required=True)
args = parser.parse_args()

config = AttrDict()
config.exp_root = args.exp_root
config.exp_name = args.exp_name
config.exp_path = os.path.join(config.exp_root, config.exp_name)
config.tb_path = os.path.join(config.exp_path, 'tb')

config.BS = 16
config.LR = 1e-4
config.N_EPOCHS = 100000
config.N_BATCHES_LOG_INTERVAL = None
config.N_BATCHES_VIZ_INTERVAL = None
config.OCC_RAY_RESOLUTION = 45
config.N_OCC_FCN_SAMPLES = 16
config.OCC_RAY_LATENT_DIM = 16
