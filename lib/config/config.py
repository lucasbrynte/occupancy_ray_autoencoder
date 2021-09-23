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
