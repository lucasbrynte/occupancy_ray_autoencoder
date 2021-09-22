from torch.utils.tensorboard import SummaryWriter
from lib.config.config import config

tb_writer = SummaryWriter(config.tb_path)
