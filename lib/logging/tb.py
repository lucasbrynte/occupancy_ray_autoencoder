from torch.utils.tensorboard import SummaryWriter
from lib.config.config import config

global tb_writer
tb_writer = None

def initialize_tensorboard():
    global tb_writer
    tb_writer = SummaryWriter(
        log_dir = config.tb_path,
        flush_secs = 30,
    )
    return tb_writer

def get_tb_writer():
    return tb_writer
