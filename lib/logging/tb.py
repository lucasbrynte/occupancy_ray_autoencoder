from torch.utils.tensorboard import SummaryWriter
from lib.config.config import config

global _tb_writer
_tb_writer = None

def initialize_tensorboard():
    global _tb_writer
    _tb_writer = SummaryWriter(
        log_dir = config.TB_PATH,
        flush_secs = 30,
    )
    return _tb_writer

def get_tb_writer():
    return _tb_writer
