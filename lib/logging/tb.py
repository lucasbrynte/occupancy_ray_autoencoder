from torch.utils.tensorboard import SummaryWriter
from lib.config.config import config

global _tb_writer
_tb_writer = None

def _initialize_tensorboard():
    global _tb_writer
    _tb_writer = SummaryWriter(
        log_dir = config.TB_DIR,
        flush_secs = 30,
    )
    return _tb_writer

def get_tb_writer():
    if _tb_writer is None:
        _initialize_tensorboard()
    # assert _tb_writer is not None, 'Before calling get_tb_writer() for the first time, initialize_tensorboard() must have been called.'
    return _tb_writer
