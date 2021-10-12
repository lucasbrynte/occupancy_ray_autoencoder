import logging
from tqdm import tqdm
from lib.config.config import config

# https://stackoverflow.com/a/38739634
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
            assert False

global _logger
_logger = None

def _initialize_logger():
    global _logger
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(TqdmLoggingHandler())

    fh = logging.FileHandler(config.LOG_PATH)
    fh.setLevel(logging.INFO)
    _logger.addHandler(fh)
    return _logger

def log():
    if _logger is None:
        _initialize_logger()
    # assert _logger is not None, 'Before calling get_logger() for the first time, initialize_logger() must have been called.'
    return _logger
