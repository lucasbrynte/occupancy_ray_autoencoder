import platform
import logging
import textwrap
from tqdm import tqdm
from lib.config.config import config

# https://stackoverflow.com/a/38739634
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            # https://stackoverflow.com/a/1336640
            levelno = record.levelno
            if(levelno >= logging.CRITICAL): # >=50
                prefix = '{:<11s}'.format('[CRITICAL]')
                color = '\x1b[31m' # red
            elif(levelno >= logging.ERROR): # >=40
                prefix = '{:<11s}'.format('[ERROR]')
                color = '\x1b[31m' # red
            elif(levelno >= logging.WARNING): # >=30
                prefix = '{:<11s}'.format('[WARNING]')
                color = '\x1b[33m' # yellow
            elif(levelno >= logging.INFO): # >=20
                prefix = '{:<11s}'.format('[INFO]')
                color = '\x1b[32m' # green 
            elif(levelno >= logging.DEBUG): # >=10
                prefix = '{:<11s}'.format('[DEBUG]')
                color = '\x1b[35m' # pink
            else:
                prefix = ''
                color = '\x1b[0m' # normal
            # record.msg = color + record.msg +  '\x1b[0m'  # normal
            if '\n' in record.msg:
                record.msg = color + prefix +  '\x1b[0m' + '\n' + textwrap.indent(record.msg, 2*' ')
            else:
                record.msg = color + prefix +  '\x1b[0m' + record.msg

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
