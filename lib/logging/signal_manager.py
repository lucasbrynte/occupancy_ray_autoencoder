import torch
import numpy as np
from lib.config.config import config
from lib.logging.logging import log
from lib.logging.tb import get_tb_writer
from lib.visualization.visualization import visualize_train_batch

class SignalManager():
    def __init__(self):
        self._tb_writer = get_tb_writer()
        self._global_batch_cnt = 0

    def record_train_batch(
        self,
        batch_data,
        log_signals = False,
        log_signals_tb = False,
        visualize_pred = False,
    ):
        metrics = self._calculate_metrics(batch_data)
        if log_signals:
            log.info('[TRAIN] loss: {:.8f}'.format(metrics['loss']))
        if log_signals_tb:
            self._tb_writer.add_scalar("loss/train", metrics['loss'], self._global_batch_cnt)
            # self._tb_writer.flush()
        if visualize_pred:
            visualize_train_batch(
                'figures/prediction/train',
                self._global_batch_cnt,
                batch_data['occ_ray_rasterized'][0],
                batch_data['radial_samples'][0],
                batch_data['occ_fcn_vals_pred'][0],
                batch_data['occ_fcn_vals_target'][0],
            )
        self._global_batch_cnt += 1

    def _calculate_metrics(self, batch_data):
        metrics = {
            'loss': batch_data['loss'],
        }
        return metrics
