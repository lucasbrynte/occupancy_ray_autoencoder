import torch
import numpy as np
from lib.config.config import config
from lib.logging.logging import log
from lib.logging.tb import get_tb_writer
from lib.visualization.visualization import visualize_prediction

class SignalManager():
    def __init__(self):
        self._tb_writer = get_tb_writer()
        self._global_train_batch_cnt = 0
        self._global_val_batch_cnt = 0
        self._global_val_all_cnt = 0
        self._val_metrics = []

    def record_train_batch(
        self,
        batch_data,
        log_signals = False,
        log_signals_tb = False,
        visualize_pred = False,
    ):
        metrics = self._calculate_metrics(batch_data)
        if log_signals:
            log().info('[TRAIN] ' + ','.join([
                'loss: {:.8f}'.format(metrics['loss']),
                'acc: {:.2f}%'.format(100*metrics['acc']),
                'mean_abs_err_anywhere: {:.4f}'.format(metrics['mean_abs_err_anywhere']),
                'mean_abs_err_surface: {:.4f}'.format(metrics['mean_abs_err_surface']),
                'std_abs_err_anywhere: {:.4f}'.format(metrics['std_abs_err_anywhere']),
                'std_abs_err_surface: {:.4f}'.format(metrics['std_abs_err_surface']),
            ]))
        if log_signals_tb:
            self._tb_writer.add_scalar("loss/train", metrics['loss'], self._global_train_batch_cnt)
            self._tb_writer.add_scalar("acc/train", metrics['acc'], self._global_train_batch_cnt)
            self._tb_writer.add_scalar("mean_abs_err_anywhere/train", metrics['mean_abs_err_anywhere'], self._global_train_batch_cnt)
            self._tb_writer.add_scalar("mean_abs_err_surface/train", metrics['mean_abs_err_surface'], self._global_train_batch_cnt)
            self._tb_writer.add_scalar("std_abs_err_anywhere/train", metrics['std_abs_err_anywhere'], self._global_train_batch_cnt)
            self._tb_writer.add_scalar("std_abs_err_surface/train", metrics['std_abs_err_surface'], self._global_train_batch_cnt)
            # self._tb_writer.flush()
        if visualize_pred:
            visualize_prediction(
                'figures/prediction/train',
                self._global_train_batch_cnt,
                batch_data['occ_ray_rasterized'][0],
                np.concatenate([
                    batch_data['anywhere_pts'][0],
                    batch_data['surface_pts'][0, :batch_data['n_surface_occ_fcn_samples'][0]],
                ], axis=0),
                np.concatenate([
                    batch_data['anywhere_occ_fcn_vals_pred'][0],
                    batch_data['surface_occ_fcn_vals_pred'][0, :batch_data['n_surface_occ_fcn_samples'][0]],
                ], axis=0),
                np.concatenate([
                    batch_data['anywhere_occ_fcn_vals_target'][0],
                    batch_data['surface_occ_fcn_vals_target'][0, :batch_data['n_surface_occ_fcn_samples'][0]],
                ], axis=0),
            )
        self._global_train_batch_cnt += 1

    def record_val_batch(
        self,
        batch_data,
        visualize_pred = False,
    ):
        metrics = self._calculate_metrics(batch_data)
        self._val_metrics.append(metrics)

        if visualize_pred:
            visualize_prediction(
                'figures/prediction/val',
                self._global_val_batch_cnt,
                batch_data['occ_ray_rasterized'][0],
                np.concatenate([
                    batch_data['anywhere_pts'][0],
                    batch_data['surface_pts'][0, :batch_data['n_surface_occ_fcn_samples'][0]],
                ], axis=0),
                np.concatenate([
                    batch_data['anywhere_occ_fcn_vals_pred'][0],
                    batch_data['surface_occ_fcn_vals_pred'][0, :batch_data['n_surface_occ_fcn_samples'][0]],
                ], axis=0),
                np.concatenate([
                    batch_data['anywhere_occ_fcn_vals_target'][0],
                    batch_data['surface_occ_fcn_vals_target'][0, :batch_data['n_surface_occ_fcn_samples'][0]],
                ], axis=0),
            )
        self._global_val_batch_cnt += 1

    def calc_avg_val_metrics(
        self,
        log_signals = False,
        log_signals_tb = False,
    ):
        avg_metrics = {
            'loss': np.mean([ metrics['loss'] for metrics in self._val_metrics ]),
            'acc': np.mean([ metrics['acc'] for metrics in self._val_metrics ]),
            'mean_abs_err_anywhere': np.mean([ metrics['mean_abs_err_anywhere'] for metrics in self._val_metrics ]),
            'mean_abs_err_surface': np.mean([ metrics['mean_abs_err_surface'] for metrics in self._val_metrics ]),
            'std_abs_err_anywhere': np.mean([ metrics['std_abs_err_anywhere'] for metrics in self._val_metrics ]),
            'std_abs_err_surface': np.mean([ metrics['std_abs_err_surface'] for metrics in self._val_metrics ]),
        }
        if log_signals:
            log().info('[VAL] ' + ','.join([
                'loss: {:.8f}'.format(avg_metrics['loss']),
                'acc: {:.2f}%'.format(100*avg_metrics['acc']),
                'mean_abs_err_anywhere: {:.4f}'.format(avg_metrics['mean_abs_err_anywhere']),
                'mean_abs_err_surface: {:.4f}'.format(avg_metrics['mean_abs_err_surface']),
                'std_abs_err_anywhere: {:.4f}'.format(avg_metrics['std_abs_err_anywhere']),
                'std_abs_err_surface: {:.4f}'.format(avg_metrics['std_abs_err_surface']),
            ]))
        if log_signals_tb:
            self._tb_writer.add_scalar("loss/val", avg_metrics['loss'], self._global_val_all_cnt)
            self._tb_writer.add_scalar("acc/val", avg_metrics['acc'], self._global_val_all_cnt)
            self._tb_writer.add_scalar("mean_abs_err_anywhere/val", avg_metrics['mean_abs_err_anywhere'], self._global_val_all_cnt)
            self._tb_writer.add_scalar("mean_abs_err_surface/val", avg_metrics['mean_abs_err_surface'], self._global_val_all_cnt)
            self._tb_writer.add_scalar("std_abs_err_anywhere/val", avg_metrics['std_abs_err_anywhere'], self._global_val_all_cnt)
            self._tb_writer.add_scalar("std_abs_err_surface/val", avg_metrics['std_abs_err_surface'], self._global_val_all_cnt)
            # self._tb_writer.flush()
        self.clear_val_metrics()
        self._global_val_all_cnt += 1

    def clear_val_metrics(self):
        self._val_metrics.clear()

    def _calculate_metrics(self, batch_data):
        metrics = {
            'loss': batch_data['loss'],
            'acc': self._calculate_accuracy(batch_data['anywhere_occ_fcn_vals_pred'], batch_data['anywhere_occ_fcn_vals_target']),
            'mean_abs_err_anywhere': self._calculate_mean_abs_err(batch_data['anywhere_occ_fcn_vals_pred'], batch_data['anywhere_occ_fcn_vals_target']),
            'mean_abs_err_surface': self._calculate_mean_abs_err(batch_data['surface_occ_fcn_vals_pred'][batch_data['surface_pt_mask']], batch_data['surface_occ_fcn_vals_target'][batch_data['surface_pt_mask']]),
            'std_abs_err_anywhere': self._calculate_std_abs_err(batch_data['anywhere_occ_fcn_vals_pred'], batch_data['anywhere_occ_fcn_vals_target']),
            'std_abs_err_surface': self._calculate_std_abs_err(batch_data['surface_occ_fcn_vals_pred'][batch_data['surface_pt_mask']], batch_data['surface_occ_fcn_vals_target'][batch_data['surface_pt_mask']]),
        }
        return metrics


    def _calculate_accuracy(self, occ_fcn_vals_pred, occ_fcn_vals_target):
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'
        border_val = 0.5
        hard_predictions = occ_fcn_vals_pred >= border_val
        assert np.all(np.isclose(np.abs(occ_fcn_vals_target - border_val), border_val))
        accuracy = np.mean(hard_predictions == occ_fcn_vals_target.astype(bool))
        return accuracy

    def _calculate_mean_abs_err(self, occ_fcn_vals_pred, occ_fcn_vals_target):
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'
        mean_abs_err = np.mean(np.abs(occ_fcn_vals_pred - occ_fcn_vals_target))
        return mean_abs_err

    def _calculate_std_abs_err(self, occ_fcn_vals_pred, occ_fcn_vals_target):
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'
        std_abs_err = np.std(np.abs(occ_fcn_vals_pred - occ_fcn_vals_target))
        return std_abs_err
