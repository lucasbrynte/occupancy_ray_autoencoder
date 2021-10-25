import numpy as np
import torch
from torch.utils.data import Dataset
from lib.config.config import config

class OccRayDataset(Dataset):
    def __init__(
        self,
        len,
        generation_parameters,
        random_seed = None,
        reset_seed_on_epoch_start = False,
        anywhere_samples = True,
        surface_samples = True,
        dense_samples = False,
    ):
        super().__init__()
        self._len = len
        self._generation_parameters = generation_parameters
        self._anywhere_samples = anywhere_samples
        self._surface_samples = surface_samples
        self._dense_samples = dense_samples
        self._random_seed = random_seed
        self._reset_seed_on_epoch_start = reset_seed_on_epoch_start
        self._reset_seed()

    def _reset_seed(self):
        self._random_generator = np.random.default_rng(self._random_seed)

    def __len__(self):
        return self._len

    def _sample_rasterized_occ_ray(self):
        center_occluded = self._random_generator.binomial(1, self._generation_parameters['prob_center_occluded'])
        def generate_start_stop_locations(N):
            start_intervals = 1 + np.floor(config.OCC_RAY_AE.OCC_RAY_RESOLUTION * self._random_generator.gamma(self._generation_parameters['alpha_start'], scale=1/self._generation_parameters['beta_start'], size=(N,)))
            stop_intervals = 1 + np.floor(config.OCC_RAY_AE.OCC_RAY_RESOLUTION * self._random_generator.gamma(self._generation_parameters['alpha_stop'], scale=1/self._generation_parameters['beta_stop'], size=(N,)))
            assert np.all(start_intervals >= 1)
            assert np.all(stop_intervals >= 1)
            locs = np.empty((2*N,), np.int64)
            locs[0::2] = start_intervals
            locs[1::2] = stop_intervals
            return locs
        avg_interval = (self._generation_parameters['beta_start'] + self._generation_parameters['beta_stop']) / 2
        margin_factor = 1
        locs = np.array([], np.int64)
        while np.sum(locs[1:]) < config.OCC_RAY_AE.OCC_RAY_RESOLUTION:
            locs = np.concatenate((locs, generate_start_stop_locations(int(np.ceil(margin_factor * 1/avg_interval)))), axis=0)
        if center_occluded:
            locs = locs[1:]
        locs = np.cumsum(locs)
        occ_ray_rasterized = np.zeros((config.OCC_RAY_AE.OCC_RAY_RESOLUTION), dtype=np.bool)
        if center_occluded:
            occ_ray_rasterized = ~occ_ray_rasterized
        for j, loc in enumerate(locs):
            if loc >= config.OCC_RAY_AE.OCC_RAY_RESOLUTION:
                n_locs = j
                break
            occ_ray_rasterized[loc:] = ~occ_ray_rasterized[loc:]
        locs = locs[:j]
        if locs.shape[0] > 0:
            assert locs[0] >= 1
        return occ_ray_rasterized, locs

    def _generate_anywhere_occ_fcn_samples(self, occ_ray_rasterized, n_samples):
        eps = 1e-6
        point_samples = self._random_generator.uniform(low=0, high=config.OCC_RAY_AE.OCC_RAY_RESOLUTION-eps, size=(n_samples,))
        occ_fcn_vals = occ_ray_rasterized[np.floor(point_samples).astype(np.int64)]
        return point_samples, occ_fcn_vals

    def _generate_dense_occ_fcn_samples(self, occ_ray_rasterized, n_samples):
        eps = 1e-6
        point_samples = np.linspace(0, config.OCC_RAY_AE.OCC_RAY_RESOLUTION-eps, n_samples)
        occ_fcn_vals = occ_ray_rasterized[np.floor(point_samples).astype(np.int64)]
        return point_samples, occ_fcn_vals

    def __getitem__(self, sample_idx):
        if self._reset_seed_on_epoch_start:
            if sample_idx == 0:
                self._reset_seed()
                self._prev_sample_idx = 0
            else:
                assert sample_idx == self._prev_sample_idx + 1, 'Previous sample idx: {}, current: {} != {}.'.format(self._prev_sample_idx, sample_idx, self._prev_sample_idx+1)
                self._prev_sample_idx += 1
        occ_ray_rasterized, all_surface_pts = self._sample_rasterized_occ_ray()
        first_gridpoint = 0.5 * config.OCC_RAY_AE.RAY_RANGE / config.OCC_RAY_AE.OCC_RAY_RESOLUTION
        last_gridpoint = (config.OCC_RAY_AE.OCC_RAY_RESOLUTION - 0.5) * config.OCC_RAY_AE.RAY_RANGE / config.OCC_RAY_AE.OCC_RAY_RESOLUTION
        grid = np.linspace(first_gridpoint, last_gridpoint, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)

        if self._anywhere_samples:
            anywhere_pts, anywhere_occ_fcn_vals = self._generate_anywhere_occ_fcn_samples(occ_ray_rasterized, config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES)
            anywhere_pt_weights = np.ones((config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES,))

        if self._surface_samples:
            n_surface_occ_fcn_samples = min(all_surface_pts.shape[0], config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES)
            surface_pts = np.pi * np.ones((config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES,)) # np.pi is an arbitrary placeholder for unused point samples.
            # surface_pts[:] = np.NaN # Although only the appropriate elements are masked out for the loss, backprop will still cause NaN gradients if any activations are NaN...
            surface_pt_weights = np.zeros((config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES,))
            if n_surface_occ_fcn_samples > 0:
                surface_pts[:n_surface_occ_fcn_samples] = self._random_generator.choice(all_surface_pts, size=(n_surface_occ_fcn_samples,), replace=False)
                surface_pt_weights[:n_surface_occ_fcn_samples] = 1.0
                # surface_pt_weights[:n_surface_occ_fcn_samples] = 1.0 * config.OCC_RAY_AE.MAX_N_SURFACE_OCC_FCN_SAMPLES / n_surface_occ_fcn_samples
            if config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability':
                border_val = 0.5
            else:
                assert False
            surface_occ_fcn_vals = border_val * np.ones_like(surface_pts)

        if self._dense_samples:
            dense_pts, dense_occ_fcn_vals = self._generate_dense_occ_fcn_samples(occ_ray_rasterized, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)
            # dense_pt_weights = np.ones((config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES,))

        sample = {}
        sample['occ_ray_rasterized'] = torch.tensor(occ_ray_rasterized.astype(np.float32))
        sample['grid'] = torch.tensor(grid.astype(np.float32))
        # NOTE: The grid above is constant!
        if self._anywhere_samples:
            sample['anywhere_pts'] = torch.tensor(anywhere_pts.astype(np.float32))
            sample['anywhere_pt_weights'] = torch.tensor(anywhere_pt_weights.astype(np.float32))
            sample['anywhere_occ_fcn_vals'] = torch.tensor(anywhere_occ_fcn_vals.astype(np.float32))
        if self._surface_samples:
            sample['n_surface_occ_fcn_samples'] = torch.tensor(n_surface_occ_fcn_samples, dtype=torch.int64)
            sample['surface_pts'] = torch.tensor(surface_pts.astype(np.float32))
            sample['surface_pt_weights'] = torch.tensor(surface_pt_weights.astype(np.float32))
            sample['surface_occ_fcn_vals'] = torch.tensor(surface_occ_fcn_vals.astype(np.float32))
        if self._dense_samples:
            sample['dense_pts'] = torch.tensor(dense_pts.astype(np.float32))
            # sample['dense_pt_weights'] = torch.tensor(dense_pt_weights.astype(np.float32))
            sample['dense_occ_fcn_vals'] = torch.tensor(dense_occ_fcn_vals.astype(np.float32))
        return sample
