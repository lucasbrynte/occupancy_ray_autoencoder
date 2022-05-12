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

    @property
    def grid(self):
        # Define a grid where each pixel stores its own location.
        # The coordinate system coincides
        grid = np.arange(config.OCC_RAY_AE.OCC_RAY_RESOLUTION).astype(np.float64) + 0.5
        assert grid.shape == (config.OCC_RAY_AE.OCC_RAY_RESOLUTION,)
        assert np.isclose(grid[0], 0.5)
        assert np.isclose(grid[-1], config.OCC_RAY_AE.OCC_RAY_RESOLUTION - 0.5)
        return grid

    def _occ_ray_to_sdf(self, occ_ray_rasterized, all_surface_pts):
        assert occ_ray_rasterized.dtype == np.bool and occ_ray_rasterized.shape == (config.OCC_RAY_AE.OCC_RAY_RESOLUTION,)
        assert all_surface_pts.dtype == np.int64 and len(all_surface_pts.shape) == 1

        # This might be efficient due to the vectorized oneliner, but also might not due to being relatively heavy in memory, and maybe also in terms of computation.
        # All grid point <-> surface point distances are computed.
        occ_ray_sdf_vals = self._calc_sdf_at_samples_for_occ_ray(occ_ray_rasterized, all_surface_pts, self.grid)

        return occ_ray_sdf_vals

    def _calc_sdf_at_samples_for_occ_ray(self, occ_ray_rasterized, all_surface_pts, point_samples):
        assert occ_ray_rasterized.dtype == np.bool and occ_ray_rasterized.shape == (config.OCC_RAY_AE.OCC_RAY_RESOLUTION,)
        assert len(all_surface_pts.shape) == 1
        assert len(point_samples.shape) == 1

        all_surface_pts = all_surface_pts.astype(np.float64)
        point_samples = point_samples.astype(np.float64)

        assert np.all(point_samples >= 0)
        assert np.all(point_samples <= config.OCC_RAY_AE.OCC_RAY_RESOLUTION)
        eps = 1e-3
        point_samples[np.isclose(point_samples, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)] = config.OCC_RAY_AE.OCC_RAY_RESOLUTION - eps
        assert np.all(point_samples < config.OCC_RAY_AE.OCC_RAY_RESOLUTION) # Strict inequality important in order to use np.floor later on and end up with proper pixel / bin indices.

        # The surface points represent the index of the first "pixel" of each new section of occupied / free space.
        # Each section has a start and and end point, except the initial / final sections, for which only the end point / start point, respectively, is given.
        # In addition to the given surface points, one may regard the boundary points as implied additional surface points, at "pixels" 0 and config.OCC_RAY_AE.OCC_RAY_RESOLUTION, the latter being out of bounds.
        # Alternatively, one may see the initial & final sections as being open / infinite towards the boundaries.
        # We will take the former perspective when defining the SDF values, such that the boundary sections are always "closed".
        # The second approach suffers from the SDF not being defined (or perhaps being infinite) if there are no surface points, which happens if the whole space is either empty or fully occupied.

        assert np.all(all_surface_pts >= 1)
        assert np.all(all_surface_pts < config.OCC_RAY_AE.OCC_RAY_RESOLUTION)
        # This more explicit approach will fail if there are no surface points:
        # assert all_surface_pts[0] >= 1
        # assert all_surface_pts[-1] < config.OCC_RAY_AE.OCC_RAY_RESOLUTION

        # Augment the surface points to always include the boundaries as well.
        all_surface_pts = np.concatenate((np.array([0]), all_surface_pts, np.array([config.OCC_RAY_AE.OCC_RAY_RESOLUTION])), axis=0)
        assert len(all_surface_pts.shape) == 1

        # First compute absolute distances to the respective closest surface points
        # All grid point <-> surface point distances are computed, and the minimum ones are selected.
        occ_ray_sdf_vals = np.min(np.abs(point_samples[:, None] - all_surface_pts[None, :]), axis=1)

        # Sample binary occupancy function at the point samples.
        occ_fcn_vals = occ_ray_rasterized[np.floor(point_samples).astype(np.int64)]

        # Apply a sign change at all exterior points, in which case we want negative distances.
        occ_ray_sdf_vals[~occ_fcn_vals] *= -1.0
        # occ_ray_sdf_vals[~occ_ray_rasterized] *= -1.0

        return occ_ray_sdf_vals

    def _generate_anywhere_occ_fcn_samples(self, occ_ray_rasterized, all_surface_pts, n_samples):
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION in ['occupancy_probability', 'SDF']

        # eps = 1e-6 # This is large enough to distinguish OCC_RAY_RESOLUTION - eps from OCC_RAY_RESOLUTION, provided we use double-precision. However! For single-precisino it might not.
        eps = 1e-3
        point_samples = self._random_generator.uniform(low=0, high=config.OCC_RAY_AE.OCC_RAY_RESOLUTION-eps, size=(n_samples,))

        if config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability':
            occ_fcn_vals = occ_ray_rasterized[np.floor(point_samples).astype(np.int64)]
        elif config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'SDF':
            occ_fcn_vals = self._calc_sdf_at_samples_for_occ_ray(occ_ray_rasterized, all_surface_pts, point_samples)
        else:
            assert False

        return point_samples, occ_fcn_vals

    def _generate_dense_occ_fcn_samples(self, occ_ray_rasterized, all_surface_pts, n_samples):
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION in ['occupancy_probability', 'SDF']

        # eps = 1e-6 # This is large enough to distinguish OCC_RAY_RESOLUTION - eps from OCC_RAY_RESOLUTION, provided we use double-precision. However! For single-precisino it might not.
        eps = 1e-3
        point_samples = np.linspace(0, config.OCC_RAY_AE.OCC_RAY_RESOLUTION-eps, n_samples)

        if config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability':
            occ_fcn_vals = occ_ray_rasterized[np.floor(point_samples).astype(np.int64)]
        elif config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'SDF':
            occ_fcn_vals = self._calc_sdf_at_samples_for_occ_ray(occ_ray_rasterized, all_surface_pts, point_samples)
        else:
            assert False

        return point_samples, occ_fcn_vals

    def __getitem__(self, sample_idx):
        # assert config.OCC_RAY_AE.OBSERVATION_REPRESENTATION == 'occupancy_probability'
        # assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'occupancy_probability'
        assert config.OCC_RAY_AE.OBSERVATION_REPRESENTATION in ['occupancy_probability', 'SDF']
        assert config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION in ['occupancy_probability', 'SDF']
        if self._reset_seed_on_epoch_start:
            if sample_idx == 0:
                self._reset_seed()
                self._prev_sample_idx = 0
            else:
                assert sample_idx == self._prev_sample_idx + 1, 'Previous sample idx: {}, current: {} != {}.'.format(self._prev_sample_idx, sample_idx, self._prev_sample_idx+1)
                self._prev_sample_idx += 1

        # Sample a binary occupancy function on an equidistant grid:
        occ_ray_rasterized, all_surface_pts = self._sample_rasterized_occ_ray()

        if config.OCC_RAY_AE.OBSERVATION_REPRESENTATION == 'occupancy_probability':
            # The binary rasterized ray is used as observation as-is.
            occ_ray_observation = occ_ray_rasterized
        elif config.OCC_RAY_AE.OBSERVATION_REPRESENTATION == 'SDF':
            sdf_rasterized = self._occ_ray_to_sdf(occ_ray_rasterized, all_surface_pts)
            occ_ray_observation = sdf_rasterized
        else:
            # As of yet, no other observation representations are implemented.
            assert False

        first_gridpoint = 0.5 * config.OCC_RAY_AE.RAY_RANGE / config.OCC_RAY_AE.OCC_RAY_RESOLUTION
        last_gridpoint = (config.OCC_RAY_AE.OCC_RAY_RESOLUTION - 0.5) * config.OCC_RAY_AE.RAY_RANGE / config.OCC_RAY_AE.OCC_RAY_RESOLUTION
        grid = np.linspace(first_gridpoint, last_gridpoint, config.OCC_RAY_AE.OCC_RAY_RESOLUTION)

        if self._anywhere_samples:
            anywhere_pts, anywhere_occ_fcn_vals = self._generate_anywhere_occ_fcn_samples(occ_ray_rasterized, all_surface_pts, config.OCC_RAY_AE.N_ANYWHERE_OCC_FCN_SAMPLES)
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
            elif config.OCC_RAY_AE.RECONSTRUCTION_REPRESENTATION == 'SDF':
                border_val = 0.0
            else:
                assert False
            surface_occ_fcn_vals = border_val * np.ones_like(surface_pts)

        if self._dense_samples:
            dense_pts, dense_occ_fcn_vals = self._generate_dense_occ_fcn_samples(occ_ray_rasterized, all_surface_pts, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)
            # dense_pt_weights = np.ones((config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES,))

        sample = {}
        sample['occ_ray_rasterized'] = torch.tensor(occ_ray_rasterized.astype(np.float32))
        sample['occ_ray_observation'] = torch.tensor(occ_ray_observation.astype(np.float32))
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
