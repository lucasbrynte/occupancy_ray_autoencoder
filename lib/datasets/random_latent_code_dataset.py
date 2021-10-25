import numpy as np
import torch
from torch.utils.data import Dataset
from lib.config.config import config

class RandomLatentCodeDataset(Dataset):
    def __init__(
        self,
        len,
        z_mu = 0,
        z_sigma = 0,
        random_seed = None,
        reset_seed_on_epoch_start = False,
    ):
        super().__init__()
        self._len = len
        self._random_seed = random_seed
        self._reset_seed_on_epoch_start = reset_seed_on_epoch_start
        self._reset_seed()
        self._z_mu = z_mu
        self._z_sigma = z_sigma

    def _reset_seed(self):
        self._random_generator = np.random.default_rng(self._random_seed)

    def __len__(self):
        return self._len

    def __getitem__(self, sample_idx):
        if self._reset_seed_on_epoch_start:
            if sample_idx == 0:
                self._reset_seed()
                self._prev_sample_idx = 0
            else:
                assert sample_idx == self._prev_sample_idx + 1, 'Previous sample idx: {}, current: {} != {}.'.format(self._prev_sample_idx, sample_idx, self._prev_sample_idx+1)
                self._prev_sample_idx += 1

        z = self._random_generator.normal(self._z_mu, self._z_sigma, config.OCC_RAY_AE.OCC_RAY_LATENT_DIM)
        eps = 1e-6
        dense_pts = np.linspace(0, config.OCC_RAY_AE.OCC_RAY_RESOLUTION-eps, config.OCC_RAY_AE.N_DENSE_OCC_FCN_SAMPLES)

        sample = {}
        sample['z'] = torch.tensor(z, dtype=torch.float32)
        sample['dense_pts'] = torch.tensor(dense_pts.astype(np.float32))
        return sample
