from torch.utils.data import Dataset

class OcclRayDataset(Dataset):
    def __init__(self, range=1, resolution=64, len=1024, n_occ_fcn_samples=4):
        super().__init__()
        self._range = range
        self._resolution = resolution
        self._len = len
        self._n_occ_fcn_samples = n_occ_fcn_samples
        self._generation_parameters = self._init_generation_parameters(self)
        pass

    def _generation_parameters(self):
        # Eventually to be replaced / augmented with parameters corresponding to occlusion rays of real data.
        # Parameters may also be "time"-varying, which might be beneficial for fitting to real data.
        # A general distribution to consider would be the generalized gamma distribution, which generalizes both the gamma distribution and the weibull distribution, each of which generalizes the exponential distribution.
        # Another simple option is to consider the gamma distribution with shape parameter alpha=k=1, i.e. in fact an exponential distribution with rate parameter lambda=beta=1/theta=1/scale, which could later be generalized beyond the exponmential distribution.
        # Note that for an exponentially distributed random variable X with rate parameter lambda (inverse scale), the random variable Y=floor(X) is geometrically distributed with success probability parameter p=1-exp(-lambda).
        # By instead taking Y=ceil(X), one can model the number of trials before success, instead of the number of failures before success.
        # We consider the continuous gamma distribution, and then apply the ceil() operation to get discrete support on positive integers.
        # Maybe one should consider a discretized gamma distribution instead..? https://www.tandfonline.com/doi/pdf/10.1080/03610926.2011.563014
        generation_parameters = {}
        generation_parameters['prob_center_occluded'] = 0.75
        generation_parameters['alpha_start'] = 1
        generation_parameters['beta_start'] = 0.1
        generation_parameters['alpha_stop'] = 1
        generation_parameters['beta_stop'] = 0.05
        return generation_parameters

    def __len__(self):
        return self._len

    def sample_rasterized_occl_ray(self):
        center_occluded = np.random.binomial(1, generation_parameters['prob_center_occluded'])
        def generate_start_stop_locations(N):
            start_intervals = 1 + np.floor(self._resolution * np.random.gamma(self._generation_parameters['alpha_start'], scale=1/self._generation_parameters['beta_start'], size=(N,)))
            stop_intervals = 1 + np.floor(self._resolution * np.random.gamma(self._generation_parameters['alpha_stop'], scale=1/self._generation_parameters['beta_stop'], size=(N,)))
            assert np.all(start_intervals >= 1)
            assert np.all(stop_intervals >= 1)
            locs = np.empty((2*N,), np.int64)
            locs[0::2] = start_intervals
            locs[1::2] = stop_intervals
            return locs
        avg_interval = (self._generation_parameters['beta_start'] + self._generation_parameters['beta_stop']) / 2
        margin_factor = 1
        locs = np.array([], np.int64)
        while np.sum(locs[1:]) < self._resolution:
            locs = np.concatenate((locs, generate_start_stop_locations(margin_factor * 1/avg_interval)), axis=0)
        if center_occluded:
            locs = locs[1:]
        locs = np.cumsum(locs)
        occ_ray_rasterized = np.zeros((self._resolution), dtype=np.bool)
        if center_occluded:
            occ_ray_rasterized = ~occ_ray_rasterized
        for loc in locs:
            if loc >= self._resolution:
                break
            occ_ray_rasterized[loc:] = ~occ_ray_rasterized[loc:]
        return occ_ray_rasterized

    def generate_occ_fcn_samples_along_ray(self, occ_ray_rasterized):
        eps = 1e-6
        radial_samples = np.random.uniform(low=0, high=self._resolution-eps, size(self._n_occ_fcn_samples,))
        occ_fcn_vals = occ_ray_rasterized[np.floor(radial_samples).astype(np.int64)]
        return radial_samples, occ_fcn_vals

    def __get_item__(self, sample_idx):
        occ_ray_rasterized = self.sample_rasterized_occl_ray()
        grid = np.linspace(0.5, self._resolution-0.5, self._resolution)
        radial_samples, occ_fcn_vals = generate_occ_fcn_samples_along_ray(occ_ray_rasterized)
        sample = {
            'occ_ray_rasterized': torch.tensor(occ_ray_rasterized.astype(np.int64)),
            'radial_samples': torch.tensor(radial_samples.astype(np.float32)),
            'occ_fcn_vals': torch.tensor(occ_fcn_vals.astype(np.float32)),
            'grid': torch.tensor(grid.astype(np.float32)),
        }
