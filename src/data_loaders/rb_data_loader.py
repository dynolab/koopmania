import os
import numpy as np

from src.data_loaders.base import BaseDataLoader


class RBLoader(BaseDataLoader):
    def __init__(
            self,
            name: str,
            T: int,
            ds: float,
            seed: int,
            x_dim: int,
            sample: int,
            data_path: str
    ):
        super().__init__(name, T, ds)
        self.seed = seed
        self.x_dim = x_dim
        self.sample = sample
        self.data_path = data_path

    def load(self):
        time_series = np.load(self.data_path)
        t = np.arange(0, self.T + self.ds, self.ds)
        if self.sample:
            mask_idxs = np.random.randint(time_series.shape[1], size=self.sample)
            time_series = time_series[:, mask_idxs]

        return t, time_series
