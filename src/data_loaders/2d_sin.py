import os

import matplotlib.pyplot as plt
import numpy as np

from bin.plotting.render_env import render_env
from src.data_loaders.base import BaseDataLoader


class TwoDimSin(BaseDataLoader):
    def __init__(self, name: str, T: int, ds: float, seed: int, sample: int, x_dim: int, save_dir: str, data_path: str,
                 render: str, render_name: str):
        super().__init__(name, T, ds)
        self.save_dir = save_dir
        self.data_path = data_path
        self.render = render
        self.render_name = render_name
        self.sample = sample
        self.x_dim = x_dim

    def load(self):
        if not os.path.exists(self.data_path):
            self.gen_sin()

        t = np.arange(0, self.T + self.ds, self.ds)
        time_series = np.load(self.data_path)
        if self.sample:
            mask_idxs = np.random.randint(time_series.shape[1], size=self.sample)
            time_series = time_series[:, mask_idxs]
        return t, time_series

    def gen_sin(self):
        t = np.arange(0, self.T + self.ds, self.ds)

        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1 / 2, 500)
        t = np.arange(0, self.T, self.ds)
        X, Y = np.meshgrid(x, y)
        ts = []
        for i in range(len(t)):
            out = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.sin(t[i])
            ts.append(out.flatten())
            plt.pause(2)
            plt.imshow(out, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f"t={np.round(t[i], 1)}")
            plt.savefig(os.path.join(self.render, f"plot_{i}.png"))
            plt.close()
        ts = np.vstack(ts)

        with open(self.data_path, "wb") as f:
            np.save(f, ts)
        render_env(
            self.T,
            self.ds,
            self.render,
            self.save_dir,
            self.render_name,
        )


@np.vectorize
def f(x, y):
    return np.sin(x) * np.sin(y)
