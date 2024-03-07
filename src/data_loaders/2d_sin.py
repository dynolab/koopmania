import os
import numpy as np
import matplotlib.pyplot as plt

from bin.plotting.render_env import render_env
from src.data_loaders.base import BaseDataLoader

class TwoDimSin(BaseDataLoader):
    def __init__(self, name: str, T: int, ds: float, save_dir: str):
        super().__init__(name, T, ds)
        self.save_dir = save_dir
    def load(self):

@np.vectorize
def f(x, y):
    return np.sin(x) * np.sin(y)


x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1 / 2, 500)
t = np.arange(0, 200, 0.1)
X, Y = np.meshgrid(x, y)
ts = []
for i in range(10000):
    if i % 10 == 0:
        print(i)
    out = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.sin(t[i])
    ts.append(out.flatten())
    plt.pause(2)
    plt.imshow(out, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"t={np.round(t[i], 1)}")
    plt.savefig(os.path.join(directory, f"plot_{i}.png"))
    plt.close()
ts = np.vstack(ts)
print(ts.shape)
with open("blobs_long.npy", "wb") as f:
    np.save(f, ts)
render_env(
    1000,
    directory,
    r"C:\Users\mWX1298408\Documents\koopmania\plots",
    "blob_example_long",
)
time_series = np.load("blobs.npy")
