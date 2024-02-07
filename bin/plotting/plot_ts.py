import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Optional

from omegaconf.dictconfig import DictConfig


def plot_forecast(
    t: NDArray,
    time_series: NDArray,
    reconstr_ts: list[NDArray],
    pred_ts: list[NDArray],
    model: str,
    y_window: int,
    metric: list[NDArray],
    metric_name: str,
    save: Optional[DictConfig],
    show: Optional,
    sample: Optional[int],
):
    if sample is not None:
        sample_idx = np.random.randint(0, pred_ts.shape[1], sample)
        iterator = sample_idx
    else:
        iterator = range(len(pred_ts))
    for i in iterator:
        plt.plot(t, time_series[:, i], label="original ts", color="blue")
        plt.plot(t[:-y_window], reconstr_ts[:, i], label=model, color="red")
        plt.plot(t[-y_window:], pred_ts[:, i], linestyle="--", color="red")
        plt.axvline(x=t[-y_window], color="black", linestyle="--")
        plt.title(f"example {i+1}, {metric_name}={metric[i]:.4f}")
        plt.legend()
        if save is not None:
            plt.savefig(save.dir, f"{model}_{i}.png")
        if show:
            plt.show()
