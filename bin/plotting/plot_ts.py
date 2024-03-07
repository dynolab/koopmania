from os.path import join as pjoin
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig

from src.utils.results import plot_ts


def plot_forecast(
    t: NDArray,
    time_series: NDArray,
    reconstr_ts: NDArray,
    pred_ts: NDArray,
    model: str,
    y_window: int,
    metric: NDArray,
    metric_name: str,
    save: Optional[DictConfig],
    show: Optional,
    n_plots: Optional[int],
) -> None:
    if n_plots is not None:
        sample_idx = np.random.randint(0, pred_ts.shape[1], n_plots)
        iterator = sample_idx
    else:
        iterator = range(pred_ts.shape[1])
    for i in iterator:
        plot_ts(t, time_series[:, i], reconstr_ts[:, i], pred_ts[:, i], model, y_window)
        plt.title(f"example {i+1}, {metric_name}={metric:.4f}")
        plt.legend()
        if save is not None:
            plt.savefig(pjoin(save.dir, f"{model}_{i}.png"))
        if show:
            plt.show()
