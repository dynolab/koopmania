import csv
import os
from os.path import join as pjoin
from typing import Optional

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig


def plot_mode(t, mode, num, dim, param=None, model=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, mode)
    ax.set_title(f"{model} mode {num}, dim {dim}, param {param}")


def plot_ts(
    t: NDArray,
    time_series: list | NDArray,
    reconstr_ts: NDArray,
    pred_ts: NDArray,
    model: str,
    y_window: int,
    ax=None,
):
    if not ax:
        _, ax = plt.subplots()
    ax.plot(
        t,
        time_series,
        label="original ts",
        color="blue",
    )
    ax.plot(t[:-y_window], reconstr_ts, label=model, color="red")
    ax.plot(t[-y_window:], pred_ts, linestyle="--", color="red")
    ax.axvline(x=t[-y_window], color="black", linestyle="--", alpha=0.6)


def log_scores(cfg: DictConfig, metrics: dict[str, float]):
    log_results = {}
    for metric_name, metric in metrics.items():
        log_results[metric_name] = metric

    if not os.path.exists(cfg.eval_save_path):
        regime = "w"
        write_header = True
    else:
        regime = "a"
        write_header = False
    with open(
        cfg.eval_save_path, mode=regime, newline="", encoding="utf-8"
    ) as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=log_results.keys())
        if write_header:
            dict_object.writeheader()

        dict_object.writerow(log_results)
