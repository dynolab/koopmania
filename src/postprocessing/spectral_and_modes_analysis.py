from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig

from bin.plotting.plot_ts import plot_forecast
from src.utils.results import plot_mode


def plot_modes(
    t: NDArray,
    model,
    x_0,
    num_modes=6,
    num_dims=1,
    plot_n_last=None,
    ax=None,
    show=False,
    stochastic=False,
):
    """
    Plot modes obtained by mode decomposition of given model in time and frequency domains
    :param T: TYPE int
    time horizon
    :param time_series: TYPE np.array (T,)
    Input data
    :param model: Instance of prediction algorithm (fourier or koopman)
    :param num_modes: TYPE int
    number of modes in mode decomposition
    :param train_ratio: TYPE float
    proportion of data included in train set
     model.fit
    :param plot_n_last: TYPE int
    default None
    if not None, plot only last n steps in prediction
    :return: Graph of real and predicted time series in time domain
            Graph of fft spectrum of real time series
            n graphs of each mode both in time and frequency domains
    """
    if plot_n_last is None:
        plot_n_last = len(t)
    n_lim = max(0, len(t) - plot_n_last)
    t = np.arange(n_lim, len(t))
    modes = model.mode_decomposition(
        len(t), num_modes, x_0, num_dims, plot=False, plot_n_last=plot_n_last
    )
    if not stochastic:
        modes = [modes]
    if ax is None:
        fig, ax = plt.subplots(
            num_dims * len(modes), num_modes, figsize=(15, 10), sharex=True
        )
    plt.suptitle(f"{model.name} Modes")
    for k in range(len(modes)):
        for j in range(num_dims):
            for i in range(num_modes):
                modes_k = modes[k]
                plot_mode(
                    t,
                    modes_k[:, j, i],
                    i,
                    j,
                    k,
                    model=model.name,
                    ax=ax[j + k * num_dims][i],
                )
    if show:
        plt.show()


# def plot_modes_stochastic(
#     T,
#     time_series,
#     time_series_mu,
#     model,
#     sigma_true=1.0,
#     num_modes=6,
#     train_ratio=0.75,
#     iterations=1000,
#     plot_n_last=None,
# ):
#     model.fit(
#         time_series[: int(T * train_ratio)].reshape(-1, 1),
#         weight_decay=1e-7,
#         lr_theta=1e-3,
#         lr_omega=1e-4,
#         iterations=iterations,
#     )
#     pred = model.predict(T)
#
#     if plot_n_last is None:
#         plot_n_last = T
#     n_lim = max(0, T - plot_n_last)
#     t = np.arange(n_lim, T)
#     # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     plt.plot(t, time_series[n_lim:], "r", label="ts with noise")
#     plt.plot(t, time_series_mu[n_lim:], "b", label="$\\mu$")
#     plt.plot(t, pred[0][n_lim:], "--k", label="$\hat \\mu$")
#     plt.plot(t, np.ones(t.shape) * sigma_true, "orange", label="$\\sigma$")
#     plt.plot(t, pred[1][n_lim:], ":k", label="$\hat \\sigma$")
#     plt.legend()
#     plt.xlabel("Time")
#     plt.show()
#
#     # plt.plot(t, time_series[n_lim:], label='true_ts')
#     model.mode_decomposition(T, num_modes, plot=True, plot_n_last=plot_n_last)
