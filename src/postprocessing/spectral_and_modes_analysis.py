import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.utils.results import plot_mode


def plot_modes(
    t: NDArray,
    model,
    x_0: NDArray,
    num_modes: int = 6,
    num_dims: int = 1,
    ax=None,
    show: bool = False,
    stochastic: bool = False,
) -> None:
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
    t = np.arange(len(t))
    modes = model.mode_decomposition(
        len(t), x_0
    )
    if not stochastic:
        modes = [modes]
    modes = [mode[:, :num_dims, :num_modes] for mode in modes]
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
