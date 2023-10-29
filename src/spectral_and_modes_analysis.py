import numpy as np
import matplotlib.pyplot as plt

from .fourier import fourier

def check_fourier(time_series, train_ratio, m_freqs, plot=True):
    '''
    Returns spectrum of time series prediction with fft algorithm
    :param time_series: TYPE np.array
    size (N,)
    Input data
    :param train_ratio: TYPE float
    proportion of data included in train set
    :param m_freqs: TYPE int
    number of frequencies used in fourier algorithm
    :param plot: if True, plots two graphs:
    Prediction of fft algorithm alongside with real time series in time domain (left plot)
    Spectrum of a real time series alongside with omegas and amplitudes from fft algorithm

    :return: (omegas: TYPE np.omegas and amplitudes learned with fft algortitm
    Omegas: model.omegas
    Amplitudes are calculated as sqrt(A_1^2 + A_2^2) where A_1 and A_2 are coefficients at cos(wt) and sin(wt) respectively

    '''
    N = len(time_series)
    train_time = int(N * train_ratio)
    model = fourier(m_freqs)
    model.fit(time_series[:train_time].reshape(-1, 1))
    pred = model.predict(N)

    X = np.fft.fft(time_series)
    freq = np.arange(N)
    A_pr = model.A.reshape(2, -1).T
    amplitudes = np.linalg.norm(A_pr, axis=1)
    omega = model.freqs * N
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(time_series, label='true ts')
        ax[0].plot(pred, label='predicted ts')
        ax[0].legend()
        ax[0].set_xlabel('Time')
        ax[1].stem(freq, np.abs(2 * X / N), 'b', markerfmt='bo', label='fft')
        ax[1].stem(omega, amplitudes, 'r', markerfmt='ro', label='fft_algorithm')
        ax[1].legend()
        ax[1].set_xlabel('Freq')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_xlim(0, 100)
        ax[1].set_yscale('log')
        plt.suptitle("m_freqs={}, train_ratio={}".format(m_freqs, train_ratio))
        plt.show()

    return omega, amplitudes


def plot_modes(T, time_series, model, num_modes=6, train_ratio=0.75, iterations=1000, plot_n_last=None):
    '''
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
    :param iterations: TYPE int
    number of iterations in model.fit
    :param plot_n_last: TYPE int
    default None
    if not None, plot only last n steps in prediction
    :return: Graph of real and predicted time series in time domain
            Graph of fft spectrum of real time series
            n graphs of each mode both in time and frequency domains
    '''
    model.fit(time_series[:int(T * train_ratio)].reshape(-1, 1), iterations=iterations)
    pred = model.predict(T)

    if plot_n_last is None:
        plot_n_last = T
    n_lim = max(0, T - plot_n_last)
    t = np.arange(n_lim, T)
    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.plot(t, time_series[n_lim:], label='true_ts')
    plt.plot(t, pred[n_lim:], label=f'{model.__class__.__name__}', linestyle='-.')
    plt.legend()
    plt.xlabel('Time')
    
    plt.title("TS Prediction")
    plt.show()

    plt.plot(t, time_series[n_lim:], label='true_ts')
    model.mode_decomposition(T, num_modes, plot=True, plot_n_last=plot_n_last)

def plot_modes_stochastic(T, time_series, time_series_mu, model, sigma_true=1.0, num_modes=6, train_ratio=0.75, iterations=1000, plot_n_last=None):
    model.fit(time_series[:int(T * train_ratio)].reshape(-1, 1),  weight_decay=1e-7, lr_theta=1e-3, lr_omega=1e-4, iterations=iterations)
    pred = model.predict(T)

    if plot_n_last is None:
        plot_n_last = T
    n_lim = max(0, T - plot_n_last)
    t = np.arange(n_lim, T)
    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.plot(t, time_series[n_lim:], "r", label='ts with noise')
    plt.plot(t, time_series_mu[n_lim:], "b", label='$\\mu$')
    plt.plot(t, pred[0][n_lim:], "--k", label="$\hat \\mu$")
    plt.plot(t, np.ones(t.shape) * sigma_true, "orange", label='$\\sigma$')
    plt.plot(t, pred[1][n_lim:], ":k", label="$\hat \\sigma$")
    plt.legend()
    plt.xlabel('Time')
    plt.show()

    # plt.plot(t, time_series[n_lim:], label='true_ts')
    model.mode_decomposition(T, num_modes, plot=True, plot_n_last=plot_n_last)