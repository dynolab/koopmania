import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

def gen_sine(T, n_sines=3):
    '''
    Generate a linear combination of sine functions with random A, w and phi
    :param T: type Int
    time horizon
    :param n_sines: TYPE int
    number of sines in linear combination
    default 3
    :return: TYPE np.array
    size (T,)

    '''

    t = np.linspace(0, 1, T)
    time_series = np.zeros_like(t)
    for _ in range(n_sines):
        A, w = np.random.uniform(0, 10, size=2)
        phi = np.random.uniform(0, np.pi)
        time_series += A * np.sin(2 * np.pi * w * t + phi)

    return time_series

def gen_sawtooth(T, freq=5):
    '''
    Generate a sawtooth wave with given frequency in Hz
    :param T: type Int
    time horizon
    :param freq: TYPE: int
    frequency, Hz (number of peaks per 1 second)
    default 5
    :return: TYPE: np.array
    size (T,)

    '''

    t = np.linspace(0, 1, T)
    time_series = signal.sawtooth(2 * np.pi * freq * t)
    return time_series

def gen_squares(T, freq=5):
    '''
    Generate a square signal with given frequency in Hz
    :param T: TYPE int
    time horizon
    :param freq: TYPE: int
    frequency, Hz (number of peaks per 1 second)
    default 5
    :return: TYPE: np.array
    size (T,)

    '''

    t = np.linspace(0, 1, T)
    time_series = signal.square(2 * np.pi * freq * t)
    return time_series


def gen_unit_impulse(T, n_repeats=5):
    '''
    Generate unit impulse signal of given size with given number of repeats
    :param T: TYPE int
    length of a signal
    :param n_repeats: TYPE int
    number of peaks
    default 5
    :return: TYPE np.array
    size (T,)

    '''

    n, exc = divmod(T, n_repeats)
    impulse = signal.unit_impulse(n, np.random.randint(exc, n))
    time_series = np.tile(impulse, n_repeats)
    time_series = np.pad(time_series, (0, exc))

    return time_series


def gen_wavelet(wavelet_name, n_repeats=5):
    '''
    Generate wavelet function repeated n times
    :param wavelet_name: TYPE str
    name of a wavelet function from 'pywavelets' library in python
    :param n_repeats: TYPE int
    number of repeats
    default 5
    :return: TYPE np.array
    '''
    """Generate wavelet function repeated n times"""
    wavelet = pywt.DiscreteContinuousWavelet(wavelet_name)
    values = wavelet.wavefun(level=10)
    if len(values) == 3:
        phi, psi, x = values
    elif len(values) == 2:
        psi, x = values
    else:
        phi, psi, _, _, x = values
    time_series = np.tile(psi, n_repeats)


    return time_series

