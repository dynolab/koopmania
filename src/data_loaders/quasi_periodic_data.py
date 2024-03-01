import numpy as np
import pywt
from scipy import signal


def gen_sine(T, ds, n_sines=3):
    """
    Generate a linear combination of sine functions with random A, w and phi
    :param T: type Int
    time horizon
    :param n_sines: TYPE int
    number of sines in linear combination
    default 3
    :return: TYPE np.array
    size (T,)

    """

    t = np.arange(0, T, ds)
    time_series = np.zeros_like(t)
    for _ in range(n_sines):
        A, w = np.random.uniform(0, 10, size=2)
        phi = np.random.uniform(0, np.pi)
        time_series += A * np.sin(2 * np.pi * w * t + phi)
    return t, time_series.reshape(-1, 1)


def gen_sawtooth(T, ds, freq=5):
    """
    Generate a sawtooth wave with given frequency in Hz
    :param T: type Int
    time horizon
    :param freq: TYPE: int
    frequency, Hz (number of peaks per 1 second)
    default 5
    :return: TYPE: np.array
    size (T,)

    """

    t = np.arange(0, T, ds)
    time_series = signal.sawtooth(2 * np.pi * freq * t)
    return t, time_series.reshape(-1, 1)


def gen_squares(T, ds, freq=5):
    """
    Generate a square signal with given frequency in Hz
    :param T: TYPE int
    time horizon
    :param freq: TYPE: int
    frequency, Hz (number of peaks per 1 second)
    default 5
    :return: TYPE: np.array
    size (T,)

    """

    t = np.arange(0, T, ds)
    time_series = signal.square(2 * np.pi * freq * t)
    return t, time_series.reshape(-1, 1)


def gen_unit_impulse(T, ds, n_repeats=None):
    """
    Generate unit impulse signal of given size with given number of repeats
    :param T: TYPE int
    length of a signal
    :param n_repeats: TYPE int
    number of peaks
    default None
    if None, use T//100
    :return: TYPE np.array
    size (T,)

    """
    T1 = int(T / ds)
    if n_repeats is None:
        n_repeats = T1 // 100
    n, exc = divmod(T1, n_repeats)
    impulse = signal.unit_impulse(n, np.random.randint(exc, n))
    time_series = np.tile(impulse, n_repeats)
    time_series = np.pad(time_series, (0, exc))
    t = np.arange(0, T, ds)
    return t, time_series.reshape(-1, 1)


def gen_wavelet(wavelet_name, n_repeats=5):
    """
    Generate wavelet function repeated n times
    :param wavelet_name: TYPE str
    name of a wavelet function from 'pywavelets' library in python
    :param n_repeats: TYPE int
    number of repeats
    default 5
    :return: TYPE np.array
    """
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


# stochastic signals


def periodic_normal(T, ds, n_freqs=5):
    T1 = int(T / ds)
    mu = np.zeros(T1)
    sigma = np.zeros(T1)
    t = np.arange(0, T, ds).astype(float)
    for i in range(n_freqs):
        A1, w1, A2, w2 = np.random.uniform(0, 10, size=4)
        phi1, phi2 = np.random.uniform(0, np.pi, size=2)
        mu += A1 * np.sin(2 * np.pi * w1 * t + phi1)
        sigma += np.exp(A2 * np.sin(2 * np.pi * w2 * t + phi2))

    time_series = np.random.normal(mu, sigma).reshape(-1, 1)
    return t, time_series


def periodic_gamma(T, ds, n_freqs=5):
    T1 = int(T / ds)
    alpha = np.zeros(T1)
    scale = np.zeros(T1)
    t = np.arange(0, T, ds).astype(float)
    for i in range(n_freqs):
        A1, w1, A2, w2 = np.random.uniform(0, 10, size=4)
        phi1, phi2 = np.random.uniform(0, np.pi, size=2)
        alpha += np.exp(A1 * np.sin(2 * np.pi * w1 * t + phi1))
        scale += np.exp(A2 * np.sin(2 * np.pi * w2 * t + phi2))

    time_series = np.random.gamma(alpha, scale=scale).reshape(-1, 1)
    return t, time_series


def add_gaus_noise(time_series, sigma=1):
    T = time_series.shape[0]
    new_ts = time_series + np.random.normal(0, sigma, size=(T, 1))
    return new_ts


NAME_DICT = {
    "sin": gen_sine,
    "sawtooth": gen_sawtooth,
    "squares": gen_squares,
    "unit_impulse": gen_unit_impulse,
    "periodic_normal": periodic_normal,
    "periodic_gamma": periodic_gamma,
}


class SyntheticGenerator:
    def __init__(self, name, T, ds, x_dim, add_noise=False, sigma=0, **kwargs):
        self.name = name
        self.T = T
        self.ds = ds
        self.add_noise = add_noise
        self.sigma = sigma
        self.kwargs = kwargs

    def load(self):
        f = NAME_DICT[self.name]
        t, time_series = f(self.T, self.ds, **self.kwargs)
        if self.add_noise:
            time_series = add_gaus_noise(time_series, self.sigma)

        return t, time_series
