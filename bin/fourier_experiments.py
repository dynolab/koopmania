import numpy as np

from src.spectral_and_modes_analysis import check_fourier
from src.quasi_periodic_data import gen_sine, gen_sawtooth, gen_squares, gen_unit_impulse

T = 1000
np.random.seed(42)

for ts_generator in [gen_sine, gen_sawtooth, gen_squares, gen_unit_impulse]:
    time_series = ts_generator(T)
    for ratio in [0.5, 0.75, 1.0]:
        for m_freqs in [1, 3, 5, 10, 50, 75]:
            check_fourier(time_series, ratio, m_freqs, plot=True)

