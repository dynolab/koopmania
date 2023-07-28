import numpy as np

from src.spectral_and_modes_analysis import plot_modes
from src.koopman import koopman, coordinate_koopman, fully_connected_mse, multi_nn_mse
from src.fourier import fourier
from src.quasi_periodic_data import gen_sine, gen_sawtooth, gen_squares, gen_unit_impulse

T = 1000
np.random.seed(42)
m_freqs = 10

for ts_generator in [gen_sine, gen_sawtooth, gen_squares, gen_unit_impulse]:
    time_series = ts_generator(T)
    model_f = fourier(num_freqs=m_freqs)
    model_k = koopman(fully_connected_mse(x_dim=1, num_freqs=m_freqs, n=512), device='cpu')
    model_ck = coordinate_koopman(multi_nn_mse(x_dim=1, num_freqs=m_freqs, base_model=fully_connected_mse(x_dim=1, num_freqs=1, n=64)), device='cpu')
    for model in [model_f, model_k, model_ck]:
        plot_modes(T, time_series, model)