import os

import numpy as np
import yaml

from src.postprocessing.spectral_and_modes_analysis import plot_modes
from src.models.koopman import (
    koopman,
    coordinate_koopman,
    fully_connected_mse,
    multi_nn_mse,
)
from src.models.fourier import fourier
from src.data_loaders.quasi_periodic_data import (
    gen_sine,
    gen_sawtooth,
    gen_squares,
    gen_unit_impulse,
)


np.random.seed(42)

path = os.getcwd()
parent_dir = os.path.dirname(path)
with open(f"{parent_dir}/config/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

T = cfg["T"]
m_freqs = cfg["m_freqs"]

functions = {
    "sine": gen_sine,
    "sawtooth": gen_sawtooth,
    "squares": gen_squares,
    "unit impulse": gen_unit_impulse,
}
method = cfg["model"]

time_series = functions[cfg["function"]](T)
print(cfg["function"])
if method == "fourier":
    model = fourier(num_freqs=m_freqs)
elif method == "koopman":
    model = koopman(
        fully_connected_mse(x_dim=1, num_freqs=m_freqs, n=512), device="cpu"
    )
elif method == "coordinate_koopman":
    model = coordinate_koopman(
        multi_nn_mse(
            x_dim=1,
            num_freqs=m_freqs,
            base_model=fully_connected_mse(x_dim=1, num_freqs=1, n=64),
        ),
        device="cpu",
    )

plot_modes(T, time_series, model, plot_n_last=10)
