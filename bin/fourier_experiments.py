import os

import numpy as np
import yaml

from src.data_loaders.quasi_periodic_data import (gen_sawtooth, gen_sine,
                                                  gen_squares,
                                                  gen_unit_impulse)
from src.postprocessing.spectral_and_modes_analysis import check_fourier

path = os.getcwd()
parent_dir = os.path.dirname(path)
with open(f"{parent_dir}/config/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
T = cfg["T"]
np.random.seed(42)

functions = {
    "sine": gen_sine,
    "sawtooth": gen_sawtooth,
    "squares": gen_squares,
    "unit impulse": gen_unit_impulse,
}


time_series = functions[cfg["function"]](T)
for ratio in [0.5, 0.75, 1.0]:
    for m_freqs in [1, 3, 5, 10, 50, 75]:
        check_fourier(time_series, ratio, m_freqs, plot=True)
