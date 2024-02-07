import os

import numpy as np
import yaml
import matplotlib.pyplot as plt

from src.models.fourier import fourier
from src.data_loaders.quasi_periodic_data import (
    gen_sine,
    gen_sawtooth,
    gen_squares,
    gen_unit_impulse,
)
from src.models.koopman import koopman
from src.models.model_objs.model_objs import fully_connected_mse


def MAPE(x, x_hat):
    assert len(x) == len(x_hat)
    return np.mean(np.abs((x - x_hat) / x))


path = os.getcwd()
parent_dir = os.path.dirname(path)
with open(f"{parent_dir}/config/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
T = cfg["T"]
train_ratio = 0.75
num_freqs = cfg["m_freqs"]
np.random.seed(42)

functions = {
    "sine": gen_sine,
    "sawtooth": gen_sawtooth,
    "squares": gen_squares,
    "unit impulse": gen_unit_impulse,
}


time_series = functions[cfg["function"]](T) + 2
plt.plot(time_series)
plt.show()

# model_k = coordinate_koopman.yaml(multi_nn_mse(1, num_freqs, fully_connected_mse(x_dim=1, num_freqs=1, n=64)), num_freqs,
#                              device='cpu')
model_k = koopman(
    fully_connected_mse(x_dim=1, num_freqs=num_freqs, n=512), 5, device="cpu"
)
#     print(model_k.num_freq, len(model_k.networks))
model_k.fit(time_series[: int(T * train_ratio)].reshape(-1, 1), iterations=1000)
pred_k = model_k.predict(T)

model_f = fourier(num_freqs=num_freqs)
model_f.fit(time_series[: int(T * train_ratio)].reshape(-1, 1))
pred_f = model_f.predict(T)

print(
    "MAPE Fourier:",
    MAPE(time_series[int(T * train_ratio) :], pred_f.flatten()[int(T * train_ratio) :]),
)
print(
    "MAPE Koopman:",
    MAPE(time_series[int(T * train_ratio) :], pred_k.flatten()[int(T * train_ratio) :]),
)
plt.plot(time_series, label="true signal")
plt.plot(pred_f, label="fourier", linestyle="-.")
plt.plot(pred_k, label="koopman", linestyle="-.")
plt.legend()
plt.xlabel("Time")
plt.ylabel("x")
plt.show()
