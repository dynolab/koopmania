import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(freq, X, N, omega, amplitudes, model, save=None, show=True):
    plt.stem(freq, np.abs(2 * X / N), "b", markerfmt="bo", label="fft")
    plt.stem(omega, amplitudes, "r", markerfmt="ro", label="fft_algorithm")
    plt.legend()
    plt.xlabel("Freq")
    plt.ylabel("Amplitude")
    plt.xlim(0, 100)
    plt.yscale("log")
    if save is not None:
        plt.savefig(save.dir, f"{model}_pred_spectrum.png")
    if show:
        plt.show()
