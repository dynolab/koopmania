import numpy as np
import matplotlib.pyplot as plt

from src.navier_stokes import run_NS
from src.DMD import DMD

def MAPE(x, x_hat):
    assert len(x) == len(x_hat)
    return np.mean(np.abs((x - x_hat) / x))

t, time_series = run_NS()

pred, DM = DMD(t, time_series, r=10)

print("MAPE:", MAPE(time_series, pred))

DM = DM.reshape(1024, 1024, -1)
for i in range(DM.shape[2]):
    plt.imshow(DM[:, :, i].real, cmap='jet')
    plt.colorbar()
    plt.title("Mode {}, real part".format(i))
    plt.show()

    plt.imshow(DM[:, :, i].imag, cmap='jet')
    plt.colorbar()
    plt.title("Mode {}, imaginary part".format(i))
    plt.show()


for i in range(10):
    plt.plot(t, time_series[:,i], label='original ts')
    plt.plot(t, pred[:,i], label='DMD prediction')
    plt.title('x dim = {}'.format(i+1))
    plt.show()