# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
#
# def initCond(Nx, Ny):
#     Om_hat = np.zeros((Nx, Ny), dtype=np.cdouble)
#
#     Om_hat[0, 4] = np.random.randn() + 1j * np.random.randn()
#     Om_hat[1, 1] = np.random.randn() + 1j * np.random.randn()
#     Om_hat[3, 0] = np.random.randn() + 1j * np.random.randn()
#
#     Om = np.real(np.fft.ifft2(Om_hat))
#     Om = Om / np.max(Om)
#
#     return Om
#
# def RHS(t, Om_vec):
#     global nu, Nx, Ny, Kx, Ky, K2, Dx, Dy
#     # print(Om_vec, 'rhs')
#     Om = Om_vec.reshape(Nx, Ny)
#     Om_hat = np.fft.fft2(Om)
#
#     Omx = np.real(np.fft.ifft2(1j * Kx * Om_hat))
#     Omy = np.real(np.fft.ifft2(1j * Ky * Om_hat))
#
#     u = np.real(np.fft.ifft2(Dy * Om_hat))
#     v = np.real(np.fft.ifft2(-Dx * Om_hat))
#
#     rhs = np.real(np.fft.ifft2(-nu * K2 * Om_hat )) - u * Omx - v * Omy
#     rhs = rhs.reshape(Nx * Ny)
#     return rhs
#
# def plot_cur(Om):
#     plt.cla()
#     plt.imshow(Om, cmap='rainbow')
#     plt.clim(-20, 20)
#     ax = plt.gca()
#     ax.invert_yaxis()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.set_aspect('equal')
#     plt.pause(0.001)
#
# def run_NS(Lx, Ly, Nx, Ny, Nxy, random_seed=1):
#     global nu, Kx, Ky, K2, Dx, Dy
#
#     dx = 2 * Lx / Nx
#     x = np.arange(- Nx/2, Nx/2 + 1) * dx
#
#     dy = 2 * Ly / Ny
#     y = np.arange(- Ny/2, Ny/2 + 1) * dy
#
#     X, Y = np.meshgrid(x, y)
#
#     np.random.seed(random_seed)
#
#     t = 0.0
#     Tf = 5.0
#     ds = 0.1
#
#     Om = initCond(Nx, Ny)
#     plot_cur(Om)
#     Om_vec = Om.reshape(Nxy)
#     # print(Om_vec.shape)
#     while t < Tf:
#         # print(Om_vec, 'rhs 0')
#         v = odeint(RHS, Om_vec, np.arange(t, t+ds, ds), tfirst=True)
#
#         Om_vec = v[-1]
#         # print(Om_vec.shape)
#         Om = Om_vec.reshape(Nx, Ny)
#         print(Om.mean())
#         plot_cur(Om)
#         t += ds
#
#
#
# nu = 1e-4
# Lx = np.pi
# Ly = np.pi
# Nx = 1024
# Ny = 1024
# Nxy = Nx * Ny
#
# kx = np.concatenate([np.arange(Nx / 2 ), np.arange(- Nx/2, 0)]).reshape(-1,1) *np.pi/Lx
# ky = np.concatenate([np.arange(Ny / 2 ), np.arange(- Ny/2, 0)]).reshape(-1,1)  *np.pi / Ly
#
# jx = np.arange(Nx//4 + 2, Nx//4 * 3)
# kx[jx] = 0
#
# jy = np.arange(Ny//4 + 2, Ny//4 * 3)
# ky[jy] = 0
#
# Kx, Ky = np.meshgrid(kx, ky)
#
# K2 = Kx ** 2 + Ky ** 2
#
# K2inv = np.zeros_like(K2)
# K2inv[K2 != 0] = 1 / K2[K2 != 0]
#
# Dx = 1j * Kx * K2inv
# Dy = 1j * Ky * K2inv
#
# run_NS(Lx, Ly, Nx, Ny, Nxy)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# declaration of global variables
dy = None
nu = None
Dx = None
Dy = None
Kx = None
Ky = None
K2 = None
Lx = None
Ly = None
Nx = None
Ny = None
Nxy = None
X = None
Y = None


def InitCond():
    global Nx, Ny

    Om_hat = np.zeros((Nx, Ny), dtype=np.cdouble)

    # We take a random distribution of the vorticity
    Om_hat[0, 4] = np.random.randn() + 1j * np.random.randn()
    Om_hat[1, 1] = np.random.randn() + 1j * np.random.randn()
    Om_hat[3, 0] = np.random.randn() + 1j * np.random.randn()

    Om = np.real(np.fft.ifft2(Om_hat))
    Om /= np.max(np.abs(Om))  # normalize to O(1)

    return Om


def Plot(Om, t):
    global dy, Lx, Ly, X, Y
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, Om, shading='auto', cmap='jet')
    plt.colorbar(label=r'$\omega(x,y,t)$')
    plt.xlim([-Lx, Lx])
    plt.ylim([-Ly + dy, Ly])
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$y$', fontsize=12, rotation=1)
    plt.title(f'Vorticity distribution at t = {t:.2f}', fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.pause(0.001)
    plt.show()


def RHS(Om_vec, t):
    global nu, Dx, Dy, Kx, Ky, K2

    Om = Om_vec.reshape((Nx, Ny))
    Om_hat = np.fft.fft2(Om)
    Omx = np.real(np.fft.ifft2(1j * Kx * Om_hat))
    Omy = np.real(np.fft.ifft2(1j * Ky * Om_hat))

    u = np.real(np.fft.ifft2(Dy * Om_hat))
    v = np.real(np.fft.ifft2(-Dx * Om_hat))

    rhs = np.real(np.fft.ifft2(-nu * K2 * Om_hat)) - u * Omx - v * Omy
    return rhs.flatten()


nu = 1e-4  # 1/Re or viscosity

# Domain definition
Lx = np.pi  # Domain half-length in x-direction
Ly = np.pi  # Domain half-length in y-direction

# Numerical parameters
Nx = 1024  # number of Fourier modes in discrete solution x-dir
Ny = 1024  # number of Fourier modes in discrete solution y-dir
Nxy = Nx * Ny
def run_NS():
    # Clear figures and set number format
    global X, Y, Dx, Dy, dx, dy, Kx, Ky, K2
    plt.close('all')
    np.set_printoptions(formatter={'float_kind': '{:.6e}'.format})

    dx = 2 * Lx / Nx  # distance between two physical points
    x = np.arange(1 - Nx / 2, Nx / 2 + 1) * dx  # physical space discretization

    dy = 2 * Ly / Ny  # distance between two physical points
    y = np.arange(1 - Ny / 2, Ny / 2 + 1) * dy  # physical space discretization

    X, Y = np.meshgrid(x, y)  # 2D composed grid

    # Vectors of wavenumbers in the transformed space
    kx = np.concatenate((np.arange(0, Nx / 2 + 1), np.arange(-Nx / 2 + 1, 0))) * np.pi / Lx
    ky = np.concatenate((np.arange(0, Ny / 2 + 1), np.arange(-Ny / 2 + 1, 0))) * np.pi / Ly

    # Antialiasing treatment
    jx = np.arange(Nx // 4 + 1, Nx // 4 * 3 + 1)  # frequencies we sacrifice
    kx[jx] = 0

    jy = np.arange(Ny // 4 + 1, Ny // 4 * 3 + 1)  # frequencies we sacrifice
    ky[jy] = 0

    # Some operators arising in NS equations
    Kx, Ky = np.meshgrid(kx, ky)
    K2 = Kx ** 2 + Ky ** 2  # to compute the Laplace operator

    K2inv = np.zeros_like(K2)
    K2inv[K2 != 0] = 1.0 / K2[K2 != 0]

    Dx = 1j * Kx * K2inv  # u velocity component reconstruction from the vorticity
    Dy = 1j * Ky * K2inv  # v velocity component reconstruction from the vorticity

    # Set random number generator (for the initial condition)
    np.random.seed(1)

    # Initialize the vorticity
    Om = InitCond()

    # Plot initial condition
    Plot(Om, 0.0)

    # Time-stepping parameters
    t = 0.0  # the discrete time variable
    Tf = 10.0  # final simulation time
    ds = 0.1 # write time of the results

    # Time-stepping loop
    mask_x = np.random.randint(0, Nx, size=100)
    mask_y = np.random.randint(0, Ny, size=100)
    time_series = []
    t_l = []
    while t < Tf:
        # Solve using ODE solver
        v = odeint(RHS, Om.flatten(), [t, t + ds])
        Om = v[-1, :].reshape(Nx, Ny)
        t_l.append(t)
        t += ds
        time_series.append(Om.flatten()) #[mask_x, mask_y])
        # Plot the solution
        Plot(Om, t)

    return np.array(t_l), np.vstack(time_series)




