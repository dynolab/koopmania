import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import get_object
from numpy.typing import NDArray
from scipy.integrate import odeint

from bin.plotting.render_env import render_env
from src.data_loaders.base import BaseDataLoader


@dataclass(kw_only=True)
class DomainParams:
    nu: float
    Lx: float
    Ly: float
    Nx: int
    Ny: int


@dataclass(kw_only=True)
class SaveParams:
    dir: str
    data_path: str
    render: str
    render_name: str


def init_cond(Nx, Ny) -> NDArray:
    Om_hat = np.zeros((Nx, Ny), dtype=np.cdouble)

    # We take a random distribution of the vorticity
    Om_hat[0, 4] = np.random.randn() + 1j * np.random.randn()
    Om_hat[1, 4] = np.random.randn() + 1j * np.random.randn()
    Om_hat[3, 3] = np.random.randn() + 1j * np.random.randn()
    Om_hat[3, 0] = np.random.randn() + 1j * np.random.randn()

    Om = np.real(np.fft.ifft2(Om_hat))
    Om /= np.max(np.abs(Om))  # normalize to O(1)

    return Om


class NSLoader(BaseDataLoader):
    def __init__(
        self,
        name: str,
        T: int,
        ds: float,
        seed: int,
        domain_params: DomainParams,
        x_dim: int,
        sample: int,
        save_params: SaveParams,
        init_cond,
        plot: bool = False,
    ):
        super().__init__(name, T, ds)
        self.seed = seed
        self.nu = domain_params.nu
        self.Lx = get_object(domain_params.Lx)
        self.Ly = get_object(domain_params.Ly)
        self.Nx = domain_params.Nx
        self.Ny = domain_params.Ny
        self.plot = plot
        self.dir = save_params.dir
        self.data_path = save_params.data_path
        self.render = save_params.render
        self.render_name = save_params.render_name

        self.x_dim = x_dim
        self.sample = sample

        self.init_cond = init_cond

    def load(self) -> tuple[NDArray]:
        if not os.path.exists(self.data_path):
            self.run_NS()

        time_series = np.load(self.data_path)
        t = np.arange(0, self.T + self.ds, self.ds)
        if self.sample:
            mask_idxs = np.random.randint(time_series.shape[1], size=self.sample)
            time_series = time_series[:, mask_idxs]

        return t, time_series

    def run_NS(self) -> None:
        # # Clear figures and set number format
        print("start NS")
        plt.close("all")
        np.set_printoptions(formatter={"float_kind": "{:.6e}".format})

        dx = 2 * self.Lx / self.Nx  # distance between two physical points
        x = (
            np.arange(1 - self.Nx / 2, self.Nx / 2 + 1) * dx
        )  # physical space discretization

        self.dy = 2 * self.Ly / self.Ny  # distance between two physical points
        y = (
            np.arange(1 - self.Ny / 2, self.Ny / 2 + 1) * self.dy
        )  # physical space discretization

        self.X, self.Y = np.meshgrid(x, y)  # 2D composed grid

        # Vectors of wavenumbers in the transformed space
        kx = (
            np.concatenate(
                (np.arange(0, self.Nx / 2 + 1), np.arange(-self.Nx / 2 + 1, 0))
            )
            * np.pi
            / self.Lx
        )
        ky = (
            np.concatenate(
                (np.arange(0, self.Ny / 2 + 1), np.arange(-self.Ny / 2 + 1, 0))
            )
            * np.pi
            / self.Ly
        )

        # Antialiasing treatment
        jx = np.arange(
            self.Nx // 4 + 1, self.Nx // 4 * 3 + 1
        )  # frequencies we sacrifice
        kx[jx] = 0

        jy = np.arange(
            self.Ny // 4 + 1, self.Ny // 4 * 3 + 1
        )  # frequencies we sacrifice
        ky[jy] = 0

        # Some operators arising in NS equations
        self.Kx, self.Ky = np.meshgrid(kx, ky)
        self.K2 = self.Kx**2 + self.Ky**2  # to compute the Laplace operator

        K2inv = np.zeros_like(self.K2)
        K2inv[self.K2 != 0] = 1.0 / self.K2[self.K2 != 0]

        self.Dx = (
            1j * self.Kx * K2inv
        )  # u velocity component reconstruction from the vorticity
        self.Dy = (
            1j * self.Ky * K2inv
        )  # v velocity component reconstruction from the vorticity

        # Set random number generator (for the initial condition)
        np.random.seed(self.seed)

        # Initialize the vorticity
        Om = self.init_cond  # InitCond()

        # Plot initial condition
        if self.plot:
            self._plot(Om, 0.0)

        # Time-stepping parameters
        t = 0.0  # the discrete time variable
        Tf = self.T  # final simulation time
        ds = self.ds  # write time of the results

        # Time-stepping loop
        time_series = []
        t_l = []
        i = 0
        while t < Tf:
            # Solve using ODE solver
            v = odeint(self.RHS, Om.flatten(), [t, t + ds])
            Om = v[-1].reshape(self.Nx, self.Ny)
            t_l.append(t)
            t += ds
            i += 1
            time_series.append(Om.flatten())
            # Plot the solution
            # if self.plot:
            if i % 40 == 0:
                self._plot(Om, t)

        # render
        if self.render:
            render_env(self.T, self.render, self.dir, self.render_name)
        # # save

        with open(self.data_path, "wb") as f:
            np.save(f, np.vstack(time_series))

    def _plot(self, Om: NDArray, t: float) -> None:
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(self.X, self.Y, Om, shading="auto", cmap="jet")
        plt.colorbar(label=r"$\omega(x,y,t)$")
        plt.xlim([-self.Lx, self.Lx])
        plt.ylim([-self.Ly + self.dy, self.Ly])
        plt.xlabel("$x$", fontsize=12)
        plt.ylabel("$y$", fontsize=12, rotation=1)
        plt.title(f"Vorticity distribution at t = {t:.2f}", fontsize=12)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(rf"{self.render}\plot_{round(t, 1)}".replace(".", ",") + ".png")
        plt.show()
        plt.close()
        plt.pause(0.001)

    def RHS(self, Om_vec: NDArray, t: float) -> NDArray:
        Om = Om_vec.reshape((self.Nx, self.Ny))
        Om_hat = np.fft.fft2(Om)
        Omx = np.real(np.fft.ifft2(1j * self.Kx * Om_hat))
        Omy = np.real(np.fft.ifft2(1j * self.Ky * Om_hat))

        u = np.real(np.fft.ifft2(self.Dy * Om_hat))
        v = np.real(np.fft.ifft2(-self.Dx * Om_hat))

        rhs = np.real(np.fft.ifft2(-self.nu * self.K2 * Om_hat)) - u * Omx - v * Omy
        return rhs.flatten()
