#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henning Lange (helange@uw.edu)
"""

import numpy as np
import torch
from numpy.typing import NDArray

from src.models.base import BaseModel


class Fourier(BaseModel):
    """

    num_freqs: number of frequencies assumed to be present in data
        type: int

    device: The device on which the computations are carried out.
        Example: cpu, cuda:0
        default = 'cpu'

    learning_rate: TYPE float, optional
    The default is 1E-5.

    iterations: TYPE int, optional
    The default is 1000.

    verbose: TYPE optional
    The default is False.

    """
    def __init__(
        self,
        name: str,
        num_freqs: int,
        device: str = "cpu",
        learning_rate: float = 1e-5,
        iterations: int = 1000,
        verbose: bool = False,
    ):
        super().__init__(name)
        self.num_freqs = num_freqs
        self.device = device
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose

    def fft(self, xt: NDArray) -> None:
        """
        Given temporal data xt, fft performs the initial guess of the
        frequencies contained in the data using the FFT.

        Parameters
        ----------
        xt : TYPE: numpy.array
            Temporal data of dimensions [T, ...]

        Returns
        -------
        None.

        """

        k = self.num_freqs
        self.freqs = []

        for i in range(k):
            N = len(xt)

            if len(self.freqs) == 0:
                residual = xt
            else:
                t = np.expand_dims(np.arange(N) + 1, -1)
                freqs = np.array(self.freqs)
                Omega = np.concatenate(
                    [np.cos(t * 2 * np.pi * freqs), np.sin(t * 2 * np.pi * freqs)], -1
                )
                self.A = np.dot(np.linalg.pinv(Omega), xt)

                pred = np.dot(Omega, self.A)

                residual = pred - xt

            ffts = 0
            for j in range(xt.shape[1]):
                ffts += np.abs(np.fft.fft(residual[:, j])[: N // 2])

            w = np.fft.fftfreq(N, 1)[: N // 2]
            idxs = np.argmax(ffts)

            self.freqs.append(w[idxs])

            t = np.expand_dims(np.arange(N) + 1, -1)

            Omega = np.concatenate(
                [
                    np.cos(t * 2 * np.pi * self.freqs),
                    np.sin(t * 2 * np.pi * self.freqs),
                ],
                -1,
            )

            self.A = np.dot(np.linalg.pinv(Omega), xt)

    def sgd(
        self,
        xt: NDArray,
        learning_rate: float = 3e-9
    ) -> None:
        """
        Given temporal data xt, sgd improves the initial guess of omega
        by SGD. It uses the pseudo-inverse to obtain A.

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data of dimensions [T, ...]

        learning_rate : TYPE float, optional
            Note that the learning rate should decrease with T. The default is 3E-9.

        Returns
        -------
        None.

        """

        A = torch.tensor(self.A, requires_grad=False, device=self.device)
        freqs = torch.tensor(self.freqs, requires_grad=True, device=self.device)
        xt = torch.tensor(xt, requires_grad=False, device=self.device)

        o2 = torch.optim.SGD([freqs], lr=learning_rate)

        t = torch.unsqueeze(
            torch.arange(len(xt), dtype=torch.get_default_dtype(), device=self.device)
            + 1,
            -1,
        )


        for i in range(self.iterations):
            Omega = torch.cat(
                [torch.cos(t * 2 * np.pi * freqs), torch.sin(t * 2 * np.pi * freqs)], -1
            )

            A = torch.matmul(torch.pinverse(Omega.data), xt)

            xhat = torch.matmul(Omega, A)
            loss = torch.mean((xhat - xt) ** 2)

            o2.zero_grad()
            loss.backward()
            o2.step()

            loss = loss.cpu().detach().numpy()
            if self.verbose:
                print(f"Loss at step {i}:", loss)

        self.A = A.cpu().detach().numpy()
        self.freqs = freqs.cpu().detach().numpy()

    def fit(self, xt: NDArray):
        """

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data of dimensions [T, ...]

        Returns
        -------
        None.

        """

        self.fft(xt)
        self.sgd(
            xt,
            learning_rate=self.learning_rate / xt.shape[0],
        )

    def predict(self, t: NDArray, x0: NDArray) -> NDArray:
        """
        Predicts the data from 1 to T.

        Parameters
        ----------
        t : TYPE NDArray
            Prediction time

        x0: TYPE NDArray
        Starting point (not used)

        Returns
        -------
        TYPE numpy.array
            xhat from 0 to len(t).

        """
        t = np.expand_dims(t, -1)
        Omega = np.concatenate(
            [np.cos(t * 2 * np.pi * self.freqs), np.sin(t * 2 * np.pi * self.freqs)], -1
        )

        return np.dot(Omega, self.A)

    def mode_decomposition(
        self,
        T: int,
        x0: NDArray
    ) -> NDArray:
        """
        Returns first n modes of prediction built by Fourier algorithm
        :param T: TYPE int
        prediction horizon

        :param x0: TYPE NDArray
        starting point (not used)

        """
        t = np.arange(T)
        arg = t[:, None] * self.freqs[None, :]
        freqs = np.concatenate(
            [np.cos(2 * np.pi * arg * 1000), np.sin(2 * np.pi * arg * 1000)], axis=-1
        )
        modes = []
        for j in range(x0.shape[0]):
            modes_dim = []
            for i in range(2 * self.num_freqs):
                mode = freqs[:, i]
                modes_dim.append(mode)
            modes_dim = np.stack(modes_dim, axis=-1)
            modes.append(modes_dim)
        modes = np.stack(modes, axis=-2)
        return modes[0]
