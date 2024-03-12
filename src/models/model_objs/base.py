from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class BaseModelObject(nn.Module):
    def __init__(self, num_freq: Any):
        super().__init__()
        self.num_freq = num_freq

    def calc_loss(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computes the error.

        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]

            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """

        raise NotImplementedError()

    def forward(self, y: torch.Tensor) -> Any:
        """
        Evaluates f at temporal snapshots y

        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]

            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """
        raise NotImplementedError()

    def get_modes(self, x: torch.Tensor) -> torch.Tensor | tuple:
        """
        Returns neural network approximation of Koopman modes
        :param x: data set
                    type: torch.tensor
                    dimensions: [T, ...]
        :return: torch.tensor
                    dimensions: [T, ..., 2*num_freqs]
        """
        raise NotImplementedError()

    def get_amplitudes(self) -> nn.Parameter | tuple:
        """
        Returns amplitudes corresponding to the Koopman modes
        :return: torch.tensor
                    dimensions: [2*num_freqs, x_dim]
        """
        raise NotImplementedError()