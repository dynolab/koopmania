from abc import abstractmethod

from numpy.typing import NDArray
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def fit(self, x_train: NDArray) -> None:
        pass

    @abstractmethod
    def predict(self, x_test: NDArray, x0: NDArray) -> NDArray:
        pass

    @abstractmethod
    def mode_decomposition(
        self, T: int, x0: NDArray
    ) -> NDArray | tuple | list:
        pass
