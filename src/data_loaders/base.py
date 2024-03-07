from abc import ABC, abstractmethod

from numpy.typing import NDArray


class BaseDataLoader(ABC):
    def __init__(self, name: str, T: int, ds: float):
        self.name = name
        self.T = T
        self.ds = ds

    @abstractmethod
    def load(self) -> tuple[NDArray, NDArray]:
        pass
