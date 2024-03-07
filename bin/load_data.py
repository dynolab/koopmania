import hydra
import numpy as np
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig

from src.data_loaders.base import BaseDataLoader
from src.utils.common import get_config_path

CONFIG_NAME = "config"


def load_data(cfg: DictConfig) -> tuple[NDArray, NDArray]:
    np.random.seed(cfg.seed)
    loader: BaseDataLoader = instantiate(cfg.dataset)
    t, time_series = loader.load()
    return t, time_series


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.2",
    )(load_data())()
