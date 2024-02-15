from pathlib import Path

import numpy as np


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    return get_project_path() / "config"


def transform_time(t):
    t0 = int(t[0] / (t[1] - t[0]))

    t = np.arange(t0, t0 + len(t))
    return t
