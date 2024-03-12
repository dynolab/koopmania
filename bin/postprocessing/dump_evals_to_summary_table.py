from os.path import join as pjoin
from pathlib import Path

import hydra
import pandas as pd
from omegaconf.dictconfig import DictConfig

from src.utils.common import get_config_path

CONFIG_NAME = "evaluations_config"


def dump_evals_to_summary_table(cfg: DictConfig):
    results_root = Path(cfg.hydra_root)
    paths = [results_root / path for path in cfg.evaluation_dirs]
    eval_dfs = []
    for exp_path in paths:
        config_path = exp_path / cfg.config_file
        csv_path = exp_path / "stats.csv"
        eval_df = pd.read_csv(csv_path)
        eval_dfs.append(eval_df)
    total_eval_df = pd.concat(eval_dfs, ignore_index=True)
    total_eval_df.to_csv(pjoin(cfg.hydra_dir, "all_stats.csv"), index=False)

if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.2"
    )(dump_evals_to_summary_table)()
