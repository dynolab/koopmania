import csv
import os

from omegaconf.dictconfig import DictConfig


def log_scores(cfg: DictConfig, metrics: dict[str, float]):
    log_results = {}
    for metric_name, metric in metrics.items():
        log_results[metric_name] = metric

    if not os.path.exists(cfg.eval_save_path):
        regime = "w"
        write_header = True
    else:
        regime = "a"
        write_header = False
    with open(
        cfg.eval_save_path, mode=regime, newline="", encoding="utf-8"
    ) as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=log_results.keys())
        if write_header:
            dict_object.writeheader()

        dict_object.writerow(log_results)
