import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit

from bin.load_data import load_data
from bin.plotting.plot_ts import plot_forecast
from bin.plotting.render_env import render_env
from src.models.base import BaseModel
from src.postprocessing.spectral_and_modes_analysis import plot_modes
from src.utils.common import get_config_path, transform_time
from src.utils.results import log_scores

CONFIG_NAME = "config"


def run_experiment(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    t, time_series = load_data(cfg)
    t_fixed = transform_time(t)
    splitter = TimeSeriesSplit(n_splits=2, test_size=cfg.y_window)
    t_train, t_test, x_train, x_test = None, None, None, None
    for train_idx, test_idx in splitter.split(t_fixed):
        t_train, t_test, x_train, x_test = (
            t_fixed[train_idx],
            t_fixed[test_idx],
            time_series[train_idx],
            time_series[test_idx],
        )
    model: BaseModel = instantiate(cfg.model)
    model.fit(x_train)
    x_0 = time_series[0]
    if cfg.modes is not None:
        plot_modes(t,
                   model,
                   x_0,
                   num_modes=cfg.modes.num_modes,
                   num_dims=cfg.modes.num_dims,
                   show=cfg.modes.show,
                   stochastic=cfg.modes.stochastic
                   )
    reconstr_ts = model.predict(t_train, x_0)
    x_1 = x_test[0]
    pred_ts = model.predict(t_test, x_1)

    mape_score = mape(x_train, reconstr_ts)
    print("MAPE:", np.mean(mape_score))
    metric_dict = {
        "MAPE": mape_score,
        "n_samples": cfg.dataset.x_dim,
        "model": model.name,
        "dataset": cfg.dataset.name,
    }
    log_scores(cfg, metric_dict)

    if cfg.plotting.plot:
        plot_forecast(
            t,
            time_series,
            reconstr_ts,
            pred_ts,
            model=model.name,
            y_window=cfg.y_window,
            metric=mape_score,
            metric_name="MAPE",
            save=cfg.plotting.save,
            n_plots=cfg.plotting.n_plots,
            show=True,
        )
    if cfg.plotting.render:
        render_env(cfg.T, cfg.delta_t, cfg.render_path, cfg.save_path, "ns_view_fast")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.2",
    )(run_experiment)()
