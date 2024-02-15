import hydra
import numpy as np
from hydra.utils import instantiate
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit

from bin.plotting.plot_ts import plot_forecast
from bin.plotting.render_env import render_env
from src.utils.common import get_config_path, transform_time
from src.utils.results import log_scores

CONFIG_NAME = "config"


def run_experiment(cfg):
    np.random.seed(cfg.seed)
    loader = instantiate(cfg.dataset)
    t, time_series = loader.load()
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
    model = instantiate(cfg.model)
    model.fit(x_train)
    x_0 = time_series[0]
    model.mode_decomposition(1000, 3, x_0, plot=True)
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
    with open("stats_kt.csv", "a") as f:
        f.write(f"{x_test.shape[1]}")
        f.write(" ")
        f.write(f"{np.mean(mape_score)}")
        f.write("\n")
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


# t = np.load("../data/t_3.npy")
# time_series = np.load("../data/ts_3.npy")

# mask_idxs = np.random.randint(512 * 4, time_series.shape[1] - 512 * 4, size=500)
# mask_idxs_list = []
# for i in range(-3, 3):
#     for j in range(-3, 3):
#         mask_idxs_list.append(mask_idxs + 512 * j + i)
#
# mask_idxs = np.concatenate(mask_idxs_list)
# dmd = DMD(10)
# dmd.fit(time_series[:, mask_idxs])
# x_0 = time_series[0, mask_idxs]
# reconstr_dmd, DM = dmd.predict(t, x_0)
# x_1 = time_series[-300, mask_idxs]
# pred_dmd, _ = dmd.predict(t, x_1)
# empty = np.empty((512, 512, 1601))
# empty[:] = np.nan
# palette = copy(plt.get_cmap("jet"))
# palette.set_bad("white", 1.0)  # 1.0 represents not transparent
#
# levels = np.arange(-1, 1, 0.1)
# levels[0] = -1 + 1e-5
# norm = BoundaryNorm(levels, ncolors=palette.N)
# for i in range(1300, empty.shape[2]):
#     idx_1, idx_2 = mask_idxs // 512, mask_idxs % 512
#     empty[idx_1, idx_2, i] = reconstr_dmd[i]
#     # norm = TwoSlopeNorm(vmin=time_series.min(), vcenter=0, vmax=time_series.max())
#     plt.imshow(empty[::-1, ::-1, i], norm=norm, cmap="jet")
#     plt.title(f"t={round(t[i],1)}")
#     plt.savefig(
#         rf"C:\Users\mWX1298408\Documents\GitHub\koopmania\bin\render_dmd\plot_{i}".replace(
#             ".", ","
#         )
#         + ".png"
#     )
