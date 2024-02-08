import hydra
import numpy as np
from hydra.utils import instantiate

from bin.plotting.plot_ts import plot_forecast
from src.utils.common import get_config_path, split

CONFIG_NAME = "config"
np.random.seed(421)


def MAPE(x, x_hat):
    assert len(x) == len(x_hat)
    x_c = x.copy()
    x_hat_c = x_hat.copy()
    x_c += 1
    x_hat_c += 1
    return np.mean(np.abs((x_c - x_hat_c) / (x_c + 1e-6)), axis=0)


def main(cfg):
    loader = instantiate(cfg.dataset)
    t, time_series = loader.load()
    x_train, x_test = split(time_series, cfg.y_window)
    t_train, t_test = split(t, cfg.y_window)
    model = instantiate(cfg.model)
    model.fit(x_train)
    x_0 = time_series[0]

    reconstr_ts = model.predict(t_train, x_0)
    x_1 = x_test[0]
    pred_ts = model.predict(t_test, x_1)

    mape = MAPE(x_test, pred_ts)
    print("MAPE:", np.mean(mape))
    if cfg.plot:
        plot_forecast(
            t,
            time_series,
            reconstr_ts,
            pred_ts,
            model=model.name,
            y_window=cfg.y_window,
            metric=mape,
            metric_name="MAPE",
            save=None,
            sample=cfg.sample_plot,
            show=True,
        )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.2",
    )(main)()


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

# # DM = DM.reshape(512, 512, -1)
# # for i in range(DM.shape[2]):
# #     plt.imshow(DM[:, :, i].real, cmap='jet')
# #     plt.colorbar()
# #     plt.title("Mode {}, real part".format(i))
# #     plt.savefig(fr"C:\Users\mWX1298408\Documents\koopman_plots\Mode_{i}_real.png")
# #     plt.show()
# #
# #     plt.imshow(DM[:, :, i].imag, cmap='jet')
# #     plt.colorbar()
# #     plt.title("Mode {}, imaginary part".format(i))
# #     plt.savefig(fr"C:\Users\mWX1298408\Documents\koopman_plots\Mode_{i}_imag.png")
# #     plt.show()
# #
# # idxs_to_plot = np.random.randint(time_
# # for i, idx in enumerate(mask_idxs[:3]):series.shape[1], size=10)
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
