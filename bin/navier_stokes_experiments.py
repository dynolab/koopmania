import numpy as np
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from PIL import Image
from src.models.koopman import koopman
from src.models.DMD import DMD
from copy import copy
from src.models.model_objs.model_objs import fully_connected_mse
from src.data_loaders.navier_stokes import NSLoader
from src.utils.common import get_config_path, split, split_t
from bin.plotting.plot_ts import plot_forecast

CONFIG_NAME = "config"
np.random.seed(421)


def MAPE(x, x_hat):
    assert len(x) == len(x_hat)
    x_c = x.copy()
    x_hat_c = x_hat.copy()
    x_c += 1
    x_hat_c += 1
    return np.mean(np.abs((x_c - x_hat_c) / (x_c + 1e-6)), axis=0)


# run_NS()
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
            sample=10,
            show=True,
        )
    # for i in range(10):
    #     plt.plot(t, time_series[:, i], label="original ts", color="blue")
    #     plt.plot(t_train, reconstr_ts[:, i], label=model.name, color="red")
    #     plt.plot(t[-cfg.predict_horizon:], pred_ts[:, i], linestyle="--", color="red")
    #     # plt.plot(t[:-300], pred_dmd[:-300, i], label="DMD", color="green")
    #     # plt.plot(t[-300:], pred_dmd[-300:, i], linestyle="--", color="green")
    #     plt.axvline(x=t[-cfg.predict_horizon], color="black", linestyle="--")
    #     plt.title("example {}".format(i + 1))
    #     plt.legend()
    #     plt.show()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.2",
    )(main)()

# # t, time_series = run_NS()
# t = np.load("../data/t_3.npy")
# time_series = np.load("../data/ts_3.npy")
# print(time_series.shape)
# koopman_model = koopman(
#     fully_connected_mse(x_dim=1000, num_freqs=10, n=256),
#     device="cpu",
# )
# mask_idxs = np.random.randint(512 * 4, time_series.shape[1] - 512 * 4, size=500)
# # print(mask_idxs.shape)
# mask_idxs_list = []
# for i in range(-3, 3):
#     for j in range(-3, 3):
#         mask_idxs_list.append(mask_idxs + 512 * j + i)
#
# print(mask_idxs_list)
# mask_idxs = np.concatenate(mask_idxs_list)
# # print(mask_idxs.shape)
# # koopman_model.fit(time_series[:-300, mask_idxs], iterations=800)
# # pred = koopman_model.predict(t.shape[0])
# dmd = DMD(10)
# dmd.fit(time_series[:, mask_idxs])
# x_0 = time_series[0, mask_idxs]
# print("x_0", x_0.shape)
# reconstr_dmd, DM = dmd.predict(t, x_0)
# x_1 = time_series[-300, mask_idxs]
# pred_dmd, _ = dmd.predict(t, x_1)

# print(pred, pred.shape)
# # pred, DM = DMD(t, time_series, r=10)
# print("MAPE:", MAPE(time_series[:, mask_idxs], pred))
# #
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
#     # print(pred[i].shape)
#     empty[idx_1, idx_2, i] = reconstr_dmd[i]
#     print(i)
#     # norm = TwoSlopeNorm(vmin=time_series.min(), vcenter=0, vmax=time_series.max())
#     plt.imshow(empty[::-1, ::-1, i], norm=norm, cmap="jet")
#     plt.title(f"t={round(t[i],1)}")
#     plt.savefig(
#         rf"C:\Users\mWX1298408\Documents\GitHub\koopmania\bin\render_dmd\plot_{i}".replace(
#             ".", ","
#         )
#         + ".png"
#     )
# plt.close()
# print(pred.shape)
# for i, idx in enumerate(mask_idxs[:10]):
#     plt.plot(t, time_series[:, idx], label="original ts", color="blue")
#     plt.plot(t[:-300], pred[:-300, i], label="Koopman", color="red")
#     plt.plot(t[-300:], pred[-300:, i], linestyle="--", color="red")
#     # plt.plot(t[:-300], pred_dmd[:-300, i], label="DMD", color="green")
#     # plt.plot(t[-300:], pred_dmd[-300:, i], linestyle="--", color="green")
#     plt.axvline(x=t[-300], color="black", linestyle="--")
#     plt.title("example {}".format(i + 1))
#     plt.legend()
#     plt.show()
# plt.savefig(
#     rf"C:\Users\mWX1298408\Documents\GitHub\koopmania\bin\render_dmd\plot_{i}".replace(
#             ".", ","
#         )
#         + ".png"
#     )
#     plt.close()
#
# # render
# frames = []
#
# for frame_number in np.arange(0, 1601):
#     frame = Image.open(
#         rf"C:\Users\mWX1298408\Documents\GitHub\koopmania\bin\render_dmd\plot_{frame_number}"
#         + ".png"
#     )
#     frames.append(frame)
#
# frames[0].save(
#     f"./render_dmd/ns_view_large2.gif",
#     save_all=True,
#     append_images=frames[1:],
#     optimize=True,
#     duration=50,
#     loop=0,
# )
