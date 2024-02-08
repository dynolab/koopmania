#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henning Lange (helange@uw.edu)
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim


class koopman(nn.Module):
    r"""

    model_obj: an object that specifies the function f and how to optimize
               it. The object needs to implement numerous function. See
               below for some examples.

    sample_num: number of samples from temporally local loss used to
                reconstruct the global error surface.

    batch_size: Number of temporal snapshots processed by SGD at a time
                default = 32
                type: int

    parallel_batch_size: Number of temporaly local losses sampled in parallel.
                         This number should be as high as possible but low enough
                         to not cause memory issues.
                         default = 1000
                         type: int

    device: The device on which the computations are carried out.
            Example: cpu, cuda:0, or list of GPUs for multi-GPU usage, i.e. ['cuda:0', 'cuda:1']
            default = 'cpu'


    """

    def __init__(
        self,
        name: str,
        model_obj,
        sample_num=12,
        weight_decay=0.00,
        lr_theta=3e-3,
        lr_omega=1e-7,
        iterations=10,
        interval=5,
        cutoff=np.inf,
        verbose=False,
        hard_code_periods=None,
        **kwargs,
    ):
        super(koopman, self).__init__()
        self.name = name
        self.num_freq = model_obj.num_freq
        self.iterations = iterations
        self.interval = interval
        self.cutoff = cutoff
        self.verbose = verbose
        if "device" in kwargs:
            self.device = kwargs["device"]
            if type(kwargs["device"]) == list:
                self.device = kwargs["device"][0]
                multi_gpu = True
            else:
                multi_gpu = False
        else:
            self.device = "cpu"
            multi_gpu = False

        # Initial guesses for frequencies
        if self.num_freq == 1:
            self.omegas = torch.tensor([0.2], device=self.device)
        else:
            self.omegas = torch.linspace(0.01, 0.5, self.num_freq, device=self.device)

        self.multi_gpu = multi_gpu

        self.parallel_batch_size = (
            kwargs["parallel_batch_size"] if "parallel_batch_size" in kwargs else 1000
        )
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32

        model_obj = model_obj.to(self.device)
        self.model_obj = (
            nn.DataParallel(model_obj, device_ids=kwargs["device"])
            if multi_gpu
            else model_obj
        )

        self.sample_num = sample_num
        self.weight_decay = weight_decay
        self.lr_theta = lr_theta
        self.lr_omega = lr_omega
        self.hard_code_periods = hard_code_periods

    def sample_error(self, xt, which):
        """

        sample_error computes all temporally local losses within the first
        period, i.e. between [0,2pi/t]

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        i : TYPE int
            Index of the entry of omega

        Returns
        -------
        TYPE numpy.array
            Matrix that contains temporally local losses between [0,2pi/t]
            dimensions: [T, sample_num]

        """

        num_samples = self.sample_num
        omega = self.omegas

        if type(xt) == np.ndarray:
            xt = torch.tensor(xt, device=self.device)

        t = torch.arange(xt.shape[0], device=self.device) + 1

        errors = []
        batch = self.parallel_batch_size
        pi_block = torch.zeros((num_samples, len(omega)), device=self.device)
        pi_block[:, which] = torch.arange(0, num_samples) * np.pi * 2 / num_samples

        for i in range(int(np.ceil(xt.shape[0] / batch))):
            t_batch = t[i * batch : (i + 1) * batch][:, None]
            wt = t_batch * omega[None]
            wt[:, which] = 0
            wt = wt[:, None] + pi_block[None]
            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)
            loss = (
                self.model_obj(k, xt[i * batch : (i + 1) * batch, None])
                .cpu()
                .detach()
                .numpy()
            )
            errors.append(loss)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return np.concatenate(errors, axis=0)

    def reconstruct(self, errors, use_heuristic=True):
        e_fft = np.fft.fft(errors)
        E_ft = np.zeros(errors.shape[0] * self.sample_num, dtype=np.complex64)

        for t in range(1, e_fft.shape[0] + 1):
            E_ft[np.arange(self.sample_num // 2) * t] += e_fft[
                t - 1, : self.sample_num // 2
            ]

        E_ft = np.concatenate([E_ft, np.conj(np.flip(E_ft, -1))])[:-1]
        E = np.real(np.fft.ifft(E_ft))

        if use_heuristic:
            E = -np.abs(E - np.median(E))
            # E = gaussian_filter(E, 5)

        return E, E_ft

    def fft(self, xt, i, verbose=False):
        """

        fft first samples all temporaly local losses within the first period
        and then reconstructs the global error surface w.r.t. omega_i
        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        i : TYPE int
            Index of the entry of omega
        verbose : TYPE boolean, optional
            DESCRIPTION. The default is False.
        Returns
        -------
        E : TYPE numpy.array
            Global loss surface in time domain.
        E_ft : TYPE
            Global loss surface in frequency domain.
        """

        E, E_ft = self.reconstruct(self.sample_error(xt, i))
        omegas = np.linspace(0, 1, len(E))

        idxs = np.argsort(E[: len(E_ft) // 2])

        omegas_actual = self.omegas.cpu().detach().numpy()
        omegas_actual[i] = -1
        found = False

        j = 0
        while not found:
            # The if statement avoids non-unique entries in omega and that the
            # frequencies are 0 (should be handle by bias term)
            if idxs[j] > 1 and np.all(
                np.abs(2 * np.pi / omegas_actual - 1 / omegas[idxs[j]]) > 1
            ):
                found = True
                if verbose:
                    print(omegas[idxs[j]])
                    print("Setting ", i, "to", 1 / omegas[idxs[j]])
                self.omegas[i] = torch.from_numpy(np.array([omegas[idxs[j]]]))
                self.omegas[i] *= 2 * np.pi

            j += 1

        return E, E_ft

    def sgd(self, xt, verbose=False):
        """

        sgd performs a single epoch of stochastic gradient descent on parameters
        of f (Theta) and frequencies omega

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        verbose : TYPE boolean, optional
            The default is False.

        Returns
        -------
        TYPE float
            Loss.

        """

        batch_size = self.batch_size

        T = xt.shape[0]

        omega = nn.Parameter(self.omegas)

        opt = optim.SGD(
            self.model_obj.parameters(),
            lr=self.lr_theta,
            weight_decay=self.weight_decay,
        )
        opt_omega = optim.SGD([omega], lr=self.lr_omega / T)

        T = xt.shape[0]
        t = torch.arange(T, device=self.device)

        losses = []

        for i in range(len(t) // batch_size):
            ts = t[i * batch_size : (i + 1) * batch_size]
            o = torch.unsqueeze(omega, 0)
            ts_ = torch.unsqueeze(ts, -1).type(torch.get_default_dtype()) + 1

            xt_t = torch.tensor(xt[ts.cpu().numpy(), :], device=self.device)

            wt = ts_ * o

            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)
            loss = torch.mean(self.model_obj(k, xt_t))

            opt.zero_grad()
            opt_omega.zero_grad()

            loss.backward()

            opt.step()
            opt_omega.step()

            losses.append(loss.cpu().detach().numpy())

        if verbose:
            print("Setting to", 2 * np.pi / omega)

        self.omegas = omega.data

        return np.mean(losses)

    def fit(self, xt):
        """
        Given a dataset, this function alternatingly optimizes omega and
        parameters of f. Specifically, the algorithm performs interval many
        epochs, then updates all entries in omega. This process is repeated
        until iterations-many epochs have been performed

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        iterations : TYPE int, optional
            Total number of SGD epochs. The default is 10.
        interval : TYPE, optional
            The interval at which omegas are updated, i.e. if
            interval is 5, then omegas are updated every 5 epochs. The default is 5.
        verbose : TYPE boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        assert len(xt.shape) > 1, "Input data needs to be at least 2D"
        losses = []
        hard_coded_omegas = (
            2 * np.pi / torch.tensor(self.hard_code_periods)
            if self.hard_code_periods is not None
            else []
        )
        # assert len(hard_coded_omegas) < self.num_fourier_modes
        if len(hard_coded_omegas) > 0:
            # set the last omegas to the hard-coded ones
            self.omegas[-len(hard_coded_omegas) :] = hard_coded_omegas
        for i in range(self.iterations):
            if i % self.interval == 0 and i < self.cutoff:
                for k in range(self.num_freq - len(hard_coded_omegas)):
                    self.fft(xt, k, verbose=self.verbose)

            if self.verbose:
                print("Iteration ", i)
                print(2 * np.pi / self.omegas)

            l = self.sgd(xt, verbose=self.verbose)
            if self.verbose:
                print("Loss: ", l)
            losses.append(l)

        return losses

    def predict(self, t, x0):
        """
        Predicts the data from 1 to T.

        Parameters
        ----------
        T : TYPE int
            Prediction horizon

        Returns
        -------
        TYPE numpy.array
            xhat from 0 to T.

        """
        # T = len(t)
        step = t[1] - t[0]

        # t = torch.arange(round(t[0]/step), round(t[-1] / step), device=self.device) + 1
        t = torch.tensor(t / step, device=self.device)
        ts_ = torch.unsqueeze(t, -1).type(torch.get_default_dtype())

        o = torch.unsqueeze(self.omegas, 0)
        k = torch.cat([torch.cos(ts_ * o), torch.sin(ts_ * o)], -1)

        if self.multi_gpu:
            mu = self.model_obj.module.decode(k)
        else:
            mu = self.model_obj.decode(k)

        return mu.cpu().detach().numpy()

    def mode_decomposition(self, T, n_modes, plot=False, plot_n_last=None):
        """
        Returns first n modes of prediction built by Koopman algorithm
        :param T: TYPE int
        prediction horizon
        :param n_modes: TYPE int
        number of modes to return
        :param plot: TYPE bool
        whether to build a plot
        The default is False
        :param plot_n_last: TYPE int
        default None
        if not None, plot only last n steps in prediction
        :return: TYPE np.array
        size (T, n_modes)

        """
        if plot_n_last is None:
            plot_n_last = T
        n_lim = max(0, T - plot_n_last)
        t = torch.arange(n_lim, T, device=self.device) + 1

        # t = torch.arange(T, device=self.device) + 1
        ts_ = torch.unsqueeze(t, -1).type(torch.get_default_dtype())

        o = torch.unsqueeze(self.omegas, 0)
        k = torch.cat([torch.cos(ts_ * o), torch.sin(ts_ * o)], -1)

        modes = self.model_obj.get_modes(k)

        amps = self.model_obj.get_amplitudes()
        idxs = torch.argsort(-amps.abs())
        for i in range(n_modes):
            mode = modes[:, idxs[i]].detach().numpy()
            if plot:
                plt.plot(mode)
                plt.xlabel("Time")
                plt.title(f"Koopman mode {i}")
                plt.show()

        return modes[:, idxs[:n_modes]].detach().numpy()


class coordinate_koopman(koopman):
    def __init__(
        self,
        name: str,
        model_obj,
        sample_num=12,
        weight_decay=0.0,
        l1_coef=0.0,
        lr_theta=3e-3,
        lr_omega=1e-7,
        lr_mlp=3e-3,
        iterations=10,
        interval=5,
        cutoff=np.inf,
        verbose=False,
        hard_code_periods=None,
        **kwargs,
    ):
        super().__init__(
            name,
            model_obj,
            sample_num,
            weight_decay,
            lr_theta,
            lr_omega,
            iterations,
            interval,
            cutoff,
            verbose,
            hard_code_periods,
            **kwargs,
        )
        self.l1_coef = l1_coef
        self.lr_mlp = lr_mlp

    def sgd(self, xt, verbose=False):
        """

        sgd performs a single epoch of stochastic gradient descent on parameters
        of f (Theta) and frequencies omega

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        verbose : TYPE boolean, optional
            The default is False.

        Returns
        -------
        TYPE float
            Loss.

        """

        batch_size = self.batch_size

        T = xt.shape[0]

        omega = nn.Parameter(self.omegas)

        opts = []
        for i in range(self.num_freq):
            opt = optim.SGD(
                self.model_obj.networks[i].parameters(),
                lr=self.lr_theta,
                weight_decay=0.0005 / (self.omegas[i].float() * T + 1).log() + 1e-8,
            )
            opts.append(opt)
        opt_mlp = optim.SGD(
            self.model_obj.mlp.parameters(),
            lr=self.lr_mlp,
            weight_decay=self.weight_decay,
        )
        opt_omega = optim.SGD([omega], lr=self.lr_omega / T)

        T = xt.shape[0]
        t = torch.arange(T, device=self.device)

        losses = []

        for i in range(len(t) // batch_size):
            ts = t[i * batch_size : (i + 1) * batch_size]
            o = torch.unsqueeze(omega, 0)
            ts_ = torch.unsqueeze(ts, -1).type(torch.get_default_dtype()) + 1

            xt_t = torch.tensor(xt[ts.cpu().numpy(), :], device=self.device)

            wt = ts_ * o

            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)
            loss = (
                torch.mean(self.model_obj(k, xt_t))
                + self.l1_coef * (omega.abs()).mean()
            )

            for opt in opts:
                opt.zero_grad()
            opt_mlp.zero_grad()
            opt_omega.zero_grad()

            loss.backward()

            for opt in opts:
                opt.step()
            opt_mlp.step()
            opt_omega.step()

            losses.append(loss.cpu().detach().numpy())

        if verbose:
            print("Setting to", 2 * np.pi / omega)

        self.omegas = omega.data

        return np.mean(losses)

    def MAE(self, x, x_hat):
        assert len(x) == len(x_hat)
        return np.sum(np.abs(x - x_hat)) / len(x)

    def MAPE(self, x, x_hat):
        assert len(x) == len(x_hat)
        return np.mean(np.abs((x - x_hat) / x))

    def validate(self, val_data, train_through, horizon, metric="MAPE"):
        total_pred = self.predict(train_through + horizon)
        val_pred = total_pred[train_through:]
        if metric == "MAPE":
            return self.MAPE(val_data, val_pred)
        elif metric == "MAE":
            return self.MAE(val_data, val_pred)


# def train(xt, train_through, horizon, freqs_list=None, metric="MAPE"):
#     T = xt.shape[0]
#     x_dim = xt.shape[1]
#     x_train = xt[:train_through].reshape(-1, 1)
#     x_val = xt[train_through : train_through + horizon].reshape(-1, 1)
#     best_m = 1
#     best_metric = np.inf
#     metrics = []
#     if freqs_list is None:
#         m_freqs = np.arange(1, 11)
#     for m in m_freqs:
#         model = coordinate_koopman(
#             multi_nn_mse(x_dim, m, fully_connected_mse(x_dim=1, num_freqs=1, n=64)),
#             device="cpu",
#         )
#         model.fit(x_train, iterations=300)
#         cur_metric = model.validate(x_val, train_through, horizon, metric=metric)
#         metrics.append(cur_metric)
#         if cur_metric < best_metric:
#             best_m = m
#
#     final_model = coordinate_koopman(
#         multi_nn_mse(x_dim, best_m, fully_connected_mse(x_dim=1, num_freqs=1, n=64)),
#         device="cpu",
#     )
#     final_model.fit(xt.reshape(-1, 1), iterations=1000)
#     return final_model, best_m, metrics
