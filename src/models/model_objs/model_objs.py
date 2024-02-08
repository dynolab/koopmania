from copy import deepcopy

import torch
from torch import nn


class model_object(nn.Module):
    def __init__(self, num_freq, x_dim):
        super(model_object, self).__init__()
        self.num_freq = num_freq
        self.x_dim = x_dim

    def forward(self, y, x):
        """
        Forward computes the error.

        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]

            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """

        raise NotImplementedError()

    def decode(self, y):
        """
        Evaluates f at temporal snapshots y

        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]

            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """
        raise NotImplementedError()

    def get_modes(self, x):
        """
        Returns modes as an output of neural network before the final fully-connected layer
        :param x: data set
                    type: torch.tensor
                    dimensions: [T, ...]
        :return: torch.tensor
                    dimensions: [T, ..., 2*num_freqs]
        """
        raise NotImplementedError()

    def get_amplitudes(self):
        """
        Returns amplitudes corresponding to the weights of final fully-connected layer
        :return: torch.tensor
                    dimensions: [2*num_freqs, x_dim]
        """
        raise NotImplementedError()


class fully_connected_mse(model_object):
    def __init__(self, x_dim, num_freqs, n):
        super(fully_connected_mse, self).__init__(num_freqs, x_dim)

        self.l1 = nn.Linear(2 * num_freqs, n)
        self.l2 = nn.Linear(n, n * 2)
        # self.l21 = nn.Linear(n * 2, n * 4)
        self.l3 = nn.Linear(n * 2, 2 * num_freqs)
        self.amplitudes = nn.Linear(2 * num_freqs, x_dim)

    def get_modes(self, x):
        o0 = x
        o1 = nn.Tanh()(self.l1(o0))
        o2 = nn.Tanh()(self.l2(o1))
        o3 = self.l3(o2)
        return o3

    def get_amplitudes(self):
        return self.state_dict()["amplitudes.weight"].flatten()

    def decode(self, x):
        o0 = x
        o1 = nn.Tanh()(self.l1(o0))
        o2 = nn.Tanh()(self.l2(o1))
        o21 = nn.Tanh()(self.l21(o2))
        o3 = self.l3(o21)
        o4 = self.amplitudes(o3)
        return o3

    def forward(self, y, x):
        xhat = self.decode(y)
        return torch.mean((xhat - x) ** 2, dim=-1)


class multi_nn_mse(model_object):
    def __init__(self, x_dim, num_freqs, base_model):
        super(multi_nn_mse, self).__init__(num_freqs, x_dim)
        self.networks = []
        for i in range(num_freqs):
            model = deepcopy(base_model)
            self.networks.append(model)
        self.mlp = nn.Linear(2 * num_freqs, x_dim)

    def get_modes(self, x):
        x = x.reshape(*x.shape[:-1], 2, -1).transpose(-2, -1)
        y = []
        for i in range(self.num_freq):
            y.append(self.networks[i].get_modes(x[:, i]))
        y = torch.cat(y, -1)
        return y

    def get_amplitudes(self):
        return self.mlp.state_dict()["weight"].flatten()

    def decode(self, x):
        x = x.reshape(*x.shape[:-1], 2, -1).transpose(-2, -1)
        y = []
        for i in range(self.num_freq):
            y.append(
                self.networks[i].get_modes(torch.index_select(x, -2, torch.tensor([i])))
            )
        y = torch.cat(y, -2)
        y = self.mlp(y.flatten(-2))
        return y

    def forward(self, y, x):
        xhat = self.decode(y)
        return torch.mean((xhat - x) ** 2, dim=-1)


class observables_lib(model_object):
    def __init__(
        self, num_freq, x_dim, hidden_dim, latent_dim, num_poly, num_exp, num_layers
    ):
        super(observables_lib, self).__init__(num_freq)
        self.num_poly = num_poly
        self.num_exp = num_exp
        self.latent_dim = latent_dim
        self.x_dim = x_dim

        model = []
        model += [nn.Linear(x_dim * num_freq * 2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, self.latent_dim * num_freq * 2 * x_dim)]

        self.model = nn.Sequential(*model)

        self.mlp = nn.Linear(latent_dim, x_dim)

    def decode(self, x):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        encoder_out = self.model(x)
        encoder_out = encoder_out.reshape(
            x.shape[0], x.shape[1], self.latent_dim, self.num_freq * 2 * self.x_dim
        )
        encoder_out = encoder_out.reshape(
            x.shape[0], x.shape[1], self.latent_dim, self.num_freq * 2, self.x_dim
        )
        x = x.reshape(x.shape[0], x.shape[1], self.num_freq * 2, self.x_dim)
        coefs = torch.einsum("blkdf, bldf -> blfk", encoder_out, x)
        embedding = torch.zeros(
            encoder_out.shape[0], encoder_out.shape[1], self.x_dim, self.latent_dim
        )
        for f in range(self.x_dim):
            for i in range(self.num_poly):
                embedding[:, :, f, i] = coefs[:, :, f, i] ** (i + 1)

            for i in range(self.num_poly, self.num_poly + self.num_exp):
                embedding[:, :, f, i] = torch.exp(coefs[:, :, f, i])

            embedding[:, :, f, self.num_poly + self.num_exp :] = coefs[
                :, :, f, self.num_poly + self.num_exp :
            ]

        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        out = self.mlp(embedding)
        return out

    def forward(self, y, x):
        xhat = self.decode(y)
        if len(x.shape) == 2:
            xhat = xhat.flatten(-2)
        return torch.mean((xhat - x) ** 2, dim=-1)
