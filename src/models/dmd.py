import matplotlib.pyplot as plt
import numpy as np
import torch


class DMD:
    def __init__(self, name: str, rank: int):
        self.name = name
        self.rank = rank
        self.Lambda = None
        self.Psi = None

    def fit(self, x_train):
        X = np.transpose(x_train)
        ## Build data matrices
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        ## Perform singular value decomposition on X1
        u, s, v = np.linalg.svd(X1, full_matrices=False)
        ## Compute the Koopman matrix
        A_tilde = (
            u[:, : self.rank].conj().T
            @ X2
            @ v[: self.rank, :].conj().T
            * np.reciprocal(s[: self.rank])
        )
        ## Perform eigenvalue decomposition on A_tilde
        L, W = np.linalg.eig(A_tilde)
        self.Lambda = np.diag(L)

        self.Psi = (
            X2 @ v[: self.rank, :].conj().T @ np.diag(np.reciprocal(s[: self.rank])) @ W
        )

    def predict(self, t, x0):
        m = x0.shape[0]
        sgm = np.diag(self.Lambda)
        b = np.dot(np.linalg.pinv(self.Psi), x0)

        t_dyn = np.zeros((m, t.shape[0]))
        t_dyn[:, 0] = x0
        for i in range(1, t.shape[0]):
            t_dyn[:, i] = self.Psi * (sgm ** (i - 1)) @ b

        return np.transpose(t_dyn)

    def mode_decomposition(
        self, T, n_modes, x0, n_dims=1, plot=False, plot_n_last=None
    ):
        modes = self.Psi[:, :n_modes]
        dyn_modes = []
        for j in range(n_dims):
            dyn_modes_dim = []
            for i in range(n_modes):
                mode = modes[j : j + 1, i : i + 1]

                sgm = np.diag(self.Lambda)[i]
                b = np.dot(np.linalg.pinv(mode), x0[j])

                t_dyn = np.zeros((1, T))
                t_dyn[:, 0] = x0[j]
                for k in range(1, T):
                    t_dyn[:, k] = mode * (sgm ** (k - 1)) @ b
                dyn_modes_dim.append(t_dyn)

                if plot:
                    print(t_dyn.shape)
                    plt.plot(t_dyn[0])
                    plt.xlabel("Time")
                    plt.title(f"DMD mode {i} at dim {j}")
                    plt.show()
            dyn_modes_dim = np.stack(dyn_modes_dim, axis=-1)
            dyn_modes.append(dyn_modes_dim)
        dyn_modes = np.stack(dyn_modes, axis=-2)
        return dyn_modes[0]
        # for i in range(DM.shape[2]):
        #     plt.imshow(DM[:, :, i].real, cmap="jet")
        #     plt.colorbar()
        #     plt.title("Mode {}, real part".format(i))
        #     plt.savefig(
        #         rf"C:\Users\mWX1298408\Documents\koopman_plots\Mode_{i}_real.png"
        #     )
        #     plt.show()
        #
        #     plt.imshow(DM[:, :, i].imag, cmap="jet")
        #     plt.colorbar()
        #     plt.title("Mode {}, imaginary part".format(i))
        #     plt.savefig(
        #         rf"C:\Users\mWX1298408\Documents\koopman_plots\Mode_{i}_imag.png"
        #     )
        #     plt.show()
