import numpy as np
from numpy.typing import NDArray

from src.models.base import BaseModel


class DMD(BaseModel):
    def __init__(self, name: str, rank: int):
        super().__init__(name)
        self.rank = rank
        self.Lambda = None
        self.Psi = None

    def fit(self, x_train: NDArray) -> None:
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

    def predict(self, t: NDArray, x0: NDArray) -> NDArray:
        m = x0.shape[0]
        sgm = np.diag(self.Lambda)
        b = np.dot(np.linalg.pinv(self.Psi), x0)

        t_dyn = np.zeros((m, t.shape[0]))
        t_dyn[:, 0] = x0
        for i in range(1, t.shape[0]):
            t_dyn[:, i] = self.Psi * (sgm ** (i - 1)) @ b

        return np.transpose(t_dyn)

    def mode_decomposition(
        self, T: int, n_modes: int, x0: NDArray, n_dims: int = 1
    ) -> NDArray:
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

            dyn_modes_dim = np.stack(dyn_modes_dim, axis=-1)
            dyn_modes.append(dyn_modes_dim)
        dyn_modes = np.stack(dyn_modes, axis=-2)
        return dyn_modes[0]
