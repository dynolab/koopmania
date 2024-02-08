import numpy as np


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
        t = t - np.min(t)
        omega = np.log(sgm) / (t[1] - t[0])
        b = np.dot(np.linalg.pinv(self.Psi), x0)

        t_dyn = np.zeros((m, t.shape[0]))
        for i in range(t.shape[0]):
            t_dyn[:, i] = self.Psi * np.exp(omega * t[i]) @ b

        return np.transpose(t_dyn)
