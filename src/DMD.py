import matplotlib.pyplot as plt
import numpy as np


def DMD(t, xt, r):
    """Dynamic Mode Decomposition (DMD) algorithm."""
    X = np.transpose(xt)
    ## Build data matrices
    X1 = X[:, : -1]
    X2 = X[:, 1:]
    m, n = X1.shape
    # print(X.shape, X2.shape)
    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices=False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    print(A_tilde.shape)
    ## Perform eigenvalue decomposition on A_tilde
    L, W = np.linalg.eig(A_tilde)
    Lambda = np.diag(L)

    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ W
    # A = Psi @ Lambda @ np.linalg.pinv(Psi)
    sgm = np.diag(Lambda)

    omega = np.log(sgm)/(t[1]-t[0])

    ## STEP 5: reconstruct the signal
    x1 = X[:, 0]  # time = 0

    b = np.dot(np.linalg.pinv(Psi), x1)

    t_dyn = np.zeros((m, t.shape[0]), dtype=X.dtype)

    for i in range(t.shape[0]):
        t_dyn[:, i] = Psi * np.exp(omega * t[i]) @ b


    return np.transpose(t_dyn), Psi

# t = np.arange(10)
# X = np.zeros((2, 10))
# X[0, :] = np.arange(1, 11)
# X[1, :] = np.arange(2, 12)
# pred_step = 2
# r = 2
# mat_hat = DMD(t, X.T, r)
# print(mat_hat, 'mat hat')
# for i in range(2):
#     plt.plot(t, X[i])
#     plt.plot(t, mat_hat[:,i])
#     plt.title('x dim = {}'.format(i+1))
#     plt.show()

