"""
HEAD problem solvers
"""

from enum import Enum

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import (LinearOperator, bicg, bicgstab, cgs, gmres,
                                 lgmres)

from .update_rules import _ipa
from .utils import iscomplex, isreal, tensor_H
from .random import crandn, rand_psd


class HEADUpdate(Enum):
    NCG = "NCG"
    IP = "IP"
    ISS = "ISS"
    IPA = "IPA"
    IP2 = "IP2"
    IPA_NCG = "IPA_NCG"
    FP1 = "FP1"


def _validate_covmat_parameter(V):
    n_freq, n_chan = V.shape[:2]
    assert V.shape[2:] == (n_chan, n_chan)

    return n_freq, n_chan


def _validate_input_parameters(V, W):
    n_freq, n_chan = _validate_covmat_parameter(V)

    eye = np.broadcast_to(np.eye(n_chan, n_chan), (n_freq, n_chan, n_chan))

    # use identity if initial value is not provided
    if W is None:
        W = np.zeros((n_freq, n_chan, n_chan), dtype=V.dtype)
        W[:] = eye
    elif callable(W):
        W = W(V)

    assert W.shape == (n_freq, n_chan, n_chan)

    return V, W, eye, n_freq, n_chan

def head_system(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    # shape (n_freq, n_chan, n_chan, 1)
    VW_H = V @ np.conj(W[..., None])
    VW_H = VW_H[..., 0].swapaxes(-2, -1)

    # shape (n_freq, n_chan, n_chan)
    WVW_H = W @ VW_H

    return WVW_H

def head_error(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)
    WVW_H = head_system(V, W)
    return np.mean(np.abs(WVW_H - eye) ** 2, axis=(1, 2))


def head_cost(V, W):

    W_rows = W[:, :, None, :]
    quad = np.abs(W_rows @ V @ tensor_H(W_rows))

    _, logdet = np.linalg.slogdet(W)

    return np.sum(quad, axis=(1, 2)) - 2 * logdet


def make_transpose_matrix(n, dtype=np.float64):
    """
    This matrix transposes a vectorized square matrix
    """
    eye = np.eye(n ** 2)
    P = np.zeros((n ** 2, n ** 2), dtype=dtype)
    for i in range(n ** 2):
        P[i, :] = eye[i, :].reshape((n, n), order="F").reshape((n ** 2), order="C")

    return P


def make_HEAD_Hessian_matrix(V):

    n_freq, n_chan = _validate_covmat_parameter(V)

    P = make_transpose_matrix(n_chan)

    if iscomplex(V):
        H = np.zeros((n_freq, 2 * n_chan ** 2, 2 * n_chan ** 2), dtype=V.dtype)
        for f in range(n_freq):
            bd = block_diag(*V[f])
            bd_star = block_diag(*tensor_H(V[f]))
            H[f] = -np.block([[P, bd], [np.conj(bd), P]])
    else:
        H = np.zeros((n_freq, n_chan ** 2, n_chan ** 2), dtype=V.dtype)
        for f in range(n_freq):
            H[f] = -P - block_diag(*V[f])

    return H


class HEADHessian(LinearOperator):
    """
    Implements the forward operation of the multiplication of a vector by the Heassian
    corresponding to a given HEAD problem.

    Parameters
    ----------
    cov_mat: numpy.ndarray, (n_freq, n_chan, n_chan, n_chan)
        The covriance matrices corresponding to the HEAD problem at all frequencies
    """

    def __init__(self, cov_mat):
        n_freq, n_chan = _validate_covmat_parameter(cov_mat)
        self._dtype = cov_mat.dtype

        if iscomplex(cov_mat):
            self._shape = (2 * n_freq * n_chan ** 2, 2 * n_freq * n_chan ** 2)
            self._iscomplex = True
        else:
            self._shape = (n_freq * n_chan ** 2, n_freq * n_chan ** 2)
            self._iscomplex = False

        self.cov_mat = cov_mat
        self.n_freq = n_freq
        self.n_chan = n_chan

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def _matvec(self, x):
        """
        Parameters
        ----------
        x: numpy.ndarray, (n_freq, n_chan, n_chan)
        """
        if self._iscomplex:
            X = x[: len(x) // 2].reshape((self.n_freq, self.n_chan, self.n_chan))
        else:
            X = x.reshape((self.n_freq, self.n_chan, self.n_chan))

        # copy to a single letter variable for conveniance
        # shape (n_freq, n_chan, n_chan, n_chan)
        Q = self.cov_mat

        # multiply the rows that are stacked by the covariance matrices
        if self._iscomplex:
            QX = Q @ np.conj(X[..., None])
        else:
            QX = Q @ X[..., None]
        QX = QX[..., 0]

        # this should be the result of the multiplication by the Hessian
        HX = -(QX + X.swapaxes(-2, -1))

        # we need to reeturn a flat vector
        hx = HX.flatten()

        if self._iscomplex:
            return np.concatenate((hx, np.conj(hx)))
        else:
            return hx


def solve_hessian_system(V, X, tol=1e-20, maxiter=None, use_cg=True):
    """
    wrapper around the conjugate gradient squared solver
    """
    """
    if maxiter is None:
        maxiter = np.prod(X.shape[1:])
    """
    n_freq, n_chan = _validate_covmat_parameter(V)

    if use_cg:
        # use iterative method to solve

        if maxiter is None:
            maxiter = 3 * X.shape[-1] ** 2

        sols = []
        info = []
        for f in range(V.shape[0]):

            if iscomplex(V):
                X_flat = X[f].flatten()
                x0 = np.concatenate((X_flat, np.conj(X_flat)))
                ret = bicgstab(HEADHessian(V[f : f + 1]), x0, tol=tol, maxiter=maxiter)
                sols.append(ret[0][: len(ret[0]) // 2].reshape(X.shape[1:]))
            else:
                ret = bicgstab(
                    HEADHessian(V[f : f + 1]), X[f].flatten(), tol=tol, maxiter=maxiter
                )
                sols.append(ret[0].reshape(X.shape[1:]))

            info.append(ret[1])

        """
        if iscomplex(V):
            X_flat = X.flatten()
            x0 = np.concatenate((X_flat, np.conj(X_flat)))
            ret = bicgstab(HEADHessian(V), x0, maxiter=None, tol=tol * V.shape[0] ** 2)
            ret = (ret[0][: len(ret[0]) // 2].reshape(X.shape),) + ret[1:]
        else:
            x0 = X.flatten()
            ret = bicgstab(HEADHessian(V), x0, maxiter=None, tol=tol * V.shape[0] ** 2)
            ret = (ret[0].reshape(X.shape),) + ret[1:]
        sols, info = ret
        """

    else:
        # use regular linear system solver
        H = make_HEAD_Hessian_matrix(V)

        if iscomplex(V):
            X_flat = X.reshape((n_freq, n_chan ** 2))
            x0 = np.concatenate((X_flat, np.conj(X_flat)), axis=1)
            sols = np.linalg.solve(H, x0)[..., : n_chan ** 2].reshape(X.shape)
        else:
            sols = np.linalg.solve(H, X.reshape(n_freq, n_chan ** 2)).reshape(X.shape)

        info = [1] * n_freq

    return sols, info


def head_update_ncg(V, W, use_cg=False):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    # update the covariance matrices
    W_bc = np.broadcast_to(W[:, None, :, :], V.shape)
    W_H_bc = np.broadcast_to(tensor_H(W)[:, None, :, :], V.shape)
    V = W_bc @ V @ W_H_bc

    # construct the deviation
    eps = eye - (V @ eye[..., None])[..., 0]
    # print("EPS:", eps)

    # solve the Hessian system
    ret = solve_hessian_system(V, eps, use_cg=use_cg)
    # print("CGS output:", ret)
    W[:] = (eye - ret[0]) @ W

    return W


def head_update_ip(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    for s in range(n_chan):
        WV = np.matmul(W, V[:, s, ...])
        rhs = np.eye(n_chan)[None, :, s]  # s-th canonical basis vector
        W[:, s, :] = np.conj(np.linalg.solve(WV, rhs))

        # normalize
        denom = np.matmul(
            np.matmul(W[:, None, s, :], V[:, s, :, :]), np.conj(W[:, s, :, None])
        )
        W[:, s, :] /= np.sqrt(denom[:, :, 0])

    return W


def head_update_ipa(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    V = V.swapaxes(0, 1)

    for s in range(n_chan):
        W[:] = _ipa(V, W, s)

    return W


def head_update_ip2(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    V = V.swapaxes(0, 1)

    for s in range(0, n_chan, 2):
        s1, s2 = s, s + 1

        rhs = eye[:, :, [s1, s2]]

        # Basis for demixing vector
        H = []
        HVH = []
        for i, s in enumerate([s1, s2]):
            H.append(np.linalg.solve(W @ V[s], rhs))
            HVH.append(tensor_H(H[i]) @ V[s] @ H[i])

        # Now solve the generalized eigenvalue problem
        lmbda_, R = np.linalg.eig(np.linalg.solve(HVH[0], HVH[1]))

        # Order by decreasing order of eigenvalues
        I_inv = lmbda_[:, 0] > lmbda_[:, 1]
        lmbda_[I_inv, :] = lmbda_[I_inv, ::-1]
        R[I_inv, :, :] = R[I_inv, :, ::-1]

        for i, s in enumerate([s1, s2]):
            denom = np.sqrt(np.conj(R[:, None, :, i]) @ HVH[i] @ R[:, :, i, None])
            W[:, s, None, :] = tensor_H(H[i] @ (R[:, :, i, None] / denom))

    return W


def head_update_iss(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    for s in range(n_chan):
        # shape (n_freq, n_chan, 1, 1)
        v_top = W[:, :, None, :] @ V @ np.conj(W[:, [s], :, None])
        v_bot = W[:, [s], None, :] @ V @ np.conj(W[:, [s], :, None])
        v = v_top / v_bot

        # adjust the scale
        v[:, s] -= np.sqrt(np.reciprocal(np.maximum(v_bot[:, s, :, :], 1e-15)))

        # remove the last dim
        v = v[..., 0]

        # subtract from demixing matrix
        W -= v * W[:, [s], :]

    return W

def head_update_fp1(V, W):
    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    # shape (n_freq, n_chan, n_chan, 1)
    VW_H = V @ np.conj(W[..., None])
    VW_H = VW_H[..., 0].swapaxes(-2, -1)  # remove last dim

    W = np.linalg.inv(VW_H)

    scale = W[..., None, :] @ V @ np.conj(W[..., None])
    W /= np.sqrt(scale[..., 0])

    return W

def head_solver(
    V, W=None, method=HEADUpdate.NCG, maxiter=1000, tol=1e-10, info=False, verbose=False
):
    """
    Solves the head problem using some iterative method.

    * NCG: Newton-CGS method [1]
    * IP: The iterative projection (IP) method described in [2]
    * IP2: The iterative projection 2 (IP2) method described in [2]
    * IPA: The iterative projection with adjustment (IPA) [3]

    Parameters
    ----------
    V: numpy.ndarray, (n_freq, n_chan, n_chan, n_chan)
        The covariance matrices of the HEAD problem.
    W: numpy.ndarray, (n_freq, n_chan, n_chan), optional
        The initial value of the demixing matrices
    method: HEADUpdate
        The method to use
    maxiter: int, optional
        The maximum number of iterations to run
    tol: float, optional
        The tolerance threshold in the residual to stop the iterations

    Notes
    -----

    [1] Arie Yeredor, On Hybrid Exact-Approximate Joint Diagonalization, Proc. IEEE CAMSAP, 2009.
    [2]	S. Degerine and A. Zaidi, Separation of an Instantaneous Mixture of Gaussian Autoregressive
        Sources by the Exact Maximum Likelihood Approach, IEEE Trans. Signal Process., vol. 52, no. 6,
        pp. 1499–1512, Jun. 2004.
    [3] R. Scheibler, Independent Vector Analysis via Log-quadratically Penalized Quadratic Minimization,
        arXiv, 2020.
    """

    f_update = {
        HEADUpdate.NCG: head_update_ncg,
        HEADUpdate.IP: head_update_ip,
        HEADUpdate.ISS: head_update_iss,
        HEADUpdate.IP2: head_update_ip2,
        HEADUpdate.IPA: head_update_ipa,
        HEADUpdate.FP1: head_update_fp1,
    }

    V, W, eye, n_freq, n_chan = _validate_input_parameters(V, W)

    if method == HEADUpdate.IPA_NCG:
        W, info_init = head_solver(
            V, W=W, tol=1e-5, method=HEADUpdate.IPA, info=True, maxiter=maxiter
        )

        # if we reached the maximum number of iterations, finish early
        if info_init["epochs"] == maxiter or info_init["head_error"] <= tol:
            if info:
                return W, info_init
            else:
                return W

        print("Going for NCG!")

        # update the parameters
        method = HEADUpdate.NCG
        maxiter = maxiter - info_init["epochs"]
        ipa_ncg = True
    else:
        ipa_ncg = False

    c_prev = None

    # compute the error
    e = head_error(V, W)
    I_proc = e > tol

    for epoch in range(maxiter):

        W[I_proc] = f_update[method](V[I_proc], W[I_proc])

        # compute the error
        e = head_error(V, W)
        I_proc = e > tol

        if verbose:
            c_prt = np.sum(head_cost(V, W))
            e_prt = np.sum(e)
            print(f"{epoch:4d}: error={e_prt:7.5e} cost={c_prt:7.5e}")

        if np.sum(I_proc) == 0:
            break

    if info:
        c = np.sum(head_cost(V, W))
        e = np.mean(e)
        info_content = {"cost": c, "head_error": e, "epochs": epoch + 1}

        if ipa_ncg:
            info_content["epochs"] += info_init["epochs"]

        return W, info_content
    else:
        return W


def test_transpose_matrix(n):
    X = np.random.randn(n, n)
    P = make_transpose_matrix(n)

    X1 = P @ X.flatten()
    X2 = X.T.flatten()

    error = np.max(np.abs(X1 - X2))


def test_Hessian_matrix(V):
    n_freq, n_chan = _validate_covmat_parameter(V)
    H = make_HEAD_Hessian_matrix(V)
    H_lo = HEADHessian(V)

    X = crandn(n_freq, n_chan, n_chan, dtype=V.dtype)

    if iscomplex(V):
        X_flat = X.reshape((n_freq, n_chan ** 2))
        x0 = np.concatenate((X_flat, np.conj(X_flat)), axis=1)

        X1 = (H @ x0[..., None])[..., : n_chan ** 2, 0].reshape(X.shape)
        X2 = H_lo.matvec(x0.flatten())[..., : n_chan ** 2].reshape(
            (n_freq, n_chan, n_chan)
        )
        print("condition number", np.linalg.cond(H))

        Y1 = np.linalg.solve(H, x0)[..., : n_chan ** 2].reshape(X.shape)
        Y2 = solve_hessian_system(V, X)[0]
    else:
        X1 = (H @ X.reshape((n_freq, n_chan ** 2))[..., None])[..., 0].reshape(
            (n_freq, n_chan, n_chan)
        )
        X2 = H_lo.matvec(X.flatten()).reshape(X.shape)

        Y1 = np.linalg.solve(H, X.reshape(n_freq, n_chan ** 2)).reshape(X.shape)
        Y2 = solve_hessian_system(V, X)[0]

    error = np.max(np.abs(X1 - X2))
    error_inv = np.max(np.abs(Y1 - Y2))
    print(f"error matvec={error} solve={error_inv}")


if __name__ == "__main__":

    n_freq = 1
    n_chan = 6
    n_samples = 100
    dtype = np.complex128
    dtype = np.float64

    V = rand_psd(n_freq, n_chan, n_chan, inner=n_samples, dtype=dtype)

    test_transpose_matrix(4)
    test_Hessian_matrix(V)
