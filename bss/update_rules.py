# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Implementation of the different types of auxiliary function
based update rules for IVA.
"""
import numpy as np

from .newton_root import bisection, init_newton, newton, newton_s
from .utils import tensor_H


def _iss_single(s, X, W, r_inv):
    """
    Update of the demixing matrix with the iterative source steering of the s-th source
    """
    _, __, n_frames = X.shape

    # demix
    Y = W @ X

    # OP: n_frames * n_src
    v_num = (Y * r_inv[None, :, :]) @ np.conj(Y[:, s, :, None])  # (n_freq, n_src, 1)
    # OP: n_frames * n_src
    v_denom = r_inv[None, :, :] @ np.abs(Y[:, s, :, None]) ** 2
    # (n_freq, n_src, 1)

    # OP: n_src
    v = v_num[:, :, 0] / v_denom[:, :, 0]
    # OP: 1
    v[:, s] = 1.0 - np.sqrt(n_frames) / np.sqrt(v_denom[:, s, 0])

    # update demixed signals
    # OP: n_frames * n_src
    W[:] -= v[:, :, None] * W[:, None, s, :]


def _ip_single(s, X, W, r_inv):
    """
    Performs update of the s-th demixing vector using
    the iterative projection rules
    """

    n_freq, n_chan, n_frames = X.shape

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = np.matmul((X * r_inv[None, None, :]), tensor_H(X)) / n_frames

    WV = np.matmul(W, V)
    rhs = np.eye(n_chan)[None, :, s]  # s-th canonical basis vector
    W[:, s, :] = np.conj(np.linalg.solve(WV, rhs))

    # normalize
    denom = np.matmul(
        np.matmul(W[:, None, s, :], V[:, :, :]), np.conj(W[:, s, :, None])
    )
    W[:, s, :] /= np.sqrt(denom[:, :, 0])


def _ip_double(s1, s2, X, W, r_inv):
    """
    Performs a joint update of the s1-th and s2-th demixing vectors
    usint the iterative projection 2 rules
    """
    n_frames = X.shape[-1]

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = [
        (X * r_inv[None, i, None, :]) @ tensor_H(X) / n_frames
        for i, s in enumerate([s1, s2])
    ]

    _ip_double_sub(s1, s2, V, W)


def _ip_double_sub(s1, s2, V, W):

    n_freq, n_chan, _ = W.shape

    # right-hand side for computation of null space basis
    # no_update = [i for i in range(n_src) if i != s]
    rhs = np.eye(n_chan)[None, :, [s1, s2]]

    # Basis for demixing vector
    H = []
    HVH = []
    for i, s in enumerate([s1, s2]):
        H.append(np.linalg.solve(W @ V[i], rhs))
        HVH.append(tensor_H(H[i]) @ V[i] @ H[i])

    # Now solve the generalized eigenvalue problem
    lmbda_, R = np.linalg.eig(np.linalg.solve(HVH[0], HVH[1]))

    # Order by decreasing order of eigenvalues
    I_inv = lmbda_[:, 0] > lmbda_[:, 1]
    lmbda_[I_inv, :] = lmbda_[I_inv, ::-1]
    R[I_inv, :, :] = R[I_inv, :, ::-1]

    for i, s in enumerate([s1, s2]):
        denom = np.sqrt(np.conj(R[:, None, :, i]) @ HVH[i] @ R[:, :, i, None])
        W[:, s, None, :] = tensor_H(H[i] @ (R[:, :, i, None] / denom))


def _ip_double_two_channels(X, W, r_inv):
    """
    Specialized update rule for the 2 sources/2 channels case.
    In this, a globally optimal update of the surrogate function exists.
    """

    n_freq, n_chan, n_frames = X.shape
    s1, s2 = 0, 1

    assert n_chan == 2, "This update rule is only valid for two channels"

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = [
        (X * r_inv[None, i, None, :]) @ tensor_H(X) / n_frames
        for i, s in enumerate([s1, s2])
    ]

    # Now solve the generalized eigenvalue problem
    lmbda_, R = np.linalg.eig(np.linalg.solve(V[0], V[1]))

    # Order by decreasing order of eigenvalues
    I_inv = lmbda_[:, 0] < lmbda_[:, 1]
    lmbda_[I_inv, :] = lmbda_[I_inv, ::-1]
    R[I_inv, :, :] = R[I_inv, :, ::-1]

    for i, s in enumerate([s1, s2]):
        denom = np.sqrt(np.conj(R[:, None, :, i]) @ V[i] @ R[:, :, i, None])
        W[:, s, None, :] = tensor_H(R[:, :, i, None] / denom)


def _parametric_background_update(n_src, W, Cx):
    """
    Update the backgroud part of a parametrized demixing matrix
    """

    W_target = W[:, :n_src, :]  # target demixing matrix
    J = W[:, n_src:, :n_src]  # background demixing matrix

    tmp = np.matmul(W_target, Cx)
    J[:, :, :] = tensor_H(np.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:]))


def _joint_demix_background(s, n_src, X, W, r_inv, Cx):
    """
    Joint update of one demixing vector and one block
    """

    n_freq, n_chan, n_frames = X.shape

    # right-hand side for computation of null space basis
    # no_update = [i for i in range(n_src) if i != s]
    update = [s] + list(range(n_src, n_chan))
    rhs = np.eye(n_chan)[None, :, update]

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = (np.matmul((X * r_inv[None, None, :]), tensor_H(X))) / n_frames

    # Basis for demixing vector
    Hw = np.linalg.solve(W @ V, rhs)
    HVH = np.matmul(tensor_H(Hw), V @ Hw)

    # Basis for the background
    Hb = np.linalg.solve(W @ Cx, rhs)
    HCH = np.matmul(tensor_H(Hb), Cx @ Hb)

    # Now solve the generalized eigenvalue problem
    B_H = np.linalg.cholesky(HCH)
    B_inv = np.linalg.inv(tensor_H(B_H))
    V_tilde = np.linalg.solve(B_H, HVH) @ B_inv
    lmbda_, R = np.linalg.eigh(V_tilde)
    R = np.matmul(B_inv, R)

    # The target demixing vector requires normalization
    R[:, :, -1] /= np.sqrt(lmbda_[:, -1, None])

    # Assign to target and background
    W[:, s, None, :] = tensor_H(Hw @ R[:, :, -1, None])
    W[:, n_src:, :] = tensor_H(Hb @ R[:, :, :-1])


def _block_ip(sources, X, W, V):
    """
    Block iterative projection update
    This is a joint update of the demixing vectors of sources
    that are not independent
    """

    n_freq, n_chan, n_frames = X.shape

    rhs = np.eye(n_chan)[None, :, sources]
    W[:, sources, :] = tensor_H(np.linalg.solve(W @ V, rhs))

    for s in sources:
        # normalize
        denom = np.matmul(
            np.matmul(W[:, None, s, :], V[:, :, :]), np.conj(W[:, s, :, None])
        )
        W[:, s, :] /= np.sqrt(denom[:, :, 0])


def _ipa_make_T(u, q, k):

    n_freq, n_chan = u.shape

    o = list(range(n_chan))
    o.remove(k)

    T = np.zeros((n_freq, n_chan, n_chan), dtype=u.dtype)
    T[:] = np.eye(n_chan)[None, :, :]
    T[:, o, k] = np.conj(q)
    T[:, k, :] = np.conj(u)

    return T


def _ipa(V, W, k):
    """
    Implementation of Iterative Projection with Adjustment (IPS) updates for IVA using the global solution based
    on Newton-Raphson root finding
    """
    n_freq, n_chan, _ = W.shape

    o = list(range(n_chan))
    o.remove(k)

    Vk = W @ V[k] @ tensor_H(W)
    Vk_inv = np.linalg.inv(Vk)

    # shape (n_freq, n_chan - 1, 1)
    a = np.real(
        [(W[:, None, k, :] @ V[m] @ np.conj(W[:, k, :, None]))[..., 0] for m in o]
    ).transpose([1, 0, 2])
    a_inv_sq = 1.0 / np.sqrt(a)

    # shape (n_freq, n_chan - 1, 1)
    b = np.array(
        [(W[:, None, k, :] @ V[m] @ np.conj(W[:, m, :, None]))[..., 0] for m in o]
    ).transpose([1, 0, 2])

    u, q = _ipa_sub(k, Vk, Vk_inv, a, a_inv_sq, b)

    T = _ipa_make_T(u[:, :, 0], q[:, :, 0], k)
    W = T @ W

    return W

def _ipa2(k, Y, W, r_inv):

    n_freq, n_chan, n_frames = Y.shape

    o = list(range(n_chan))
    o.remove(k)

    Vk = (Y * r_inv[k, None, None, :]) @ tensor_H(Y) / n_frames
    Vk = 0.5 * (Vk + np.conj(Vk))
    Vk_inv = np.linalg.inv(Vk)

    # shape (n_freq, n_chan - 1, 1)
    a = (r_inv[None, o, :]) @ np.abs(Y[:, k, :, None]) ** 2
    a /= n_frames
    a_inv_sq = 1.0 / np.sqrt(a)

    # shape (n_freq, n_chan - 1, 1)
    b = (Y[:, o, :] * r_inv[None, o, :]) @ np.conj(Y[:, k, :, None])
    b = np.conj(b)
    b /= n_frames

    u, q = _ipa_sub(k, Vk, Vk_inv, a, a_inv_sq, b)

    # compute the new separation matrix
    T = _ipa_make_T(u[:, :, 0], q[:, :, 0], k)
    W[:] = T @ W

    # update the output signal
    Yk = tensor_H(u) @ Y
    for i, ell in enumerate(o):
        Y[:, ell, :] += np.conj(q[:, i, :]) * Y[:, k, :]
    Y[:, [k], :] = Yk


def _ipa_sub(k, Vk, Vk_inv, a, a_inv_sq, b):

    n_freq, n_chan, _ = Vk.shape
    o = list(range(n_chan))
    o.remove(k)

    # shape (n_freq, n_chan, n_chan)
    C = np.conj(Vk_inv[:, :, o][:, o, :])

    # shape (n_freq, n_chan, n_chan)
    U = a_inv_sq * C * tensor_H(a_inv_sq)
    # sometimes, the hermitian symmetry is lost due to small numerical innacuracies
    # and numpy.linalg.eigh returns negative eigenvalues, throwing off the algorithm,
    # we thus enforce hermtian symmetry of U here
    U = 0.5 * (U + tensor_H(U))

    # shapes (n_freq, n_chan), (n_freq, n_chan - 1, n_chan - 1)
    phi, Sigma = np.linalg.eigh(U)

    # shape (n_freq, n_chan - 1, 1)
    ag = a_inv_sq * np.conj(Vk_inv[:, o, k, None])

    # shape (n_freq)
    g_C_inv_g = np.sum(np.abs(tensor_H(Sigma) @ ag) ** 2 / phi[:, :, None], axis=(1, 2))
    z = np.real(Vk_inv[:, k, k] - g_C_inv_g)

    # shape (n_freq, n_chan, 1)
    v = tensor_H(Sigma) @ (-U @ (a_inv_sq * b) - ag)

    # normalize so that the largest eigenvalue is one
    phi_max = phi[:, -1]
    phi_hat = phi / phi_max[:, None]
    v_hat = v[:, :, 0] / phi_max[:, None]
    z_hat = z / phi_max

    # make an array to receive the solution
    q = np.zeros((n_freq, n_chan - 1, 1), dtype=Vk.dtype)
    lambda_ = np.zeros(n_freq, dtype=np.float)

    # when v is very small, the solution is given by the dominant eigenvector
    # of U
    I_small = np.linalg.norm(v[:, :, 0], axis=1) < 1e-10
    if np.sum(I_small) > 0:

        # there is one more case we need to account for
        # when z is larger than the largest eigenvalue,
        # the optimal q vector is zero, and lambda = z
        I_z = np.logical_and(I_small, z >= phi_max)
        lambda_[I_z] = z[I_z]

        # when z is smaller than the largest eigenvalue
        # then lambda = largest eigenvalue and q is the
        # corresponding eigenvector
        I_ow = np.logical_and(I_small, z < phi_max)
        uv = Sigma[I_ow, :, -1, None]  # leading eigenvector
        lambda_[I_ow] = phi[I_ow, -1]

        scale = np.sqrt(phi[I_ow, -1] - z[I_ow]) / np.sqrt(
            np.real(tensor_H(uv) @ U[I_ow] @ uv)[:, 0, 0]
        )
        q[I_ow, :, 0] = (
            a_inv_sq[I_ow, :, 0] * (uv[:, :, 0] * scale[:, None]) + b[I_ow, :, 0]
        )

    # initial value for root finding
    # anything larger than the largest eigenvalue should be fine
    I_big = np.logical_not(I_small)

    # use the smart initialization
    lambda_0 = init_newton(phi_hat[I_big, :], v_hat[I_big, :], z_hat[I_big])

    # newton
    lambda_[I_big] = newton(
        np.maximum(lambda_0, z_hat[I_big]),
        phi_hat[I_big, :],
        v_hat[I_big, :],
        z_hat[I_big],
        atol=1e-5,
        max_iter=1000,
        verbose=False,
        plot=False,
    )

    # rescale
    lambda_[I_big] *= phi_max[I_big]

    # finally piece together the optimum vector
    t = v[I_big, :] / (lambda_[I_big, None, None] - phi[I_big, :, None])
    q[I_big, :] = a_inv_sq[I_big] * (Sigma[I_big] @ t) - b[I_big] / a[I_big]

    # build the other demixing vector
    x = np.ones((n_freq, n_chan, 1), dtype=Vk.dtype)
    x[:, o, :] = -np.conj(q)
    u = Vk_inv[:, :, k, None] - Vk_inv[:, :, o] @ np.conj(q)

    I = np.where(lambda_ <= 0.0)[0]
    if len(I) > 0:
        print("Detected bad cases, investigate:")
        v_not_hat = -U @ (a_inv_sq * b) - ag
        for i in I:
            print("*****")
            print("freq:", i)
            print("U=")
            print(U[i])
            print("v=")
            print(v_not_hat[i])
            print("z=", z[i])
            print("phi=")
            print(phi[i])
            print("v_hat=")
            print(v[i])
        print("*****")
        raise ValueError(
            "Numerical problem. Please report to "
            " https://github.com/fakufaku/auxiva-ipa"
        )

    u *= 1.0 / np.sqrt(lambda_[:, None, None])

    return u, q

