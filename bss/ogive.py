# Copyright (c) 2019 Robin Scheibler
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
Blind Source Extraction using Independent Vector Extraction via the OGIVE algorithm [1].

[1]	Z. Koldovský and P. Tichavský, “Gradient Algorithms for Complex
Non-Gaussian Independent Component/Vector Extraction, Question of Convergence,”
IEEE Trans. Signal Process., pp. 1050–1064, Dec. 2018.
"""
import os
import numpy as np

from .projection_back import project_back
from .utils import tensor_H
from .update_rules import _parametric_background_update


def ogive_mix(X, **kwargs):
    return ogive(X, update="mix", **kwargs)


def ogive_demix(X, **kwargs):
    return ogive(X, update="demix", **kwargs)


def ogive_switch(X, **kwargs):
    return ogive(X, update="switching", **kwargs)


def ogive(
    X,
    n_iter=4000,
    step_size=0.1,
    tol=1e-3,
    update="demix",
    proj_back=True,
    model="laplace",
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):

    """
    Implementation of Orthogonally constrained Independent Vector Extraction
    (OGIVE) described in

    Z. Koldovský and P. Tichavský, “Gradient Algorithms for Complex
    Non-Gaussian Independent Component/Vector Extraction, Question of Convergence,”
    IEEE Trans. Signal Process., pp. 1050–1064, Dec. 2018.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    step_size: float
        The step size of the gradient ascent
    tol: float
        Stop when the gradient is smaller than this number
    update: str
        Selects update of the mixing or demixing matrix, or a switching scheme,
        possible values: "mix", "demix", "switching"
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence
    callback_checkpoints: list of int
        A list of epoch number when the callback should be called

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape
    n_src = 1

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)
    Cx_inv = np.linalg.inv(Cx)
    Cx_norm = np.linalg.norm(Cx, axis=(1, 2))

    w = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    a = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    delta = np.zeros((n_freq, n_chan, 1), dtype=X.dtype)
    lambda_a = np.zeros((n_freq, 1, 1), dtype=np.float64)

    # initialize A and W
    w[:, 0] = 1.0

    def update_a_from_w(I):
        v_new = Cx[I] @ w[I]
        lambda_w = 1.0 / np.real(tensor_H(w[I]) @ v_new)
        a[I, :, :] = lambda_w * v_new

    def update_w_from_a(I):
        v_new = Cx_inv @ a
        lambda_a[:] = 1.0 / np.real(tensor_H(a) @ v_new)
        w[I, :, :] = lambda_a[I] * v_new[I]

    def switching_criterion():

        a_n = a / a[:, :1, :1]
        b_n = Cx @ a_n
        lmb = b_n[:, :1, :1].copy()  # copy is important here!
        b_n /= lmb

        p1 = np.linalg.norm(a_n - b_n, axis=(1, 2)) / Cx_norm
        Cbb = (
            lmb
            * (b_n @ tensor_H(b_n))
            / np.linalg.norm(b_n, axis=(1, 2), keepdims=True) ** 2
        )
        p2 = np.linalg.norm(Cx - Cbb, axis=(1, 2))

        kappa = p1 * p2 / np.sqrt(n_chan)

        thresh = 0.1
        I_do_a[:] = kappa >= thresh
        I_do_w[:] = kappa < thresh

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = X @ np.conj(W)

    # The very first update of a
    update_a_from_w(np.ones(n_freq, dtype=np.bool))

    if update == "mix":
        I_do_w = np.zeros(n_freq, dtype=np.bool)
        I_do_a = np.ones(n_freq, dtype=np.bool)
    else:  # default is "demix"
        I_do_w = np.ones(n_freq, dtype=np.bool)
        I_do_a = np.zeros(n_freq, dtype=np.bool)

    r_inv = np.zeros((n_frames, n_src))
    r = np.zeros((n_frames, n_src))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    X_ref = X  # keep a reference to input signal
    X = X.swapaxes(0, 1).copy()  # more efficient order for processing

    if callback is not None or return_filters:
        W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
        W[:, :, :] = np.eye(n_chan)[None, :, :]
        W[:, 1:, 1:] *= -1

    # Extract target
    demix(Y, X, w)

    for epoch in range(n_iter):
        # compute the switching criterion
        if update == "switching" and epoch % 10 == 0:
            switching_criterion()

        # simple loop as a start
        # shape: (n_frames, n_src)
        if model == "laplace":
            r[:, :] = np.linalg.norm(Y, axis=0) / np.sqrt(n_freq)

        elif model == "gauss":
            r[:, :] = (np.linalg.norm(Y, axis=0) ** 2) / n_freq

        eps = 1e-15
        r[r < eps] = eps

        r_inv[:, :] = 1.0 / r

        # Compute the score function
        psi = r_inv[None, :, :] * np.conj(Y)

        # "Nu" in Algo 3 in [1]
        # shape (n_freq, 1, 1)
        zeta = Y.swapaxes(1, 2) @ psi

        x_psi = (X.swapaxes(1, 2) @ psi) / zeta

        # The w-step
        # shape (n_freq, n_chan, 1)
        delta[I_do_w] = a[I_do_w] - x_psi[I_do_w]
        w[I_do_w] += step_size * delta[I_do_w]

        # The a-step
        # shape (n_freq, n_chan, 1)
        delta[I_do_a] = w[I_do_a] - (Cx_inv[I_do_a] @ x_psi[I_do_a]) * lambda_a[I_do_a]
        a[I_do_a] += step_size * delta[I_do_a]

        # Apply the orthogonal constraints
        update_a_from_w(I_do_w)
        update_w_from_a(I_do_a)

        # Extract the target signal
        demix(Y, X, w)

        max_delta = np.max(np.linalg.norm(delta, axis=(1, 2)))

        # Now run any necessary callback
        if callback is not None and (epoch + 1) in callback_checkpoints:
            W[:, :1, :] = tensor_H(w)
            _parametric_background_update(1, W, Cx)
            Y_tmp = Y.swapaxes(0, 1).copy()
            callback(Y_tmp, model)

        if max_delta < tol:
            break

    Y = Y.swapaxes(0, 1).copy()
    X = X.swapaxes(0, 1)

    if proj_back:
        Y = project_back(Y, X_ref[:, :, 0])

    if return_filters:
        W[:, :1, :] = tensor_H(w)
        W[:, 1:, :1] = a[:, 1:, :]
        return Y, W
    else:
        return Y
