# Copyright (c) 2018-2019 Robin Scheibler
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
FIVE: Fast Independent Vector Extraction
=====================================================

This algorithm extracts one source independent from a minimum energy background
using an iterative algorithm that attempts to maximize the SINR at every step.

References
----------

.. [1] R. Scheibler and N. Ono, Fast Independent Vector Extraction by Iterative
    SINR Maximization, arXiv, 2019.
"""
import numpy as np
from scipy import linalg

from .projection_back import project_back
from .utils import tensor_H
from . import default


def five(
    X,
    n_iter=3,
    proj_back=True,
    W0=None,
    model=default.model,
    init_eig=False,
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):

    """
    This algorithm extracts one source independent from a minimum energy background.
    The separation is done in the time-frequency domain and the FFT length
    should be approximately equal to the reverberation time. The residual
    energy in the background is minimized.

    Two different statistical models (Laplace or time-varying Gauss) can
    be used by specifying the keyword argument `model`. The performance of Gauss
    model is higher in good conditions (few sources, low noise), but Laplace
    (the default) is more robust in general.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_iter: int, optional
        The number of iterations (default 3)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    init_eig: bool, optional (default ``False``)
        If ``True``, and if ``W0 is None``, then the weights are initialized
        using the principal eigenvectors of the covariance matrix of the input
        data. When ``False``, the demixing matrices are initialized with identity
        matrix.
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence
    callback_checkpoints: list of int
        A list of epoch number when the callback should be called

    Returns
    -------
    Returns an (nframes, nfrequencies, 1) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    n_src = 1

    if model not in ["laplace", "gauss"]:
        raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    # We will need the inverse square root of Cx
    e_val, e_vec = np.linalg.eigh(Cx)
    Q_H_inv = (1.0 / np.sqrt(e_val[:, :, None])) * tensor_H(e_vec)
    # Q_H = e_vec[:, :, :] * np.sqrt(e_val[:, None, :])

    eps = 1e-10
    V = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    r_inv = np.zeros((n_src, n_frames))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X.copy()

    # pre-whiten the input signal
    # X = np.linalg.solve(Q_H, X.transpose([1, 2, 0]))
    X = Q_H_inv @ X.transpose([1, 2, 0])

    # initialize the output signal
    if init_eig:
        # Principal component
        Y = X[:, -1:, :].copy()
    else:
        # First microphone
        Y = X_original[:, :, :1].transpose([1, 2, 0]).copy()

    for epoch in range(n_iter):

        # update the source model
        # shape: (n_frames, n_src)
        if model == "laplace":
            r_inv = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y[:, 0, :], axis=0))
        elif model == "gauss":
            r_inv = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y[:, 0, :], axis=0) ** 2) / n_freq
            )

        # Compute Auxiliary Variable
        # shape: (n_freq, n_chan, n_chan)
        V[:, :, :] = (X * r_inv[None, None, :]) @ np.conj(X.swapaxes(1, 2)) / n_frames

        # Solve the Eigenvalue problem
        # We only need the smallest eigenvector and eigenvalue,
        # so we could solve this more efficiently, but it is faster to
        # just solve everything rather than wrap this in a for loop
        lambda_, R = np.linalg.eigh(V)
        R[:, :, :1] /= np.sqrt(lambda_[:, None, :1])

        # Update the output signal
        # note: eigenvalues are in ascending order, we use the smallest
        Y[:, :, :] = np.matmul(tensor_H(R[:, :, :1]), X)

        if return_filters and epoch == n_iter - 1:
            W = tensor_H(R)

        if callback is not None and (epoch + 1) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1])
            callback(Y_tmp, model)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        return Y, W @ Q_H_inv
    else:
        return Y
