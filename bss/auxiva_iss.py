# Copyright (c) 2020 Robin Scheibler
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
Implementation of AuxIVA with independent source steering (ISS) updates
"""
import numpy as np

from .projection_back import project_back
from .utils import TwoStepsIterator, demix, tensor_H


def auxiva_iss(
    X,
    n_iter=20,
    proj_back=True,
    model="laplace",
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):

    """
    Blind source separation based on independent vector analysis with
    alternating updates of the mixing vectors
    Robin Scheibler, Nobutaka Ono, Unpublished, 2019
    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    model: str
        The model of source distribution 'gauss' r 'laplace' (default)
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence
    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    n_src = X.shape[2]

    # for now, only supports determined case
    assert n_chan == n_src

    # pre-allocate arrays
    r_inv = np.zeros((n_src, n_frames))
    v_num = np.zeros((n_freq, n_src), dtype=X.dtype)
    v_denom = np.zeros((n_freq, n_src), dtype=np.float64)
    v = np.zeros((n_freq, n_src), dtype=X.dtype)

    # Things are more efficient when the frequencies are over the first axis
    X = X.transpose([1, 2, 0]).copy()

    # Initialize the demixed outputs
    Y = X.copy()

    for epoch in range(n_iter):

        # shape: (n_src, n_frames)
        # OP: n_frames * n_src
        eps = 1e-10
        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )

        # Update now the demixing matrix
        for s in range(n_src):

            # OP: n_frames * n_src
            v_num = (Y * r_inv[None, :, :]) @ np.conj(
                Y[:, s, :, None]
            )  # (n_freq, n_src, 1)
            # OP: n_frames * n_src
            v_denom = r_inv[None, :, :] @ np.abs(Y[:, s, :, None]) ** 2
            # (n_freq, n_src, 1)

            # OP: n_src
            v[:, :] = v_num[:, :, 0] / v_denom[:, :, 0]
            # OP: 1
            v[:, s] -= 1 / np.sqrt(v_denom[:, s, 0])

            # update demixed signals
            # OP: n_frames * n_src
            Y[:, :, :] -= v[:, :, None] * Y[:, s, None, :]

        # Monitor the algorithm progression
        if callback is not None and (epoch + 1) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1]).copy()
            callback(Y_tmp, model)

    if return_filters is not None:
        # Demixing matrices were not computed explicitely so far,
        # do it here, if necessary
        W = Y[:, :, :n_chan] @ np.linalg.inv(X[:, :, :n_chan])

    Y = Y.transpose([2, 0, 1]).copy()
    X = X.transpose([2, 0, 1])

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y
