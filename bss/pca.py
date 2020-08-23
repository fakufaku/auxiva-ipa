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
import numpy as np

from .projection_back import project_back
from .utils import tensor_H


def pca(X, n_src=None, proj_back=False, return_filters=False, normalize=True, **kwargs):
    """
    Whitens the input signal X using principal component analysis (PCA)

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        The input signal
    n_src: int
        The desired number of principal components
    return_filters: bool
        If this flag is set to true, the PCA matrix
        is also returned (default False)
    normalize: bool
        If this flag is set to false, the decorrelated
        channels are not normalized to unit variance
        (default True)
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = n_chan

    assert (
        n_src <= n_chan
    ), "The number of sources cannot be more than the number of channels."

    # compute the cov mat (n_freq, n_chan, n_chan)
    X_T = X.transpose([1, 2, 0])
    covmat = (X_T @ tensor_H(X_T)) * (1.0 / n_frames)

    # make sure the covmat is hermitian symmetric and positive semi-definite
    covmat = 0.5 * (covmat + tensor_H(covmat))

    # Compute EVD
    # v.shape == (n_freq, n_chan), w.shape == (n_freq, n_chan, n_chan)
    eig_val, eig_vec = np.linalg.eigh(covmat)

    # Reorder the eigenvalues from so that they are in descending order
    eig_val = eig_val[:, ::-1]
    eig_vec = eig_vec[:, :, ::-1]

    # The whitening matrices
    if normalize:
        Q = (1.0 / np.sqrt(eig_val[:, :, None])) * tensor_H(eig_vec)
    else:
        Q = tensor_H(eig_vec)

    # The decorrelated signal
    Y = (Q[:, :n_src, :] @ X_T).transpose([2, 0, 1]).copy()

    # Optional projection back
    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        return Y, Q
    else:
        return Y
