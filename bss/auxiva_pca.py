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
Blind Source Separation using Independent Vector Analysis with Auxiliary Function
with a principal component analysis pre-processing step used to reduce the number
of channels.
"""
import numpy as np

import pyroomacoustics as pra
from .projection_back import project_back
from .overiva import overiva
from .pca import pca


def auxiva_pca(X, n_src=None, return_filters=False, **kwargs):
    """
    Implementation of overdetermined IVA with PCA followed by determined IVA

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    return_filters: bool
        If true, the function will return the demixing matrix too

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    new_X, W_pca = pca(X, n_src=n_src, return_filters=True)

    kwargs.pop("proj_back")
    Y, W_iva = overiva(new_X, proj_back=False, return_filters=True, **kwargs)

    if "proj_back" in kwargs and kwargs["proj_back"]:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        W = W_pca.copy()
        W[:, :n_src, :] = W_iva @ W_pca[:, :n_src, :]
        return Y, W
    else:
        return Y
