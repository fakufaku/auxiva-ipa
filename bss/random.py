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
A few routines to generate random data
"""
from typing import Optional, Tuple

import numpy as np
from scipy.stats import cauchy, expon, gamma

from .utils import iscomplex, tensor_H


def crandn(*shape, dtype=np.complex128):
    """ wrapper for numpy.random.randn that can produce complex numbers """
    out = np.zeros(shape, dtype=dtype)
    if iscomplex(out):
        out = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        return out.astype(dtype)
    else:
        out = np.random.randn(*shape)
        return out.astype(dtype)


def rand_psd(*shape, inner=None, dtype=np.complex):
    """ Random PSD matrices """
    if inner is None:
        shape = shape + (shape[-1],)
    else:
        shape = shape + (inner,)
    X = crandn(*shape, dtype=dtype)
    V = X @ tensor_H(X) / shape[-1]
    return 0.5 * (V + tensor_H(V))


def circular_symmetric(
    loc=0, scale=1.0, dim=1, size=None, distrib="laplace", dtype=np.complex128
):

    if size is None:
        size = [1]
    elif not isinstance(size, tuple):
        size = (size,)

    # generate first normal vectors
    out = crandn(*((dim,) + size), dtype=dtype)
    out /= np.linalg.norm(out, axis=0)

    # generate the norms according to exponential distribution
    if distrib == "laplace":
        # the marginal of the norm of symmetric circularl Laplace distributed
        # vectors is a gamma random variable with shape=dim and scale=1
        if dtype in [np.complex128, np.complex64]:
            a = 2 * dim
        else:
            a = dim
        norms = gamma.rvs(a, size=size)
    elif distrib == "gauss":
        raise NotImplementedError
        norms = cauchy.rvs(loc=loc, scale=scale, size=size)
    else:
        raise NotImplementedError()

    out *= norms[None, ...]

    return out


def rand_mixture(
    n_freq: int,
    n_sources: int,
    n_frames: int,
    distrib: Optional[str] = "laplace",
    scale: Optional[float] = 1.0,
    dtype: Optional[np.dtype] = np.complex128,
    ) -> Tuple[np.array, np.array, np.array]:

    A = crandn(n_freq, n_sources, n_sources, dtype=dtype)
    groundtruths = circular_symmetric(
        dim=n_freq,
        size=(n_sources, n_frames),
        distrib=distrib,
        scale=scale,
        dtype=dtype,
    )
    mixtures = A @ groundtruths

    return mixtures, groundtruths, A
