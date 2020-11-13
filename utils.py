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
A few routines for complex valued operations
"""
import numpy as np


def iscomplex(x):
    x = np.array(x)
    return x.dtype in [np.complex64, np.complex128]


def tensor_H(X):
    return np.conj(X).swapaxes(-1, -2)


def crandn(*shape, dtype=np.complex):
    out = np.zeros(shape, dtype=dtype)
    t = dtype(1.0)
    if iscomplex(t):
        out = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        return out.astype(dtype)
    else:
        out = np.random.randn(*shape)
        return out.astype(dtype)


def rand_psd(*shape, dtype=np.complex):
    X = crandn(*shape, dtype=dtype)
    return X @ tensor_H(X) / shape[-1]
