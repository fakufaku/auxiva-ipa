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
A few routines that are used accross the board.
"""
import numpy as np


# Compute the demixed output
def demix(Y, X, W):
    Y[:, :, :] = np.matmul(W, X)


def iscomplex(x):
    x = np.array(x)
    return x.dtype in [np.complex64, np.complex128]


def isreal(x):
    return not iscomplex(x)


def tensor_H(T):
    return np.conj(T).swapaxes(-2, -1)


def cost_iva(X, Y, model=None):
    """
    Parameters
    ----------
    X: ndarray (n_frames, n_freq, n_chan)
        The microphone signals
    Y: ndarray (n_frames, n_freq, n_src)
        The demixed sources
    model: str
        The source model
    """
    n_frames, n_freq, n_src = Y.shape
    n_chan = X.shape[2]

    XX = X.transpose([1, 2, 0])
    YY = Y.transpose([1, 2, 0])

    if n_src < n_chan:
        U = np.zeros((n_freq, n_chan - n_src, n_chan), dtype=XX.dtype)

        YZ = np.zeros_like(XX)

        YX = YY @ tensor_H(XX)
        J = np.linalg.solve(YX[:, :, :n_src], YX[:, :, n_src:])
        U[:, :, :n_src] = tensor_H(J)
        U[:, :, n_src:] = -np.eye(n_chan - n_src)[None, :, :]

        YZ[:, :n_src, :] = YY
        YZ[:, n_src:, :] = U @ XX
    else:
        YZ = YY

    W_H = np.linalg.solve(XX @ tensor_H(XX), XX @ tensor_H(YZ))

    if model == "laplace":
        target_loss = np.sum(np.linalg.norm(Y, axis=1))
    elif model == "gauss":
        target_loss = np.sum(np.log(1.0 / np.maximum(1e-15, np.linalg.norm(Y, axis=1))))
    else:
        raise ValueError("Invalid model")

    # background loss is constant

    _, logdet = np.linalg.slogdet(W_H)
    demix_loss = -2 * n_frames * np.sum(logdet)

    return target_loss + demix_loss


class TwoStepsIterator(object):
    """
    Iterates two elements at a time between 0 and m - 1
    """

    def __init__(self, m):
        self.m = m

    def _inc(self):
        self.next = (self.next + 1) % self.m
        self.count += 1

    def __iter__(self):
        self.count = 0
        self.next = 0
        return self

    def __next__(self):

        if self.count < 2 * self.m:

            m = self.next
            self._inc()
            n = self.next
            self._inc()

            return m, n

        else:
            raise StopIteration
