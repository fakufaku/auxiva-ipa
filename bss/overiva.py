# Copyright (c) 2018-2020 Robin Scheibler
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
AuxIVA
======

Blind Source Separation using independent vector analysis based on auxiliary function.
This function will separate the input signal into statistically independent sources
without using any prior information.

The algorithm in the determined case, i.e., when the number of sources is equal to
the number of microphones, is AuxIVA [1]_. When there are more microphones
(the overdetermined case), a computationaly cheaper variant (OverIVA) is used [2]_.

Example
-------
.. code-block:: python

    from scipy.io import wavfile
    import pyroomacoustics as pra

    # read multichannel wav file
    # audio.shape == (nsamples, nchannels)
    fs, audio = wavfile.read("my_multichannel_audio.wav")

    # STFT analysis parameters
    fft_size = 4096  # `fft_size / fs` should be ~RT60
    hop == fft_size // 2  # half-overlap
    win_a = pra.hann(fft_size)  # analysis window
    # optimal synthesis window
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # STFT
    # X.shape == (nframes, nfrequencies, nchannels)
    X = pra.transform.analysis(audio, fft_size, hop, win=win_a)

    # Separation
    Y = pra.bss.auxiva(X, n_iter=20)

    # iSTFT (introduces an offset of `hop` samples)
    # y contains the time domain separated signals
    # y.shape == (new_nsamples, nchannels)
    y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)

References
----------

.. [1] N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique,* Proc. IEEE, WASPAA, pp. 189-192, Oct. 2011.

.. [2] R. Scheibler and N. Ono, Independent Vector Analysis with more Microphones
    than Sources, arXiv, 2019.  https://arxiv.org/abs/1905.07880
"""
import numpy as np

from .head import HEADUpdate, head_error, head_solver, head_update_ncg
from .projection_back import project_back
from .update_rules import (_block_ip, _ip_double, _ip_double_two_channels,
                           _ip_single, _ipa, _iss_single,
                           _joint_demix_background,
                           _parametric_background_update)
from .utils import TwoStepsIterator, demix, tensor_H

_eps = 1e-15
_update_rules_choice = [
    "ip-param",
    "ip2-param",
    "demix-bg",
    "ip-block",
    "ip2-block",
    "ipa",
    "auxiva-iss",
    "ipancg",
    "fullhead",
]
_dual_update_rules = ["ip2-param", "ip2-block", "ipa"]


def overiva_ip_param(X, **kwargs):
    return overiva(X, update_rule="ip-param", **kwargs)


def overiva_ip2_param(X, **kwargs):
    return overiva(X, update_rule="ip2-param", **kwargs)


def overiva_demix_bg(X, **kwargs):
    return overiva(X, update_rule="demix-bg", **kwargs)


def overiva_ip_block(X, **kwargs):
    return overiva(X, update_rule="ip-block", **kwargs)


def overiva_ip2_block(X, **kwargs):
    return overiva(X, update_rule="ip2-block", **kwargs)


def auxiva(X, **kwargs):
    kwargs.pop("n_src")
    return overiva(X, n_src=None, update_rule="ip-param", **kwargs)


def auxiva2(X, **kwargs):
    kwargs.pop("n_src")
    return overiva(X, n_src=None, update_rule="ip2-param", **kwargs)


def auxiva_iss(X, **kwargs):
    kwargs.pop("n_src")
    return overiva(X, n_src=None, update_rule="auxiva-iss", **kwargs)


def auxiva_ipa(X, **kwargs):
    kwargs.pop("n_src")
    return overiva(X, n_src=None, update_rule="ipa", **kwargs)


def auxiva_ipancg(X, **kwargs):
    kwargs.pop("n_src")
    return overiva(X, n_src=None, update_rule="ipancg", **kwargs)


def auxiva_fullhead(X, **kwargs):
    kwargs.pop("n_src")
    return overiva(X, n_src=None, update_rule="fullhead", **kwargs)


def overiva(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    model="laplace",
    update_rule="ip-param",
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):

    """
    This is an implementation of AuxIVA/OverIVA that separates the input
    signal into statistically independent sources. The separation is done
    in the time-frequency domain and the FFT length should be approximately
    equal to the reverberation time.

    Two different statistical models (Laplace or time-varying Gauss) can
    be used by using the keyword argument `model`. The performance of Gauss
    model is higher in good conditions (few sources, low noise), but Laplace
    (the default) is more robust in general.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components. When
        ``n_src==nchannels``, the algorithms is identical to AuxIVA. When
        ``n_src==1``, then it is doing independent vector extraction.
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    update_rule: str
        The update rules to use for the algorithm, one of ``ip-param``, ``ip2-param``,
        ``demix-bg``, ``ip-block``
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

    # default to determined case
    if n_src is None:
        n_src = n_chan

    assert (
        n_src <= n_chan
    ), "The number of sources cannot be more than the number of channels."

    if model not in ["laplace", "gauss"]:
        raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

    # allocate demixing matrix and take view to the target and background parts
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)

    # buffer and helpers
    r_inv = np.zeros((n_src, n_frames))

    # Initialize the demixing matrix to identity
    W[:, :n_src, :n_src] = np.eye(n_src)[None, :, :]
    W[:, n_src:, n_src:] = -np.eye(n_chan - n_src)[None, :, :]  # for the constraint

    # Things are more efficient when the frequencies are over the first axis
    X_original = X  # keep the original for the projection back
    X = X.transpose([1, 2, 0]).copy()

    # Because we initialize the demixing matrix to identity
    # we can just copy X to Y at the beginning
    Y = X[:, :n_src, :].copy()

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    if n_chan > n_src:
        Cx = (X @ tensor_H(X)) / n_frames
        # ensures the matrix is hermitian positive semi-definite
        Cx = 0.5 * (Cx + tensor_H(Cx))

    if update_rule in _dual_update_rules:
        iter_step = 2
    else:
        iter_step = 1

    for epoch in range(0, n_iter, iter_step):

        # update the source model
        # shape: (n_src, n_frames)
        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(_eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                _eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )

        # Update now the demixing matrix
        # using the requested algorithm

        if update_rule == "ip-param":

            for s in range(n_src):

                # Update the mixing matrix according to orthogonal constraints
                if n_src < n_chan:
                    _parametric_background_update(n_src, W, Cx)

                # Iterative Projection
                _ip_single(s, X, W, r_inv[s, :])

        elif update_rule == "ip2-param":

            # Update source pairs with a joint IP2 update
            for s1, s2 in TwoStepsIterator(n_src):

                if n_src < n_chan:
                    _parametric_background_update(n_src, W, Cx)

                # Iterative Projection 2
                if n_chan == 2:
                    _ip_double_two_channels(X, W, r_inv)
                else:
                    _ip_double(s1, s2, X, W, r_inv[[s1, s2], :])

        elif update_rule == "demix-bg":

            assert n_chan > n_src, "demix-bg only works in the overdetermined case"

            for s in range(n_src):
                _joint_demix_background(s, n_src, X, W, r_inv[s, :], Cx)

        elif update_rule == "ip-block":

            # Update each source with an IP update
            for s in range(n_src):

                if n_src < n_chan:
                    # Update the background with a block IP update
                    _block_ip(list(range(n_src, n_chan)), X, W, Cx)

                # Iterative Projection
                _ip_single(s, X, W, r_inv[s, :])

        elif update_rule == "ip2-block":

            # Update source pairs with a joint IP2 update
            for s1, s2 in TwoStepsIterator(n_src):

                if n_src < n_chan:
                    # Update the background with a block IP update
                    _block_ip(list(range(n_src, n_chan)), X, W, Cx)

                # Iterative Projection 2
                _ip_double(s1, s2, X, W, r_inv[[s1, s2], :])

        elif update_rule == "ipa":

            # compute all the covariance matrices
            V = [
                (X * r_inv[k, None, None, :]) @ tensor_H(X) / n_frames
                for k in range(n_src)
            ]

            # enforce hermitian symmetry of covariance matrices
            for i, vv in enumerate(V):
                V[i] = 0.5 * (vv + tensor_H(vv))

            for k in range(n_src):
                W[:] = _ipa(V, W, k)

        elif update_rule == "ipancg":
            # compute all the covariance matrices
            V = np.array(
                [
                    (X * r_inv[k, None, None, :]) @ tensor_H(X) / n_frames
                    for k in range(n_src)
                ]
            )

            # enforce hermitian symmetry of covariance matrices
            for i, vv in enumerate(V):
                V[i] = 0.5 * (vv + tensor_H(vv))

            # check the value of the head_error
            error = head_error(V.swapaxes(0, 1), W)
            print(f"HEAD error:", error)
            if error < 1e-8:
                print(f"epoch={epoch} use NCG")
                W[:] = head_update_ncg(V.swapaxes(0, 1), W)
            else:
                for k in range(n_src):
                    W[:] = _ipa(V, W, k)

        elif update_rule == "fullhead":
            # compute all the covariance matrices
            V = np.array(
                [
                    (X * r_inv[k, None, None, :]) @ tensor_H(X) / n_frames
                    for k in range(n_src)
                ]
            )

            # enforce hermitian symmetry of covariance matrices
            for i, vv in enumerate(V):
                V[i] = 0.5 * (vv + tensor_H(vv))

            # now solve head
            W[:], info = head_solver(
                V.swapaxes(0, 1), W=W, method=HEADUpdate.IPA_NCG, tol=1e-1, info=True
            )
            print(info["epochs"])

        elif update_rule == "auxiva-iss":

            assert n_chan == n_src, "ISS is only implemented in the determined case"

            for k in range(n_chan):
                _iss_single(k, X, W, r_inv)

        else:
            raise ValueError("Invalid update rules")

        demix(Y, X, W[:, :n_src, :])

        # Monitor the algorithm progression
        if callback is not None and (epoch + iter_step) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1])
            callback(Y_tmp, model)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y
