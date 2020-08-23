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
Implementation of the SI-SDR toolbox to measure the performance of BSS
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def si_bss_eval(reference_signals, estimated_signals, scaling=True):
    """
    Compute the Scaled Invariant Signal-to-Distortion Ration (SI-SDR) and related
    measures according to [1]_.

    .. [1] J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR - half-baked or well
        done?", 2018, https://arxiv.org/abs/1811.02508

    Parameters
    ----------
    reference_signals: ndarray (n_samples, n_channels)
        The reference clean signals
    estimated_signal: ndarray (n_samples, n_channels)
        The signals to evaluate
    scaling: bool
        Flag that indicates whether we want to use the scale invariant (True)
        or scale dependent (False) method

    Returns
    -------
    SDR: ndarray (n_channels)
        Signal-to-Distortion Ratio
    SIR: ndarray (n_channels)
        Signal-to-Interference Ratio
    SAR: ndarray (n_channels)
        Signal-to-Artefact Ratio
    """

    n_samples, n_chan = estimated_signals.shape

    Rss = np.dot(reference_signals.transpose(), reference_signals)

    SDR = np.zeros((n_chan, n_chan))
    SIR = np.zeros((n_chan, n_chan))
    SAR = np.zeros((n_chan, n_chan))

    for r in range(n_chan):
        for e in range(n_chan):
            SDR[r, e], SIR[r, e], SAR[r, e] = _compute_measures(
                estimated_signals[:, e], reference_signals, Rss, r, scaling=scaling
            )

    dum, p_opt = _linear_sum_assignment_with_inf(-SIR)

    return SDR[dum, p_opt], SIR[dum, p_opt], SAR[dum, p_opt], p_opt


def _compute_measures(estimated_signal, reference_signals, Rss, j, scaling=True):
    """
    Compute the Scale Invariant SDR and other metrics

    This implementation was provided by Johnathan Le Roux
    [here](https://github.com/sigsep/bsseval/issues/3)

    Parameters
    ----------
    estimated_signal: ndarray (n_samples, n_channels)
        The signals to evaluate
    reference_signals: ndarray (n_samples, n_channels)
        The reference clean signals
    Rss: ndarray(n_channels, n_channels)
        The covariance matrix of the reference signals
    j: int
        The index of the reference source to evaluate
    scaling: bool
        Flag that indicates whether we want to use the scale invariant (True)
        or scale dependent (False) method
    """
    n_samples, n_chan = reference_signals.shape

    this_s = reference_signals[:, j]

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss[j, j]
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true ** 2).sum()
    Snn = (e_res ** 2).sum()

    SDR = 10 * math.log10(Sss / Snn)

    # Get the SIR
    Rsr = np.dot(reference_signals.transpose(), e_res)
    b = np.linalg.solve(Rss, Rsr)

    e_interf = np.dot(reference_signals, b)
    e_artif = e_res - e_interf

    SIR = 10 * math.log10(Sss / (e_interf ** 2).sum())
    SAR = 10 * math.log10(Sss / (e_artif ** 2).sum())

    return SDR, SIR, SAR


def _linear_sum_assignment_with_inf(cost_matrix):
    """
    Solves the permutation problem efficiently via the linear sum
    assignment problem.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    This implementation was proposed by @louisabraham in
    https://github.com/scipy/scipy/issues/6900
    to handle infinite entries in the cost matrix.
    """
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        cost_matrix = cost_matrix.copy()
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder

    return linear_sum_assignment(cost_matrix)


if __name__ == "__main__":

    import argparse
    from pathlib import Path

    from scipy.io import wavfile

    parser = argparse.ArgumentParser(
        description="Compute SI-SDR, SI-SIR, and SI-SAR between wav files"
    )
    parser.add_argument(
        "references", type=Path, help="The file corresponding to the reference signals"
    )
    parser.add_argument(
        "signals", type=Path, help="The file corresponding to the signals to evaluate"
    )

    args = parser.parse_args()

    fs, ref = wavfile.read(args.references)
    fs, sig = wavfile.read(args.signals)

    n_samples, n_chan = ref.shape

    sdr, sir, sar, perm = si_bss_eval(ref, sig)
    for s in range(n_chan):
        print(
            f"Source {s+1}: SDR={sdr[s]:0.2f} dB   SIR={sir[s]:0.2f} dB   SAR={sar[s]:0.2f} dB"
        )
