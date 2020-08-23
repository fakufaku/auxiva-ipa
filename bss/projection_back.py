# Copyright 2020  Robin Scheibler
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
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.
"""
Implementation of Minimum distortion principle to disambiguate the scale of the output signals
"""
import numpy as np


def project_back(Y, ref):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    .. note::

        Despite its name, this function is an implementation of the minimum
        distortion principle described in the two following papers.
        The bad naming comes from the confusion of the author of this code
        (myself) with a different method. m(_ _)m

        - K. Matsuoka and S. Nakashima, “Minimal distortion principle for blind
          source separation,” in Proc. ICA, San Diego, Dec. 2001, pp. 722–727.
        - K.Matsuoka,“Minimal distortion principle for blind source separation,” in
          Proc. SICE, Aug. 2002, pp. 2138–2143.

    Derivation of the projection
    ----------------------------

    The optimal filter `z` minimizes the squared error.

    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    """

    num = np.sum(np.conj(ref[:, :, None]) * Y, axis=0)
    denom = np.sum(np.abs(Y) ** 2, axis=0)
    c = num / np.maximum(1e-15, denom)

    return np.conj(c[None, :, :]) * Y
