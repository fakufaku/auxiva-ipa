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
This package contains the main algorithms for determined and overdetermined
independent vector analysis
"""
from . import default, head, utils
from .auxiva_iss import auxiva_iss
from .auxiva_pca import auxiva_pca
from .five import five
from .ogive import ogive, ogive_demix, ogive_mix, ogive_switch
from .overiva import (auxiva, auxiva2, auxiva_fullhead, auxiva_ipa,
                      auxiva_ipancg, overiva, overiva_demix_bg,
                      overiva_ip2_block, overiva_ip2_param, overiva_ip_block,
                      overiva_ip_param)
from .pca import pca
from .projection_back import project_back
from .utils import cost_iva

algos = {
    "auxiva": auxiva,
    "auxiva2": auxiva2,
    "auxiva-iss": auxiva_iss,
    "overiva": overiva,
    "overiva-ip": overiva_ip_param,
    "overiva-ip2": overiva_ip2_param,
    "overiva-ip-block": overiva_ip_block,
    "overiva-ip2-block": overiva_ip2_block,
    "overiva-demix-bg": overiva_demix_bg,
    "five": five,
    "ogive": ogive,
    "ogive-mix": ogive_mix,
    "ogive-demix": ogive_demix,
    "ogive-switch": ogive_switch,
    "auxiva-pca": auxiva_pca,
    "auxiva-ipa": auxiva_ipa,
    "auxiva-ipancg": auxiva_ipancg,
    "auxiva-fullhead": auxiva_fullhead,
    "pca": pca,
}


algos_init = {
    "pca": pca,
}


def separate(
    X,
    algorithm="overiva",
    proj_back=True,
    return_filters=False,
    init=None,
    init_kwargs={},
    **kwargs
):

    n_frames, n_freq, n_chan = X.shape

    if init is not None:
        X0, W0 = algos_init[init](X, return_filters=True, **init_kwargs)
    else:
        X0 = X
        W0 = None

    Y, W = algos[algorithm](X0, proj_back=False, return_filters=True, **kwargs)

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        if W0 is not None:
            W = W @ W0
        return Y, W
    else:
        return Y


is_single_source = {
    "auxiva": False,
    "auxiva2": False,
    "auxiva-iss": False,
    "auxiva-ipa": False,
    "auxiva-ipancg": False,
    "auxiva-fullhead": False,
    "overiva": False,
    "overiva-ip": False,
    "overiva-ip2": False,
    "overiva-ip-block": False,
    "overiva-ip2-block": False,
    "overiva-demix-bg": False,
    "auxiva-pca": False,
    "pca": False,
    "five": True,
    "ogive": True,
    "ogive-mix": True,
    "ogive-demix": True,
    "ogive-switch": True,
}

# This is a list that indicates which algorithms
# can only work with two or more sources
is_dual_update = {
    "auxiva": False,
    "auxiva2": True,
    "auxiva-iss": False,
    "auxiva-ipa": True,
    "auxiva-ipancg": True,
    "auxiva-fullhead": True,
    "overiva": False,
    "overiva-ip": False,
    "overiva-ip2": True,
    "overiva-ip-block": False,
    "overiva-ip2-block": True,
    "overiva-demix-bg": False,
    "auxiva-pca": True,
    "pca": False,
    "five": False,
    "ogive": False,
    "ogive-mix": False,
    "ogive-demix": False,
    "ogive-switch": False,
}

is_determined = {
    "auxiva": True,
    "auxiva2": True,
    "auxiva-iss": True,
    "auxiva-ipa": True,
    "auxiva-ipancg": True,
    "auxiva-fullhead": True,
    "overiva": False,
    "overiva-ip": False,
    "overiva-ip2": False,
    "overiva-ip-block": False,
    "overiva-ip2-block": False,
    "overiva-demix-bg": False,
    "auxiva-pca": False,
    "pca": False,
    "five": False,
    "ogive": False,
    "ogive-mix": False,
    "ogive-demix": False,
    "ogive-switch": False,
}

is_overdetermined = {
    "auxiva": False,
    "auxiva2": False,
    "auxiva-iss": False,
    "auxiva-ipa": False,
    "auxiva-ipancg": False,
    "auxiva-fullhead": False,
    "overiva": True,
    "overiva-ip": True,
    "overiva-ip2": True,
    "overiva-ip-block": True,
    "overiva-ip2-block": True,
    "overiva-demix-bg": True,
    "auxiva-pca": True,
    "pca": False,
    "five": True,
    "ogive": True,
    "ogive-mix": True,
    "ogive-demix": True,
    "ogive-switch": True,
}

is_iterative = {
    "auxiva": True,
    "auxiva2": True,
    "auxiva-iss": True,
    "auxiva-ipa": True,
    "auxiva-ipancg": True,
    "auxiva-fullhead": True,
    "overiva": True,
    "overiva-ip": True,
    "overiva-ip2": True,
    "overiva-ip-block": True,
    "overiva-ip2-block": True,
    "overiva-demix-bg": True,
    "auxiva-pca": True,
    "pca": False,
    "five": True,
    "ogive": True,
    "ogive-mix": True,
    "ogive-demix": True,
    "ogive-switch": True,
}
