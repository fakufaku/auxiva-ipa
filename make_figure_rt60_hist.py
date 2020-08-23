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
This script generates a histogram of the RT60 of the impulse responses used
in the experiments
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates histogram plot of RT60 in experiment"
    )
    parser.add_argument("file", type=str, help="File containing the room information")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        room_data = json.load(f)

    rt60s = np.array(room_data["rt60s"])

    sns.set(
        style="whitegrid",
        context="paper",
        font_scale=0.75,
        rc={
            # 'figure.figsize': (3.39, 3.15),
            "lines.linewidth": 1.0,
            # 'font.family': 'sans-serif',
            # 'font.sans-serif': [u'Helvetica'],
            # 'text.usetex': False,
        },
    )

    # Histogram of RT60
    plt.figure(figsize=(1.5 * 1.675, 1.5 * 1.2))
    plt.hist(rt60s * 1000.0)
    plt.xlabel("$T_{60}$ [ms]")
    plt.ylabel("Frequency")
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks([50, 150, 250, 350, 450])

    plt.tight_layout(pad=0.0)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_fn = f"figures/figure0_rt60_hist.pdf"
    plt.savefig(fig_fn)
