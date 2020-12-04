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
This script creates figure 4 from the paper.
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from data_loader import load_data

matplotlib.rc("pdf", fonttype=42)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Plot the data simulated by separake_near_wall"
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Read the aggregated data table from a pickle cache",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Display the plots at the end of data analysis",
    )
    parser.add_argument(
        "--pca", action="store_true", help="Plot results for PCA initialization",
    )
    parser.add_argument(
        "dirs",
        type=str,
        nargs="+",
        metavar="DIR",
        help="The directory containing the simulation output files.",
    )

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    df, rt60, parameters = load_data(cli_args.dirs, pickle=pickle_flag)

    # Draw the figure
    print("Plotting...")

    # sns.set(style='whitegrid')
    # sns.plotting_context(context='poster', font_scale=2.)
    # pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    df_melt = df.melt(id_vars=df.columns[:-5], var_name="metric")
    # df_melt = df_melt.replace(substitutions)

    # Aggregate the convergence curves
    df_agg = (
        df_melt.groupby(
            by=[
                "Algorithm",
                "Sources",
                "Interferers",
                "SINR",
                "Mics",
                "Iteration",
                "metric",
            ]
        )
        .mean()
        .reset_index()
    )

    if cli_args.pca:
        pca_str = " (PCA)"
    else:
        pca_str = ""

    all_algos = [
        "AuxIVA-IP" + pca_str,
        "AuxIVA-ISS" + pca_str,
        "AuxIVA-IP2" + pca_str,
        "AuxIVA-IPA" + pca_str,
        "AuxIVA-IPA2" + pca_str,
        "FastIVA" + pca_str,
        "NG" + pca_str,
    ]

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
    pal = sns.cubehelix_palette(
        4, start=0.5, rot=-0.5, dark=0.3, light=0.75, reverse=True, hue=1.0
    )
    sns.set_palette(pal)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_dir = "figures/{}_{}_{}".format(
        parameters["name"], parameters["_date"], parameters["_git_sha"]
    )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    plt_kwargs = {
        # "improvements": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        # "raw": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        # "runtime": {"ylim": [-0.5, 40.5], "yticks": [0, 10, 20, 30]},
        1: {
            "xlim": [
                [-0.05, 0.2],
                [-0.05, 0.3],
                [-0.05, 0.5],
                [-0.05, 1.0],
                [-0.05, 1.5],
            ],
            "xticks": [
                [0.0, 0.1, 0.2],
                [0.0, 0.15, 0.3],
                [0.0, 0.25, 0.5],
                [0.0, 0.5, 1.0],
                [0.0, 0.75, 1.5],
            ],
            "ylim": [[0, 5], [0, 14]],
        },
        2: {
            "xlim": [
                [-0.05, 0.4],
                [-0.05, 0.7],
                [-0.05, 1.2],
                [-0.05, 2.0],
                [-0.05, 4.0],
            ],
            "xticks": [
                [0.0, 0.2, 0.4],
                [0.0, 0.35, 0.7],
                [0.0, 0.6, 1.2],
                [0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0],
            ],
            "ylim": [[-2, 10], [0, 18]],
        },
        3: {
            "xlim": [[-0.05, 1.5], [-0.05, 3.0], [-0.05, 5.0], [-0.05, 8.0]],
            "xticks": [
                [0.0, 0.5, 1.0, 1.5],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.5, 3.5, 5.0],
                [0.0, 2.0, 4.0, 6.0, 8.0],
            ],
            "ylim": [[-2, 9], [0, 16]],
        },
        4: {
            "xlim": [[-0.05, 3.0], [-0.05, 6.0], [-0.05, 10.0]],
            "xticks": [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 2.0, 4.0, 6.0],
                [0.0, 2.5, 5.0, 7.5, 10.0],
            ],
            "ylim": [[-4, 5], [0, 13]],
        },
    }

    full_width = 6.93  # inches, == 17.6 cm, double column width
    half_width = 3.35  # inches, == 8.5 cm, single column width

    # Second figure
    # Convergence curves: Time/Iteration vs SDR
    aspect = 1.2
    # height = ((full_width - 0.8) / len(parameters["sinr"])) / aspect
    height = 1.2
    n_interferers = 0

    for x_axis in ["Runtime [s]", "Iteration"]:
        for sinr in parameters["sinr"]:

            select = np.logical_and(
                df_agg["SINR"] == sinr, df_agg["Interferers"] == n_interferers
            )

            # row_order = ["\u0394SI-SDR [dB]", "\u0394SI-SIR [dB]"]
            # row_order = ["SI-SDR [dB]", "SI-SIR [dB]"]
            row_order = ["\u0394SI-SIR [dB]"]

            local_algo = df_agg[select]["Algorithm"].unique()
            algo_order = [a for a in all_algos if a in local_algo]
            n_mics_list = [n for n in parameters["n_mics"]]

            # select = np.logical_and(df_agg["Interferers"] == 5, select)
            g = sns.FacetGrid(
                df_agg[select],
                row="metric",
                row_order=row_order,
                col="Mics",
                hue="Algorithm",
                hue_order=algo_order,
                hue_kws=dict(
                    # marker=["o", "o", "s", "s", "d", "d", "^", "^"],
                    # linewidth=[1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
                ),
                aspect=aspect,
                height=height,
                sharex="col",
                sharey="row",
            )
            g.map(plt.plot, x_axis, "value", markersize=1.5)
            g.set_titles("{col_name} Sources/Mics")

            for r, lbl in enumerate(row_order):
                g.facet_axis(r, 0).set_ylabel(lbl)

            plt.tight_layout(pad=0.5, w_pad=2.0, h_pad=2.0)
            g.despine(left=True).add_legend(title="", fontsize="x-small")

            """
            for r in range(len(row_order)):
                for c in range(len(n_mics_list)):
                    if c not in plt_kwargs:
                        continue
                    g.axes[r][c].set_xlim(plt_kwargs[c]["xlim"][c])
                    g.axes[r][c].set_ylim(plt_kwargs[c]["ylim"][r])
                    g.axes[r][c].set_xticks(plt_kwargs[c]["xticks"][c])
                    g.axes[r][c].grid(False, axis="x")
                    if r == 0:
                        g.axes[r][c].yaxis.set_major_locator(MaxNLocator(integer=True))
            """

            # align the y-axis labels
            g.fig.align_ylabels(g.axes[:, 0])

            for ext in ["pdf", "png"]:
                x_inc = "time" if x_axis == "Runtime [s]" else "iterations"
                pca_fn_str = "_pca" if cli_args.pca else ""
                fig_fn = os.path.join(
                    fig_dir,
                    f"figure4_conv_interf{n_interferers}_{x_inc}_sinr{sinr}{pca_fn_str}.{ext}",
                )
                plt.savefig(fig_fn, bbox_inches="tight")
            plt.close()

    if plot_flag:
        plt.show()
