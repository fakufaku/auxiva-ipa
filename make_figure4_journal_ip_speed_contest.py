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
from plot_config import seaborn_config

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

    df, final_val_tbl, conv_tbl, rt60, parameters = load_data(
        cli_args.dirs, pickle=pickle_flag
    )

    # Draw the figure
    print("Plotting...")

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
        "IVA-NG" + pca_str,
        "FastIVA" + pca_str,
        "AuxIVA-IP" + pca_str,
        "AuxIVA-ISS" + pca_str,
        "AuxIVA-IP2" + pca_str,
        "AuxIVA-IPA" + pca_str,
    ]

    seaborn_config(n_colors=len(all_algos), style="whitegrid")

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_dir = "figures/{}_{}_{}".format(
        parameters["name"], parameters["_date"], parameters["_git_sha"]
    )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    plt_kwargs = {
        "Runtime [s]": {"\u0394SI-SDR [dB]": {}, "\u0394SI-SIR [dB]": {}},
        "Iteration": {"\u0394SI-SDR [dB]": {}, "\u0394SI-SIR [dB]": {}},
    }

    full_width = 6.93  # inches, == 17.6 cm, double column width
    half_width = 3.35  # inches, == 8.5 cm, single column width

    # Second figure
    # Convergence curves: Time/Iteration vs SDR
    cm2in = 0.39
    fig_width = 17.78  # cm (7 inch)

    aspect = 1.5
    height = ((full_width * cm2in) / len(parameters["n_mics"])) * aspect
    n_interferers = 0

    for x_axis in ["Runtime [s]", "Iteration"]:
        for metric in ["\u0394SI-SDR [dB]", "\u0394SI-SIR [dB]"]:
            select = np.logical_and(
                df_agg["metric"] == metric, df_agg["Interferers"] == n_interferers,
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
                row="SINR",
                # row_order=row_order,
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
                margin_titles=True,
            )
            g.map(plt.semilogx, x_axis, "value", markersize=1.5, nonpositive="mask")
            g.set_titles(
                col_template="{col_name} channels", row_template="SNR {row_name} [dB]"
            )

            for ax_row in g.axes:
                ax_row[0].set_ylabel(metric)

            # fix the legend
            leg_handles = {}
            for ax_row in g.axes:
                for ax in ax_row:
                    handles, labels = ax.get_legend_handles_labels()
                    for lbl, hand in zip(labels, handles):
                        if lbl not in leg_handles:
                            if lbl.endswith(" (PCA)"):
                                lbl = lbl[:-6]
                            leg_handles[lbl] = hand

            if x_axis == "Iteration":
                g.set(xticks=[1, 10, 100, 1000])
            elif x_axis == "Runtime [s]":
                g.set(xticks=[0.1, 1, 10])

            plt.tight_layout(pad=0.5, w_pad=2.0, h_pad=2.0)
            g.despine(left=True)
            g.fig.legend(
                leg_handles.values(),
                leg_handles.keys(),
                loc="upper center",
                title="",
                fontsize="x-small",
                frameon=False,
                ncol=len(algo_order),
            )
            g.fig.subplots_adjust(top=0.90)

            # align the y-axis labels
            g.fig.align_ylabels(g.axes[:, 0])

            for ext in ["pdf", "png"]:
                x_inc = "time" if x_axis == "Runtime [s]" else "iterations"
                pca_fn_str = "_pca" if cli_args.pca else ""
                fig_fn = os.path.join(
                    fig_dir,
                    f"figure4_conv_interf{n_interferers}_{x_inc}_{metric}{pca_fn_str}.{ext}",
                )
                plt.savefig(fig_fn, bbox_inches="tight")
        plt.close()

    if plot_flag:
        plt.show()
