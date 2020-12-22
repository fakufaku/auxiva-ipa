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
This script generates figure 3 in the paper
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

    df_all_iters, final_val_tbl, conv_tbl, rt60, parameters = load_data(
        cli_args.dirs, pickle=pickle_flag
    )

    # in this script, we only care about the final values
    df = final_val_tbl

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
        "improvements": {
            "yticks": [[-10, 0, 10, 20], [-10, 0, 10, 20], [0, 10, 20, 30]],
        },
        "raw": {
            "yticks": [[-20, -10, 0, 10, 20], [-10, 0, 10, 20], [-10, 0, 10, 20, 30]],
        },
    }

    full_width = 6.93  # inches, == 17.6 cm, double column width
    half_width = 3.35  # inches, == 8.5 cm, single column width

    # Third figure
    # Classic # of microphones vs metric (box-plots ?)
    the_metrics = {
        "improvements": ["\u0394SI-SDR [dB]", "\u0394SI-SIR [dB]"],
        "raw": ["SI-SDR [dB]", "SI-SIR [dB]"],
    }

    # width = aspect * height
    aspect = 1.2  # width / height
    height = (half_width / 2) / aspect
    # height = 1.6

    iteration_index = 12
    n_interferers = 0

    # first we need to pair the algorithm with their maximum number of iterations

    for m_name in the_metrics.keys():

        metric = the_metrics[m_name]

        select = (
            # np.logical_or(df_melt["Iteration"] == 100, df_melt["Iteration"] == 2000) &
            (df_melt["Interferers"] == n_interferers)
            & df_melt.metric.isin(metric)
        )

        fig = plt.figure()
        g = sns.catplot(
            data=df_melt[select],
            x="Mics",
            y="value",
            hue="Algorithm",
            row="SINR",
            col="metric",
            col_order=metric,
            hue_order=all_algos,
            kind="box",
            legend=False,
            aspect=aspect,
            height=height,
            linewidth=0.5,
            fliersize=0.3,
            # whis=np.inf,
            sharey="row",
            # size=3, aspect=0.65,
            margin_titles=True,
        )

        g.set(clip_on=False)
        # remove original titles before adding custom ones
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(col_template="{col_name}", row_template="SNR {row_name} dB")
        g.set_ylabels("Decibels")
        g.set_xlabels("# channels")

        # remove the white background on the margin titles on the right
        for the_ax in g.axes.flat:
            plt.setp(the_ax.texts, bbox=dict(alpha=0.0))  # , fontsize="large")
            plt.setp(the_ax.title, bbox=dict(alpha=0.0))  # , fontsize="large")

        all_artists = []

        leg_handles = {}
        for r in range(3):
            for c, _ in enumerate(metric):
                if m_name in plt_kwargs and r < len(plt_kwargs[m_name]["yticks"]):
                    g.facet_axis(r, c).set_yticks(plt_kwargs[m_name]["yticks"][r])

                handles, labels = g.facet_axis(r, c).get_legend_handles_labels()
                for lbl, hand in zip(labels, handles):
                    if lbl not in leg_handles:
                        if lbl.endswith(" (PCA)"):
                            lbl = lbl[:-6]
                        leg_handles[lbl] = hand

        sns.despine(offset=10, trim=False, left=True, bottom=True)
        g.fig.tight_layout()

        left_ax = g.facet_axis(0, 0)
        leg = g.fig.legend(
            leg_handles.values(),
            leg_handles.keys(),
            title="",
            frameon=False,
            framealpha=0.85,
            fontsize="xx-small",
            loc="upper center",
            bbox_to_anchor=[0.5, 1.01],
            ncol=len(all_algos),
        )
        # leg.get_frame().set_linewidth(0.0)
        all_artists.append(leg)
        g.fig.align_ylabels()

        g.fig.subplots_adjust(top=0.90)

        """
        for c, lbl in enumerate(metric):
            g_ax = g.facet_axis(0, c)
            g_ax.set_ylabel(lbl)
        """

        for ext in ["pdf", "png"]:
            pca_fn_str = "_pca" if cli_args.pca else ""
            fig_fn = os.path.join(
                fig_dir, f"figure3_{m_name}_interf{n_interferers}{pca_fn_str}.{ext}"
            )
            plt.savefig(
                fig_fn, bbox_extra_artists=all_artists
            )  # , bbox_inches="tight")
        plt.close()

    if plot_flag:
        plt.show()
