import argparse
import datetime
import multiprocessing
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

### CONFIG ###
figure_dir = Path("./figures")

# figure size
cm2in = 0.39
fig_width = 8.5  # cm
fig_height = 6  # cm
leg_space = 1.6  # cm
figsize = (fig_width * cm2in, fig_height * cm2in)

methods_order = [
    "ISS",
    "IP",
    "IP2",
    "IPA",
    "IPA+NCG",
    "NCG",
]
### END CONFIG ###

rcParams["ytick.major.pad"] = "0"


def seaborn_config(n_colors):
    sns.set_theme(context="paper", style="white", font="sans-serif", font_scale=0.75)
    sns.set_palette("viridis", n_colors=7)


def make_figure(
    data, config, methods, infos, runtimes,
):

    seaborn_config(len(methods))

    fig, axes = plt.subplots(2, len(config["n_chan"]), figsize=figsize)

    leg_handles = {}
    for c, n_chan in enumerate(config["n_chan"]):

        for method in methods:
            print(f"=== {method:7s} runtime={runtimes[n_chan][method]:.3f} ===")

        # make the figure
        min_cost = np.inf
        for method in methods:
            loc_min = infos[n_chan][method]["head_costs"].min()
            if loc_min < min_cost:
                min_cost = loc_min
        min_cost = 0

        cost_ylim = [np.inf, -np.inf]
        for method in methods_order:
            key = methods[method]

            axes[0, c].loglog(
                np.median(infos[n_chan][method]["head_errors"], axis=0), label=method
            )
            axes[0, c].set_ylim([1e-32, 100])
            axes[0, c].set_xticks([1, 10, 100, 1000])
            axes[0, c].set_xticklabels(["", "", "", ""])
            if c > 0:
                axes[0, c].set_yticks([])

            # cost
            cost_agg = np.median(infos[n_chan][method]["head_costs"] - min_cost, axis=0)
            if method != "NCG" and method != "IPA+NCG":
                cost_ylim[0] = np.minimum(cost_agg.min(), cost_ylim[0])
                cost_ylim[1] = np.maximum(cost_agg.max(), cost_ylim[1])

            axes[1, c].semilogx(np.arange(len(cost_agg)) + 1, cost_agg, label=method)
            axes[1, c].set_xticks([1, 10, 100, 1000])
            axes[1, c].yaxis.labelpad = 1

            # X axis limits
            # axes[0, c].set_xlim([1, 1000])
            # axes[1, c].set_xlim([1, 1000])
            # Y axis labels
            axes[0, c].set_title(f"$M={n_chan}$")
            axes[1, c].set_xlabel("Iteration")
            if c == 0:
                axes[0, c].set_ylabel("SeDJoCo Residual")
                axes[1, c].set_ylabel("SeDJoCo Cost")

            # keep track of the legend
            handles, labels = axes[0, c].get_legend_handles_labels()
            for lbl, hand in zip(labels, handles):
                if lbl not in leg_handles:
                    if lbl.endswith(" (PCA)"):
                        lbl = lbl[:-6]
                    leg_handles[lbl] = hand

        cost_ylim = np.array(cost_ylim)
        cost_ylim_m = 0.90 * cost_ylim[0] + 0.10 * cost_ylim[1]
        cost_ylim = cost_ylim_m + np.r_[1.05, 0.0] * (cost_ylim - cost_ylim_m)
        axes[1, c].set_ylim(cost_ylim)

    sns.despine(fig=fig)

    fig.tight_layout(pad=0.1)

    fig.legend(
        leg_handles.values(),
        leg_handles.keys(),
        fontsize="x-small",
        loc="upper center",
        bbox_to_anchor=[0.5, 1.01],
        ncol=len(methods_order),
        frameon=False,
    )
    fig.subplots_adjust(top=0.86)

    # fig.align_ylabels(axes[:, 0])

    for j in range(2):
        axes[j, 0].yaxis.set_label_coords(-0.41, 0.5)

    return fig, axes


def make_table(
    data, config, methods, infos, runtimes,
):

    n_iters = [0, 1, 2, 3]
    ref_method = "IPA"
    assert ref_method in methods

    res = []

    for c, n_chan in enumerate(config["n_chan"]):

        cost_table_ref = infos[n_chan][ref_method]["head_costs"]

        for n in n_iters:

            cost_progress_ref = cost_table_ref[:, 0] - cost_table_ref[:, n]

            for method in methods_order:
                key = methods[method]

                cost_table = infos[n_chan][method]["head_costs"]
                cost_progress = cost_table[:, 0] - cost_table[:, n]

                ratio_percent = np.mean(cost_progress / cost_progress_ref) * 100
                # ratio_percent = np.median(cost_table[:, n])

                res.append(
                    {"n_chan": n_chan, "n": n, "algo": method, "ratio": ratio_percent}
                )

    df = pd.DataFrame(res)
    pt = df.pivot_table(columns=["n_chan", "n"], index="algo")
    print()
    print("%### START TABLE ###")
    print(pt.to_latex(float_format="{:.0f}%".format), end="")
    print("%### END TABLE ###")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plots the result of the HEAD experiment with synthetic data"
    )
    parser.add_argument("data", type=Path, help="Path to simulation data")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    config = data["config"].tolist()
    methods = data["methods"].tolist()
    infos = data["infos"].tolist()
    runtimes = data["runtimes"].tolist()

    os.makedirs(figure_dir, exist_ok=True)

    # sort out methods
    for i, m in enumerate(methods_order):
        if m not in methods:
            methods_order.pop(i)

    # create the figure
    fig, axies = make_figure(data, config, methods, infos, runtimes)
    filename = figure_dir / (args.data.stem + ".pdf")
    fig.savefig(filename)

    # create the table
    df = make_table(data, config, methods, infos, runtimes)

    plt.show()
