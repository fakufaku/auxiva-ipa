import argparse
import os
from pathlib import Path
from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import bss
from plot_config import seaborn_config

### CONFIG ###
figure_dir = Path("./figures")

title_dict = {
    "iva-ng-0.5": "ng-0.5",
    "iva-ng-0.3": "NG",
    "iva-ng-0.2": "ng-0.2",
    "iva-ng-0.1": "ng-0.1",
    "auxiva": "IP",
    "auxiva2": "IP2",
    "auxiva-iss": "ISS",
    "auxiva-iss2": "ISS2",
    "auxiva-ipa": "IPA",
    "auxiva-ipa2": "IPA2",
    "auxiva-fullhead": "FH",
    "auxiva-fullhead_1e-5": "SeDJoCo",
    "auxiva-fullhead_1e-10": "SeDJoCo",
    "auxiva-fullhead_1e-20": "SeDJoCo",
    "fastiva": "FastIVA",
}

include_algos = [
    "iva-ng-0.3",
    "fastiva",
    "auxiva",
    "auxiva-iss",
    "auxiva2",
    "auxiva-ipa",
    "auxiva-fullhead_1e-20",
]

include_algos_cost = [
    "auxiva",
    "auxiva-iss",
    "auxiva2",
    "auxiva-ipa",
    "auxiva-fullhead_1e-20",
]


fail_thresh = -10.0

# number of bins for histogram
n_bins = 60

# figure size
cm2in = 0.39
fig_width = 17.78  # cm (7 inch)
fig_height = 8  # cm
leg_space = 1.6  # cm
figsize = (fig_width * cm2in, fig_height * cm2in)

# criteria for convergence of cost function
cost_eps_convergence = 1e-3
isr_eps_convergence = 1e-1
fig_width_cost = 8.5
fig_heigh_cost = 4
figsize_cost = (fig_width_cost * cm2in, fig_heigh_cost * cm2in)
### END CONFIG ###

ascii_letters = ascii_letters + "1234567890*#/!?$,[;:"


def make_plot(config, params, isr_tables, cost_tables, filename=None):

    # expand some parameters
    n_freq = params["n_freq"]
    n_chan = params["n_chan"]
    pca = params["pca"]

    # construct the mosaic
    n_algos = len(include_algos)
    assert 2 * n_algos + 2 <= len(ascii_letters)
    mosaic_array = [[ascii_letters[0] * n_algos], [ascii_letters[1] * n_algos]]
    for b in range(2):
        for i in range(1, n_algos + 1):
            mosaic_array[b].append(ascii_letters[2 * i + b])
    mosaic = "\n".join(["".join(a) for a in mosaic_array])

    fig, axes = plt.subplot_mosaic(mosaic)
    map_up = [mosaic_array[0][0][0]] + mosaic_array[0][1:]
    map_do = [mosaic_array[1][0][0]] + mosaic_array[1][1:]

    n_bins = 30

    y_lim_isr = [0, 1]
    y_lim_cost = [0, 1]
    x_lim_hist_isr = [0, 1.0]
    x_lim_hist_cost = [0, 1.0]
    for i, (algo, table) in enumerate(isr_tables.items()):
        n_iter = config["algos"][algo]["kwargs"]["n_iter"]
        algo_name = config["algos"][algo]["algo"]
        if bss.is_dual_update[algo_name]:
            callback_checkpoints = np.arange(0, n_iter + 1, 2)
        else:
            callback_checkpoints = np.arange(0, n_iter + 1)

        if not algo.startswith("fullhead"):
            # isr
            y_lim_isr = [
                min(y_lim_isr[0], table.min()),
                max(y_lim_isr[1], table.max()),
            ]
            # cost
            y_lim_cost = [
                min(y_lim_cost[0], cost_tables[algo].min()),
                max(y_lim_cost[1], np.percentile(cost_tables[algo], 0.9)),
            ]

        I_s = table[:, -1] < fail_thresh  # separation is sucessful
        I_f = table[:, -1] >= fail_thresh  # separation fails

        # ISR convergence
        p = axes[map_up[0]].semilogx(
            np.array(callback_checkpoints) + 1,
            np.mean(table[I_s, :], axis=0),
            label=algo,
        )
        c = p[0].get_color()
        axes[map_up[0]].plot(
            np.array(callback_checkpoints) + 1,
            np.mean(table[I_f, :], axis=0),
            label=algo,
            alpha=0.6,
            c=c,
        )

        # Cost
        axes[map_do[0]].semilogx(
            np.array(callback_checkpoints) + 1,
            np.mean(cost_tables[algo], axis=0),
            label=algo,
        )

        # Histograms
        axes[map_up[i + 1]].hist(
            table[:, -1], bins=n_bins, orientation="horizontal", density=True
        )
        axes[map_do[i + 1]].hist(
            cost_tables[algo][:, -1],
            bins=n_bins,
            orientation="horizontal",
            density=True,
        )

        axes[map_up[i + 1]].set_title(title_dict[algo])
        axes[map_up[i + 1]].set_xlabel("")

    for i in range(len(isr_tables) + 1):
        axes[map_up[i]].set_ylim(y_lim_isr)
        axes[map_do[i]].set_ylim(y_lim_cost)
        if i > 0:
            axes[map_up[i]].set_yticks([])
            axes[map_do[i]].set_yticks([])
            """
            axes[map_up[i]].set_xlim(x_lim_hist_isr)
            axes[map_do[i]].set_xlim(x_lim_hist_cost)
            """

    axes[map_up[0]].legend()
    axes[map_do[0]].legend()
    axes[map_do[0]].set_xlabel("Iteration")
    axes[map_up[0]].set_ylabel("ISR [dB]")
    axes[map_do[0]].set_ylabel("Cost")

    return fig, axes


def make_plot_isr(config, arg_isr_tables, arg_cost_tables, with_pca=True):

    # pick only the desired params and
    params = []
    isr_tables = []
    cost_tables = []
    n_algos = 0
    for p, isr, cost in zip(config["params"], arg_isr_tables, arg_cost_tables):
        if p["pca"] != with_pca:
            continue
        params.append(p)
        cost_tables.append(cost)

        isr_tables.append({})
        n_algos = 0
        for alg_name, alg_dict in isr.items():
            if alg_name in include_algos:
                isr_tables[-1][alg_name] = alg_dict
                n_algos += 1

    # pick the algorithms to include
    """
    new_isr_tables = {}
    for sub_dict in isr_tables:
        n_algos = 0
        rm_list = []
        for algo, d in sub_dict:
            if algo not in include_algos:
                rm_list.append(algo)
            else:
                n_algos += 1
        for a in rm_list:
            sub_dict.pop(a)
    """

    # construct the mosaic
    # n_algos = len(include_algos)
    n_rows = len(params)
    mosaic_array = []
    mosaic_len_left = n_algos // 2
    mosaic_row_len = mosaic_len_left + n_algos
    mos_map = []
    assert n_rows * (n_algos + 1) <= len(ascii_letters)
    for b in range(n_rows):
        mosaic_array.append([ascii_letters[b] * mosaic_len_left])
        mos_map.append([mosaic_array[b][0][0]])
        for i in range(1, n_algos + 1):
            letters = ascii_letters[n_rows * i + b]
            mosaic_array[b].append(letters)
            mos_map[b].append(letters)
    mosaic = "\n".join(["".join(a) for a in mosaic_array])

    # prepare the style
    seaborn_config(n_algos)

    # create the figure
    fig_size = (figsize[0], figsize[1] * len(params) / 3)
    fig, axes = plt.subplot_mosaic(mosaic, figsize=fig_size)

    # container for some info we will fill as we go
    leg_handles = {}
    y_lim_isr = [0, 1]
    x_lim_hist_isr = [0, 0]
    percent_converge = []

    for ip, pmt in enumerate(params):
        n_freq = pmt["n_freq"]
        n_chan = pmt["n_chan"]
        percent_converge.append({})

        for i, algo in enumerate(include_algos):
            if algo not in isr_tables[ip]:
                continue
            table = isr_tables[ip][algo]

            n_iter = config["algos"][algo]["kwargs"]["n_iter"]
            algo_name = config["algos"][algo]["algo"]

            if bss.is_dual_update[algo_name]:
                callback_checkpoints = np.arange(0, n_iter + 1, 2)
            else:
                callback_checkpoints = np.arange(0, n_iter + 1)

            # isr
            y_lim_isr = [
                min(y_lim_isr[0], table.min()),
                max(y_lim_isr[1], np.percentile(table, 99.5)),
            ]

            I_s = table[:, -1] < fail_thresh  # separation is sucessful
            I_f = table[:, -1] >= fail_thresh  # separation fails

            f_agg = np.mean

            # ISR convergence
            p = axes[mos_map[ip][0]].semilogx(
                np.array(callback_checkpoints),
                f_agg(table[I_s, :], axis=0),
                label=title_dict[algo],
            )

            # keep the percentage and mean of success/failure
            percent_converge[ip][algo] = (
                np.sum(I_s) / len(I_s),
                f_agg(table[I_s, -1]),
                f_agg(table[I_f, -1]),
            )

            # get color of main line
            c = p[0].get_color()

            # now draw the divergent line
            axes[mos_map[ip][0]].plot(
                np.array(callback_checkpoints),
                f_agg(table[I_f, :], axis=0),
                alpha=0.6,
                c=c,
                linestyle="--",
            )

            # Histograms
            bin_heights, bins, patches = axes[mos_map[ip][i + 1]].hist(
                table[:, -1],
                bins=n_bins,
                orientation="horizontal",
                density=True,
                color=c,
                linewidth=0.0,
            )

            # keep track of required length of x-axis for the histograms
            x_lim_hist_isr[1] = max(x_lim_hist_isr[1], bin_heights.max())

        # collect the labels
        handles, labels = axes[mos_map[ip][0]].get_legend_handles_labels()
        for lbl, hand in zip(labels, handles):
            if lbl not in leg_handles:
                leg_handles[lbl] = hand

    sns.despine(fig=fig, offset=1.0)

    # arrange the parameters
    for ip, pmt in enumerate(params):
        n_freq = pmt["n_freq"]
        n_chan = pmt["n_chan"]

        # set the x/y-axis limit for all histograms
        if ip == n_rows - 1:
            axes[mos_map[ip][0]].set_xticks([1, 10, 100, 1000])
            axes[mos_map[ip][0]].set_xlim([0.9, 1000])
        else:
            axes[mos_map[ip][0]].set_xticks([])
            axes[mos_map[ip][0]].set_xlim([0.9, 1000])

        axes[mos_map[ip][0]].set_ylim(y_lim_isr)
        for i, algo in enumerate(include_algos):
            if algo not in isr_tables[ip]:
                continue
            table = isr_tables[ip][algo]
            axes[mos_map[ip][i + 1]].set_ylim(y_lim_isr)
            axes[mos_map[ip][i + 1]].set_yticks([])
            axes[mos_map[ip][i + 1]].set_xlim(x_lim_hist_isr)

            if ip == n_rows - 1:
                axes[mos_map[ip][i + 1]].set_xticks([np.mean(x_lim_hist_isr)])
                axes[mos_map[ip][i + 1]].set_xticklabels(
                    [title_dict[algo]]  # , rotation=75
                )
            else:
                axes[mos_map[ip][i + 1]].set_xticks([])

            # write down the percentage of convergent point onto the histogram directly
            p, y_s, y_f = percent_converge[ip][algo]
            # success
            pts = [0.25 * x_lim_hist_isr[1], y_s + 4.0]
            axes[mos_map[ip][i + 1]].annotate(
                f"{100 * p:4.1f}%", pts, fontsize="x-small", ha="left"
            )
            # failure
            pts = [0.25 * x_lim_hist_isr[1], min(y_f + 4.0, y_lim_isr[1] - 1.0)]
            axes[mos_map[ip][i + 1]].annotate(
                f"{100 * (1 - p):4.1f}%", pts, fontsize="x-small", ha="left"
            )

        axes[mos_map[ip][0]].set_title(f"$F={n_freq}$ $M={n_chan}$")
        axes[mos_map[ip][0]].set_ylabel("ISR [dB]")

    axes[mos_map[0][0]].annotate("IP/ISS overlap", [1, -28], fontsize="xx-small")
    axes[mos_map[-1][0]].set_xlabel("Iteration")

    fig.tight_layout(pad=0.1, w_pad=0.2, h_pad=1)
    figleg = fig.legend(
        leg_handles.values(),
        leg_handles.keys(),
        title="Algorithm",
        title_fontsize="x-small",
        fontsize="x-small",
        # bbox_to_anchor=[1 - leg_space / fig_width, 0.5],
        bbox_to_anchor=[1, 0.5],
        loc="center right",
        frameon=False,
    )

    fig.subplots_adjust(right=1 - leg_space / (fig_width))

    return fig, axes


def make_figure_cost(config, arg_isr_tables, arg_cost_tables, with_pca=True):

    # pick only the desired params and
    params = []
    isr_tables = []
    cost_tables = []
    for p, isr, cost in zip(config["params"], arg_isr_tables, arg_cost_tables):
        if p["pca"] != with_pca:
            continue
        params.append(p)
        cost_tables.append(cost)

        isr_tables.append({})
        n_algos = 0
        for alg_name, alg_dict in isr.items():
            if alg_name in include_algos:
                isr_tables[-1][alg_name] = alg_dict
                n_algos += 1

    """
    # pick the algorithms to include
    for sub_dict in isr_tables:
        n_algos = 0
        rm_list = []
        for algo in sub_dict:
            if algo not in include_algos_cost:
                rm_list.append(algo)
            else:
                n_algos += 1
        for a in rm_list:
            sub_dict.pop(a)
    """

    results = []

    # prepare the style
    seaborn_config(n_algos)

    # create the figure
    fig, axes = plt.subplots(1, len(params), figsize=figsize_cost)

    # container for some info we will fill as we go
    leg_handles = {}
    y_lim = [np.inf, -np.inf]

    y_lim_up = -100000
    ticks = [
        [-200000, y_lim_up],
        [-300000, y_lim_up],
        [-400000, y_lim_up],
    ]
    ticklabels = [
        [
            "$-2 x 10^5$",
            "$-10^5$",
        ],
        [
            "$-3 x 10^5$",
            "$-10^5$",
        ],
        [
            "$-4 x 10^5$",
            "$-10^5$",
        ],
    ]

    for ip, pmt in enumerate(params):
        n_freq = pmt["n_freq"]
        n_chan = pmt["n_chan"]

        for i, algo in enumerate(include_algos_cost):
            if algo not in isr_tables[ip]:
                continue
            table = cost_tables[ip][algo]

            agg_cost = np.mean(table, axis=0)

            axes[ip].semilogx(
                np.arange(1, len(agg_cost) + 1), agg_cost, label=title_dict[algo]
            )

            y_lim = [
                min(y_lim[0], agg_cost.min()),
                max(y_lim[1], agg_cost.max()),
            ]

        y_lim[1] = y_lim_up

        y_lim[0] = y_lim[0] - 0.05 * np.diff(y_lim)

        # collect the labels
        handles, labels = axes[ip].get_legend_handles_labels()
        for lbl, hand in zip(labels, handles):
            if lbl not in leg_handles:
                leg_handles[lbl] = hand

        axes[ip].set_xlabel("Iteration")
        axes[ip].set_xticks([1, 10, 100])
        axes[ip].set_title(f"$F={n_freq}$ M={n_chan}")
        axes[ip].set_ylim(y_lim)

        if ip < len(ticks):
            axes[ip].set_yticks(ticks[ip])
        if ip < len(ticklabels):
            axes[ip].set_yticklabels(ticklabels[ip], fontsize=20)
        axes[ip].tick_params(axis="y", labelsize="x-small", rotation=30, pad=-0.1)

    sns.despine(fig=fig, offset=0.1)

    fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=1.0)
    figleg = axes[-1].legend(
        leg_handles.values(),
        leg_handles.keys(),
        title="Algorithm",
        title_fontsize="x-small",
        fontsize="x-small",
        # bbox_to_anchor=[1 - leg_space / fig_width, 0.5],
        # bbox_to_anchor=[1, 0.5],
        bbox_to_anchor=[1.15, 1.1],
        loc="upper right",
        frameon=False,
    )
    # fig.subplots_adjust(right=1 - leg_space / (fig_width_cost))

    return fig, axes


def make_table_cost(config, arg_isr_tables, arg_cost_tables, with_pca=True):

    # pick only the desired params and
    params = []
    isr_tables = []
    cost_tables = []
    for p, isr, cost in zip(config["params"], arg_isr_tables, arg_cost_tables):
        if p["pca"] != with_pca:
            continue
        params.append(p)
        isr_tables.append(isr)
        cost_tables.append(cost)

    # pick the algorithms to include
    for sub_dict in isr_tables:
        n_algos = 0
        rm_list = []
        for algo in sub_dict:
            if algo not in include_algos:
                rm_list.append(algo)
            else:
                n_algos += 1
        for a in rm_list:
            sub_dict.pop(a)

    results = []

    for ip, pmt in enumerate(params):
        n_freq = pmt["n_freq"]
        n_chan = pmt["n_chan"]

        for i, algo in enumerate(include_algos):
            if algo not in isr_tables[ip]:
                continue

            # Cost
            table = cost_tables[ip][algo]
            num_table = np.abs(np.diff(table, axis=1))
            denom_table = np.abs(table[:, 1:] - table[:, :1])
            denom_table = 1
            dtable = num_table / denom_table
            converge_epoch = []
            for r in range(dtable.shape[0]):
                # find first step from the back which is larger than the threshold
                S = np.where((dtable[r] >= 0) & (dtable[r] > cost_eps_convergence))[0]
                if len(S) == 0:
                    converge_epoch.append(len(dtable[r]))
                else:
                    converge_epoch.append(S[-1] + 1)

            # Objective
            mean_objective = np.mean(table[:, -1])

            # ISR
            table = isr_tables[ip][algo]
            dtable = np.abs(np.diff(table, axis=1))
            converge_epoch_isr = []
            for r in range(dtable.shape[0]):
                # find first step from the back which is larger than the threshold
                S = np.where(dtable[r] > isr_eps_convergence)[0]
                if len(S) == 0:
                    converge_epoch_isr.append(len(dtable[r]))
                else:
                    converge_epoch_isr.append(S[-1] + 1)

            results.append(
                {
                    "$F$": n_freq,
                    "$M$": n_chan,
                    "algo": title_dict[algo],
                    "mean_epoch": np.mean(converge_epoch),
                    "median_epoch": int(np.median(converge_epoch)),
                    "max_epoch": np.max(converge_epoch),
                    "median_isr_epoch": int(np.median(converge_epoch_isr)),
                    "obj_val": mean_objective,
                }
            )

    algo_order = [title_dict[algo] for algo in include_algos]

    df = pd.DataFrame(results)

    ret = {}

    """
    # create the pivot table
    df_loc = df[["$F$", "$M$", "algo", "median_isr_epoch", "median_epoch"]].pivot_table(
        columns=["$F$", "$M$"], index=["algo"],  # values=["obj_val"]
    )

    # reorder the columns
    df_loc = df_loc.reindex(index=algo_order)

    # print the stuff
    print("---===---")
    print(metric)
    print()
    print(df_loc)
    print()
    print(df_loc.to_latex(float_format="%.1f"))
    print()
    """
    df_loc = df.pivot_table(index=["algo"], columns=["$F$", "$M$"])
    df_loc = df_loc[["median_epoch", "median_isr_epoch"]]

    # print the stuff
    print("---===---")
    if with_pca:
        print("[with PCA]")
    else:
        print("[without PCA]")
    print("Median epoch (ISR+Cost)")
    print()
    print(df_loc)
    print()
    print(df_loc.to_latex(float_format="%.1f"))
    print()
    print("---===---")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plots the result of the IVA experiment with synthetic data"
    )
    parser.add_argument("data", type=Path, help="Path to simulation data")
    parser.add_argument("--show", action="store_true", help="Show figure")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    config = data["config"].tolist()
    isr_tables = data["isr_tables"].tolist()
    cost_tables = data["cost_tables"].tolist()

    os.makedirs(figure_dir, exist_ok=True)

    for pca in [True]:
        pca_str = "_pca" if pca else ""

        # create the ISR plots
        fig, axes = make_plot_isr(config, isr_tables, cost_tables, with_pca=pca)
        filename = figure_dir / (args.data.stem + f"_isr{pca_str}.pdf")
        fig.savefig(filename)

        # cost figure
        fig, axes = make_figure_cost(config, isr_tables, cost_tables, with_pca=pca)
        filename = figure_dir / (args.data.stem + f"_cost{pca_str}.pdf")
        fig.savefig(filename)

        # compute the values for the cost function
        tables = make_table_cost(config, isr_tables, cost_tables, with_pca=pca)

    for p, isr, cost in zip(config["params"], isr_tables, cost_tables):
        fig, axes = make_plot(config, p, isr, cost)
        pca_str = "_pca" if p["pca"] else ""
        filename = figure_dir / (
            args.data.stem + f"_f{p['n_freq']}_c{p['n_chan']}{pca_str}.pdf"
        )
        fig.savefig(filename)

    if args.show:
        plt.show()
