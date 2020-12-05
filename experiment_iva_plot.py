import argparse
import os
from pathlib import Path
from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np

import bss

figure_dir = Path("./figures")

title_dict = {
    "iva-ng-0.5": "ng-0.5",
    "iva-ng-0.3": "ng-0.3",
    "iva-ng-0.2": "ng-0.2",
    "iva-ng-0.1": "ng-0.1",
    "auxiva": "ip",
    "auxiva2": "ip2",
    "auxiva-iss": "iss",
    "auxiva-iss2": "iss2",
    "auxiva-ipa": "ipa",
    "auxiva-ipa2": "ipa2",
    "auxiva-fullhead": "fh",
    "auxiva-fullhead_1e-5": "fh",
    "auxiva-fullhead_1e-10": "fh-10",
    "fastiva": "fiva",
}

fail_thresh = -10.0


def make_plot(config, params, isr_tables, cost_tables, filename=None):

    # expand some parameters
    n_freq = params["n_freq"]
    n_chan = params["n_chan"]
    pca = params["pca"]

    # construct the mosaic
    n_algos = len(config["algos"])
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
    axes[map_up[0]].set_ylabel("ISR")
    axes[map_do[0]].set_ylabel("Cost")

    return fig, axes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plots the result of the IVA experiment with synthetic data"
    )
    parser.add_argument("data", type=Path, help="Path to simulation data")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    config = data["config"].tolist()
    isr_tables = data["isr_tables"].tolist()
    cost_tables = data["cost_tables"].tolist()

    os.makedirs(figure_dir, exist_ok=True)

    for p, isr, cost in zip(config["params"], isr_tables, cost_tables):
        fig, axes = make_plot(config, p, isr, cost)
        pca_str = "_pca" if p["pca"] else ""
        filename = figure_dir / (
            args.data.stem + f"_f{p['n_freq']}_c{p['n_chan']}{pca_str}.pdf"
        )
        fig.savefig(filename)

    plt.show()
