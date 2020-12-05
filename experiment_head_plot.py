import argparse
import datetime
import multiprocessing
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

figure_dir = Path("./figures")

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

    for n_chan in config["n_chan"]:
        print(f"{n_chan} channels")
        for method in methods:

            print(f"=== {method:7s} runtime={runtimes[n_chan][method]:.3f} ===")

        # make the figure
        min_cost = np.inf
        for method in methods:
            loc_min = infos[n_chan][method]["head_costs"].min()
            if loc_min < min_cost:
                min_cost = loc_min

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax3_ylim = [0, 1]
        for method, key in methods.items():
            ax1.loglog(
                np.mean(infos[n_chan][method]["head_errors"], axis=0), label=method
            )
            ax1.set_title("Mean HEAD error")

            ax2.loglog(
                np.median(infos[n_chan][method]["head_errors"], axis=0), label=method
            )
            ax2.set_title("Median HEAD error")

            # cost
            cost_mean = np.mean(infos[n_chan][method]["head_costs"] - min_cost, axis=0)
            if method != "NCG" and method != "IPA+NCG":
                ax3_ylim[0] = np.minimum(cost_mean.min(), ax3_ylim[0])
                ax3_ylim[1] = np.maximum(cost_mean.max(), ax3_ylim[0])

            ax3.semilogx(
                np.arange(len(cost_mean)) + 1, cost_mean ** (0.1), label=method
            )
            ax3.set_title("Mean HEAD cost")

        # ax3_ylim[0] = ax3_ylim[0] - 0.01 * np.diff(ax3_ylim)[0]
        ax3_ylim = np.array(ax3_ylim) ** (0.1)

        ax3.set_ylim(ax3_ylim)
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
    plt.show()
