import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import bss
from bss.head import HEADUpdate

if __name__ == "__main__":
    methods = {
        "IPA": HEADUpdate.IPA,
        "IP": HEADUpdate.IP,
        "ISS": HEADUpdate.ISS,
        "IP2": HEADUpdate.IP2,
        "NCG": HEADUpdate.NCG,
        "IPA+NCG": HEADUpdate.IPA_NCG,
        # "FP1": HEADUpdate.FP1,
    }

    parser = argparse.ArgumentParser(description="Experiment of solving HEAD problem")
    parser.add_argument("--seed", type=int, help="Specify the seed for the RNG")
    parser.add_argument(
        "--method", "-m", choices=list(methods.keys()), help="Method used to solve HEAD"
    )
    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(2 ** 32)
    np.random.seed(seed)
    print(f"Initializing the RNG with {seed}")

    n_freq = 10
    n_chan = 4
    n_samples = 2 * n_chan
    verbose = False
    tol = -1.0  # i.e. ignore tolerance
    maxiter = 1000
    dtype = np.complex128
    # dtype = np.float64

    V = bss.random.rand_psd(n_freq, n_chan, n_chan, inner=n_samples, dtype=dtype)

    # Compute the worst condition number
    ev = np.abs(np.linalg.eigvals(V))
    ev_max = np.max(ev, axis=(1, 2))
    ev_min = np.min(ev, axis=(1, 2))
    print("Worst condition number", np.max(ev_max / ev_min))

    infos = {}
    runtimes = {}

    for method, key in methods.items():

        runtimes[method] = 0.0
        for f in range(n_freq):

            t_start = time.perf_counter()

            W, info = bss.head.head_solver(
                V[[f]], maxiter=maxiter, tol=tol, method=key, verbose=verbose, info=True
            )

            ellapsed = time.perf_counter() - t_start

            if method not in infos:
                infos[method] = {
                    "head_errors": [info["head_errors"]],
                    "head_costs": [info["head_costs"]],
                }
            else:
                infos[method]["head_costs"].append(info["head_costs"])
                infos[method]["head_errors"].append(info["head_errors"])

            runtimes[method] += ellapsed

        runtimes[method] /= n_freq
        infos[method]["head_costs"] = np.concatenate(
            infos[method]["head_costs"], axis=0
        )
        infos[method]["head_errors"] = np.concatenate(
            infos[method]["head_errors"], axis=0
        )

        print(f"=== {method:7s} runtime={runtimes[method]:.3f} ===")

    # make the figure
    min_cost = np.inf
    for method in methods:
        loc_min = infos[method]["head_costs"].min()
        if loc_min < min_cost:
            min_cost = loc_min

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax3_ylim = [0, 1]
    for method, key in methods.items():
        ax1.semilogy(np.mean(infos[method]["head_errors"], axis=0), label=method)
        ax1.set_title("Mean HEAD error")

        ax2.semilogy(np.median(infos[method]["head_errors"], axis=0), label=method)
        ax2.set_title("Median HEAD error")

        # cost
        cost_mean = np.mean(infos[method]["head_costs"] - min_cost, axis=0)
        if method != "NCG" and method != "IPA+NCG":
            ax3_ylim[0] = np.minimum(cost_mean.min(), ax3_ylim[0])
            ax3_ylim[1] = np.maximum(cost_mean.max(), ax3_ylim[0])

        ax3.plot(cost_mean ** (0.1), label=method)
        ax3.set_title("Mean HEAD cost")

    # ax3_ylim[0] = ax3_ylim[0] - 0.01 * np.diff(ax3_ylim)[0]
    ax3_ylim = np.array(ax3_ylim) ** (0.1)

    ax3.set_ylim(ax3_ylim)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()
