import argparse
import time

import numpy as np

import bss
from bss.head import HEADUpdate
from utils import rand_psd

if __name__ == "__main__":
    methods = {
        "IPA": HEADUpdate.IPA,
        "IP": HEADUpdate.IP,
        "ISS": HEADUpdate.ISS,
        "IP2": HEADUpdate.IP2,
        "NCG": HEADUpdate.NCG,
        "IPA+NCG": HEADUpdate.IPA_NCG,
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
    n_chan = 6
    n_samples = 60
    verbose = False
    tol = 1e-8
    maxiter = 300
    # dtype = np.complex128
    dtype = np.float64

    V = bss.random.rand_psd(n_freq, n_chan, n_chan, inner=n_samples, dtype=dtype)

    # Compute the worst condition number
    ev = np.abs(np.linalg.eigvals(V))
    ev_max = np.max(ev, axis=(1, 2))
    ev_min = np.min(ev, axis=(1, 2))
    print("Worst condition number", np.max(ev_max / ev_min))

    for method, key in methods.items():

        t_start = time.perf_counter()

        W, info = bss.head.head_solver(
            V, maxiter=maxiter, tol=tol, method=key, verbose=verbose, info=True
        )

        ellapsed = time.perf_counter() - t_start

        print(
            f"=== {method:7s} runtime={ellapsed:.3f} iterations={info['epochs']:4d} error={info['head_error']:7.5e} "
            f"cost={info['cost']:7.5e} ==="
        )
