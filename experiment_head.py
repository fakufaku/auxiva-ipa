import argparse
import datetime
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np

import bss
from bss.head import HEADUpdate

config = {
    "seed": 873009,
    "n_repeat": 1000,
    "n_chan": [4, 6, 8],
    "tol": -1.0,  # i.e. ignore tolerance
    "maxiter": 100,
    "dtype": np.complex128,
}

methods = {
    "IPA": HEADUpdate.IPA,
    "IP": HEADUpdate.IP,
    "ISS": HEADUpdate.ISS,
    "IP2": HEADUpdate.IP2,
    "NCG": HEADUpdate.NCG,
    "IPA+NCG": HEADUpdate.IPA_NCG,
}


def progress_tracker(n_tasks, queue):

    n_digits = len(str(n_tasks))
    fmt = "Remaining tasks: {n:" + str(n_digits) + "d} / " + str(n_tasks)

    start_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Start processing at {start_date}")

    def print_status():
        print(fmt.format(n=n_tasks), end="\r")

    print_status()

    while n_tasks > 0:
        _ = queue.get(block=True)
        n_tasks -= 1
        print_status()

    end_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"All done. Finished at {end_date}")


def f_loop(args):
    import mkl

    mkl.set_num_threads(1)

    # expand arguments
    V, maxiter, tol, methods, the_queue = args

    infos = {}
    runtimes = {}

    v_mat = V[None, ...]

    for method, key in methods.items():
        t_start = time.perf_counter()
        _, info = bss.head.head_solver(
            v_mat, maxiter=maxiter, tol=tol, method=key, verbose=False, info=True
        )
        ellapsed = time.perf_counter() - t_start

        infos[method] = info
        runtimes[method] = ellapsed

    the_queue.put(True)

    return V.shape[-1], infos, runtimes


def rand_V(n_freq, n_chan, n_mat=None, dtype=np.complex128):

    if n_mat is None:
        n_mat = n_chan

    # random hermitian PSD matrices
    X = bss.random.crandn(n_freq, n_mat, n_chan, n_chan)
    w, U = np.linalg.eigh(X)
    w[:] = np.random.rand(*w.shape)
    V = (U * np.abs(w[..., None, :])) @ bss.utils.tensor_H(U)
    V = 0.5 * (V + bss.utils.tensor_H(V))

    X = bss.random.crandn(n_freq, n_mat, n_chan, 10 * n_chan)
    V = X @ bss.utils.tensor_H(X)

    return V


if __name__ == "__main__":

    np.random.seed(config["seed"])

    # we need a queue for inter-process communication
    m = multiprocessing.Manager()
    the_queue = m.Queue()

    # construct the list of parallel arguments
    parallel_args = []
    for n_chan in config["n_chan"]:
        V = rand_V(config["n_repeat"], n_chan, dtype=config["dtype"],)
        for v_mat in V:
            parallel_args.append(
                (v_mat, config["maxiter"], config["tol"], methods, the_queue)
            )
    np.random.shuffle(parallel_args)

    # run all the simulation in parallel
    prog_proc = multiprocessing.Process(
        target=progress_tracker, args=(len(parallel_args), the_queue,)
    )
    prog_proc.start()
    t_start = time.perf_counter()
    pool = multiprocessing.Pool()
    results = pool.map(f_loop, parallel_args)
    pool.close()
    t_end = time.perf_counter()

    infos = dict(zip(config["n_chan"], [{} for c in config["n_chan"]]))
    runtimes = dict(zip(config["n_chan"], [{} for c in config["n_chan"]]))

    # Post process the results
    for (n_chan, res_info, res_rt) in results:

        for method in methods:

            if method not in infos[n_chan]:
                infos[n_chan][method] = {
                    "head_errors": [res_info[method]["head_errors"]],
                    "head_costs": [res_info[method]["head_costs"]],
                }
            else:
                infos[n_chan][method]["head_costs"].append(
                    res_info[method]["head_costs"]
                )
                infos[n_chan][method]["head_errors"].append(
                    res_info[method]["head_errors"]
                )

            if method not in runtimes[n_chan]:
                runtimes[n_chan][method] = res_rt[method]
            else:
                runtimes[n_chan][method] += res_rt[method]

    for n_chan in config["n_chan"]:
        print(f"{n_chan} channels")
        for method in methods:

            runtimes[n_chan][method] /= config["n_repeat"]
            infos[n_chan][method]["head_costs"] = np.concatenate(
                infos[n_chan][method]["head_costs"], axis=0
            )
            infos[n_chan][method]["head_errors"] = np.concatenate(
                infos[n_chan][method]["head_errors"], axis=0
            )

            print(f"=== {method:7s} runtime={runtimes[n_chan][method]:.3f} ===")

    # get the date
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"data/{date}_experiment_head_results.npz"

    # save to compressed numpy file
    np.savez(filename, config=config, methods=methods, infos=infos, runtimes=runtimes)

    print(f"Total time spent {t_end - t_start:.3f} s")
    print(f"Results saved to {filename}")
