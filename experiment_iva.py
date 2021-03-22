import datetime
import multiprocessing
import time
from multiprocessing import Pool, Process, Queue

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import bss
# --- HACK ---
# fix the implementation of auxiva-iss used
# to allow monitoring of ISR
from bss.overiva import auxiva_iss

bss.algos["auxiva-iss"] = auxiva_iss
# --- END HACK ---

config = {
    "master_seed": 8856641,
    "n_repeat": 1000,
    "params": [
        {"n_freq": 6, "n_chan": 4, "pca": True},
        {"n_freq": 6, "n_chan": 6, "pca": True},
        {"n_freq": 6, "n_chan": 8, "pca": True},
    ],
    "n_frames": 5000,
    "distrib": "laplace",
    "algos": {
        "iva-ng-0.3": {"algo": "iva-ng", "kwargs": {"step_size": 0.3, "n_iter": 1000}},
        "fastiva": {"algo": "fastiva", "kwargs": {"n_iter": 1000}},
        "auxiva": {"algo": "auxiva", "kwargs": {"n_iter": 1000}},
        "auxiva2": {"algo": "auxiva2", "kwargs": {"n_iter": 1000}},
        "auxiva-iss": {"algo": "auxiva-iss", "kwargs": {"n_iter": 1000}},
        "auxiva-ipa": {"algo": "auxiva-ipa", "kwargs": {"n_iter": 1000}},
        "auxiva-fullhead_1e-20": {
            "algo": "auxiva-fullhead",
            "kwargs": {"tol": 1e-20, "n_iter": 1000},
        },
    },
}


def ISR(W, A):

    n_freq, n_chan, _ = W.shape

    isr = np.zeros(W.shape[1:])

    for m in range(n_chan):
        B = np.abs(W[:, [m], :] @ A) ** 2  # shape: (n_freq, 1, n_chan)
        for m_prime in range(n_chan):
            isr[m, m_prime] = np.mean(
                np.delete(B[:, 0, :], m_prime, axis=1) / B[:, 0, [m_prime]]
            )

    rows, perm = linear_sum_assignment(isr)
    isr_opt = np.mean(isr[rows, perm])

    return 10 * np.log10(isr_opt)


def one_loop(args):
    # expand the input arguments
    (param_index, n_freq, n_chan, use_pca, n_frames, distrib, algos, seed, queue) = args

    # fix the random seed
    np.random.seed(seed)

    # make sure execution is sequential
    import mkl

    mkl.set_num_threads(1)

    # the identity matrix repeated for all frequencies
    eye = np.array([np.eye(n_chan) for f in range(n_freq)])

    mix, ref, mix_mat = bss.random.rand_mixture(
        n_freq, n_chan, n_frames, distrib=distrib, dtype=np.complex128
    )

    if use_pca:
        # init with PCA
        Y_init, demix_init = bss.pca(
            mix.transpose([2, 0, 1]).copy(), return_filters=True
        )
    else:
        # init with identity
        Y_init = mix.transpose([2, 0, 1]).copy()
        demix_init = np.zeros((n_freq, n_chan, n_chan), dtype=mix.dtype)
        demix_init[:] = eye

        # initialization close to groundtruth
        # demix_init = np.linalg.inv(mix_mat) + bss.random.crandn(*mix_mat.shape) * 1e-1
        # Y_init = (demix_init @ mix).transpose([2, 0, 1])

    isr = {}
    cost = {}

    """
    import pdb

    pdb.set_trace()
    """

    for algo, pmt in algos.items():

        algo_name = pmt["algo"]
        algo_kwargs = pmt["kwargs"]

        if bss.is_dual_update[algo_name]:
            callback_checkpoints = np.arange(2, algo_kwargs["n_iter"] + 1, 2)
        else:
            callback_checkpoints = np.arange(1, algo_kwargs["n_iter"] + 1)

        isr_list = []
        cost_list = []

        def callback(Y, loc_demix, model):
            isr_list.append(ISR(loc_demix @ demix_init, mix_mat))

            # cost
            cost = np.sum(np.linalg.norm(Y, axis=1))
            _, logdet = np.linalg.slogdet(loc_demix)
            cost -= 2 * Y.shape[0] * np.sum(logdet)
            cost_list.append(cost)

        # ISR of mixture
        callback(mix.transpose([2, 0, 1]), eye.copy(), distrib)

        # separate with IVA
        est, demix_mat = bss.algos[algo_name](
            # mix.transpose([2, 0, 1]).copy(),
            Y_init.copy(),
            return_filters=True,
            model=distrib,
            callback=callback,
            callback_checkpoints=callback_checkpoints,
            proj_back=False,
            eval_demix_mat=True,
            **algo_kwargs,
        )

        # print(f"{algo} {ISR(demix_mat, mix_mat)}")

        isr[algo] = np.array(isr_list).tolist()
        cost[algo] = np.array(cost_list).tolist()

    queue.put(True)  # push something to indicate that we are done

    return param_index, isr, cost


def gen_args(master_seed, n_repeat, params, n_frames, distrib, algos, queue):

    np.random.seed(master_seed)

    args = []
    for i, p in enumerate(params):
        for r in range(n_repeat):
            seed = np.random.randint(2 ** 32)
            args.append(
                (
                    i,
                    p["n_freq"],
                    p["n_chan"],
                    p["pca"],
                    n_frames,
                    distrib,
                    algos,
                    seed,
                    queue,
                )
            )

    return args


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


if __name__ == "__main__":

    # we need a queue for inter-process communication
    m = multiprocessing.Manager()
    the_queue = m.Queue()

    # generate all the arguments
    args = gen_args(queue=the_queue, **config)
    np.random.shuffle(args)

    # run all the simulation in parallel
    prog_proc = Process(target=progress_tracker, args=(len(args), the_queue,))
    prog_proc.start()
    t_start = time.perf_counter()
    pool = multiprocessing.Pool()
    results = pool.map(one_loop, args)
    pool.close()
    t_end = time.perf_counter()

    print(f"Processing finished in {t_end - t_start} seconds")

    # create structure to collect sim results
    isr_tables = []
    cost_tables = []
    for p in config["params"]:
        isr_tables.append(dict(zip(config["algos"], [[] for a in config["algos"]])))
        cost_tables.append(dict(zip(config["algos"], [[] for a in config["algos"]])))

    # now distribute all the results
    for (pi, isr, cost) in results:
        for algo in config["algos"]:
            isr_tables[pi][algo].append(isr[algo])
            cost_tables[pi][algo].append(cost[algo])

    # save as numpy file
    for pi in range(len(config["params"])):
        for algo in config["algos"]:
            isr_tables[pi][algo] = np.array(isr_tables[pi][algo])
            cost_tables[pi][algo] = np.array(cost_tables[pi][algo])

    # get the date
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"data/{date}_experiment_iva_results.npz"

    # save to compressed numpy file
    np.savez(filename, config=config, isr_tables=isr_tables, cost_tables=cost_tables)
