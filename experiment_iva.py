import datetime
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import bss

config = {
    "master_seed": 8856641,
    "n_repeat": 50,
    "params": [
        {"n_freq": 6, "n_chan": 4},
        {"n_freq": 6, "n_chan": 6},
        {"n_freq": 6, "n_chan": 8},
    ],
    "n_frames": 5000,
    "distrib": "laplace",
    "algos": {
        # "iva-ng-0.3": {"algo": "iva-ng", "kwargs": {"step_size": 0.3, "n_iter": 100}},
        # "iva-ng-0.2": {"algo": "iva-ng", "kwargs": {"step_size": 0.2, "n_iter": 100}},
        # "iva-ng-0.1": {"algo": "iva-ng", "kwargs": {"step_size": 0.1, "n_iter": 100}},
        "auxiva": {"algo": "auxiva", "kwargs": {"n_iter": 100}},
        "auxiva2": {"algo": "auxiva2", "kwargs": {"n_iter": 100}},
        "auxiva-iss": {"algo": "auxiva-iss", "kwargs": {"n_iter": 100}},
        # "auxiva-iss2": {"algo": "auxiva-iss2", "kwargs": {"n_iter": 100}},
        # "auxiva-ipa": {"algo": "auxiva-ipa", "kwargs": {"n_iter": 100}},
        "auxiva-ipa2": {"algo": "auxiva-ipa2", "kwargs": {"n_iter": 100}},
        "auxiva-fullhead": {
            "algo": "auxiva-fullhead",
            "kwargs": {"tol": 1e-5, "n_iter": 100},
        },
    },
}


def separation_error(est, mix, ref, ref_ch=0):

    # fix scale
    est_hat = bss.project_back(est.transpose([2, 0, 1]), mix[:, ref_ch, :].T).transpose(
        [1, 2, 0]
    )

    # compute the error matrix
    err_mat = np.mean(
        np.abs(est_hat[:, None, :, :] - ref[:, :, None, :]) ** 2, axis=(0, 3)
    )

    # now find the best permutation
    r, c = linear_sum_assignment(err_mat)
    error = np.mean(err_mat[r, c])

    return error, c, est_hat[:, c, :]


def scale_permute(W, A):

    eye = np.eye(W.shape[-1])

    P = W @ A
    W = W.copy()

    # fix the scale by setting the largest element in each row to 1
    scale = np.max(np.abs(P), axis=-1, keepdims=True)
    W /= scale
    P /= scale

    # now find the permutation
    err_mat = np.mean(
        np.abs(eye[None, :, None, :] - np.abs(P[:, None, :, :])) ** 2, axis=(0, 3)
    )
    _, perm = linear_sum_assignment(err_mat)

    # fix the permutation
    W = W[:, perm, :]
    P = P[:, perm, :]

    return W, P


def ISR(W, A):

    n_freq, n_chan, _ = W.shape

    # restore scale and permutation
    W, P = scale_permute(W, A)

    isr = np.zeros(W.shape)

    for r in range(n_chan):
        for c in range(n_chan):
            if r == c:
                continue
            isr[:, r, c] = np.abs(P[:, r, c]) ** 2 / np.abs(P[:, r, r]) ** 2

    return 10 * np.log10(np.mean(isr))


def one_loop(args):
    # expand the input arguments
    (param_index, n_freq, n_chan, n_frames, distrib, algos, seed,) = args

    # fix the random seed
    np.random.seed(seed)

    # make sure execution is sequential
    import mkl

    mkl.set_num_threads(1)

    mix, ref, mix_mat = bss.random.rand_mixture(
        n_freq, n_chan, n_frames, distrib=distrib, dtype=np.complex128
    )

    isr = {}
    cost = {}

    # used to compute the input ISR
    init_W = np.array([np.eye(n_chan) for f in range(n_freq)])

    for algo, pmt in algos.items():

        algo_name = pmt["algo"]
        algo_kwargs = pmt["kwargs"]

        if bss.is_dual_update[algo_name]:
            callback_checkpoints = np.arange(2, algo_kwargs["n_iter"] + 1, 2)
        else:
            callback_checkpoints = np.arange(1, algo_kwargs["n_iter"] + 1)

        isr_list = []
        cost_list = []

        # init with PCA
        Y_pca, demix_pca = bss.pca(mix.transpose([2, 0, 1]).copy(), return_filters=True)

        def callback(Y, loc_demix, model):
            isr_list.append(ISR(loc_demix @ demix_pca, mix_mat))

            # cost
            cost = np.sum(np.linalg.norm(Y, axis=1))
            _, logdet = np.linalg.slogdet(loc_demix)
            cost -= 2 * Y.shape[0] * np.sum(logdet)
            cost_list.append(cost)

            new_Y = (loc_demix @ mix).transpose([2, 0, 1])

        # ISR of mixture
        callback(mix.transpose([2, 0, 1]), init_W.copy(), distrib)

        # separate with IVA
        est, demix_mat = bss.algos[algo_name](
            # mix.transpose([2, 0, 1]).copy(),
            Y_pca.copy(),
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

    return param_index, isr, cost


def gen_args(master_seed, n_repeat, params, n_frames, distrib, algos):

    np.random.seed(master_seed)

    args = []
    for i, p in enumerate(params):
        for r in range(n_repeat):
            seed = np.random.randint(2 ** 32)
            args.append((i, p["n_freq"], p["n_chan"], n_frames, distrib, algos, seed,))

    return args


if __name__ == "__main__":

    from multiprocessing import Pool

    args = gen_args(**config)

    # run all the simulation in parallel
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
