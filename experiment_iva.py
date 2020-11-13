import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import bss


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


if __name__ == "__main__":
    n_repeat = 1

    n_freq = 10
    n_chan = 6
    n_frames = 5000
    distrib = "laplace"
    algos = ["auxiva", "auxiva2", "auxiva-ipa", "auxiva-fullhead"]

    n_iter = 30
    init_W = np.array([np.eye(n_chan) for f in range(n_freq)])

    isr_tables = {}
    cost_tables = {}
    for algo in algos:
        if bss.is_dual_update[algo]:
            isr_tables[algo] = np.zeros((n_repeat, n_iter // 2 + 1))
            cost_tables[algo] = np.zeros((n_repeat, n_iter // 2 + 1))
        else:
            isr_tables[algo] = np.zeros((n_repeat, n_iter + 1))
            cost_tables[algo] = np.zeros((n_repeat, n_iter + 1))

    for epoch in range(n_repeat):

        mix, ref, mix_mat = bss.random.rand_mixture(
            n_freq, n_chan, n_frames, distrib=distrib
        )

        for algo in algos:

            if bss.is_dual_update[algo]:
                callback_checkpoints = np.arange(2, n_iter + 1, 2)
            else:
                callback_checkpoints = np.arange(1, n_iter + 1)

            isr_list = []
            cost_list = []

            def callback(Y, W, model):
                isr_list.append(ISR(W, mix_mat))
                cost_list.append(bss.cost_iva(mix.transpose([2, 0, 1]), Y, model))

            # ISR of mixture
            callback(mix.transpose([2, 0, 1]), init_W.copy(), distrib)

            # separate with IVA
            est, demix_mat = bss.algos[algo](
                mix.transpose([2, 0, 1]),
                n_iter=n_iter,
                return_filters=True,
                model=distrib,
                tol=1e-1,
                callback=callback,
                callback_checkpoints=callback_checkpoints,
                eval_demix_mat=True,
            )

            print(f"{algo} {ISR(demix_mat, mix_mat)}")

            isr_tables[algo][epoch, :] = np.array(isr_list)
            cost_tables[algo][epoch, :] = np.array(cost_list)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for algo, table in isr_tables.items():
        if bss.is_dual_update[algo]:
            callback_checkpoints = np.arange(0, n_iter + 1, 2)
        else:
            callback_checkpoints = np.arange(0, n_iter + 1)

        ax1.plot(callback_checkpoints, table.mean(axis=0), label=algo)
        ax2.plot(callback_checkpoints, cost_tables[algo].mean(axis=0), label=algo)
    fig.legend()
    plt.show()
