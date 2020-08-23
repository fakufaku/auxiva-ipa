# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Core loop of the simulation, simulates one room with several microphones and
sources, separates the sources from the recorded signal and computes the
performance metrics.
"""
# external library imports
import json
import os
import time
import traceback

import numpy as np

import bss
import pyroomacoustics as pra
from metrics import si_bss_eval
# local imports
from room_builder import callback_noise_mixer, convergence_callback
from samples.generate_samples import wav_read_center

# set MKL to only use one thread if present
try:
    import mkl

    mkl.set_num_threads(1)
except ImportError:
    pass


def run(args, parameters):
    """
    This is the core loop of the simulation
    """

    # expand arguments
    sinr, n_targets, n_interf, n_mics, dist_ratio, room_params, seed = args

    n_sources = n_targets + n_interf

    # this is the underdetermined case. We don't do that.
    if n_mics < n_targets:
        return []

    # set the RNG seed
    rng_state = np.random.get_state()
    np.random.seed(seed)

    # get all the signals
    files_absolute = [
        os.path.join(parameters["base_dir"], fn)
        for fn in room_params["wav"][:n_sources]
    ]
    source_signals = wav_read_center(files_absolute, seed=123)

    # create the room
    room = pra.ShoeBox(**room_params["room_kwargs"])
    R = np.array(room_params["mic_array"])
    room.add_microphone_array(pra.MicrophoneArray(R[:, :n_mics], room.fs))
    source_locs = np.array(room_params["sources"])
    for n in range(n_sources):
        room.add_source(source_locs[:, n], signal=source_signals[n, :])

    # compute RIRs and RT60
    room.compute_rir()
    rt60 = np.median(
        [
            pra.experimental.measure_rt60(room.rir[0][n], fs=room.fs)
            for n in range(n_targets)
        ]
    )

    # signals after propagation but before mixing
    # (n_sources, n_mics, n_samples)
    premix = room.simulate(return_premix=True)
    n_samples = premix.shape[-1]

    # create the mix (n_mics, n_samples)
    # this routine will also resize the signals in premix
    mix = callback_noise_mixer(
        premix,
        sinr=sinr,
        n_src=n_targets + n_interf,
        n_tgt=n_targets,
        **parameters["mix_params"]
    )

    # create the reference signals
    # (n_sources + 1, n_samples)
    if n_mics == n_targets:
        refs = np.zeros((n_targets, n_samples))
    else:
        # in the overdetermined case, we add the background as an extra reference
        refs = np.zeros((n_targets + 1, n_samples))
        refs[-1, :] = np.sum(premix[n_targets:, 0, :], axis=0)
    refs[:n_targets, :] = premix[:n_targets, parameters["mix_params"]["ref_mic"], :]

    # STFT parameters
    framesize = parameters["stft_params"]["framesize"]
    hop = parameters["stft_params"]["hop"]
    if parameters["stft_params"]["window"] == "hamming":
        win_a = pra.hamming(framesize)
    else:  # default is Hann
        win_a = pra.hann(framesize)

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_all = pra.transform.analysis(mix.T, framesize, hop, win=win_a)
    X_mics = X_all[:, :, :n_mics]

    # store results in a list, one entry per algorithm
    results = []

    # compute the initial values of SDR/SIR
    init_sdr = []
    init_sir = []

    for full_name, params in parameters["algorithm_kwargs"].items():

        name = params["algo"]
        kwargs = params["kwargs"]

        if not bss.is_determined[name] and bss.is_dual_update[name] and n_targets == 1:
            # Overdetermined algorithms with dual updates cannot be used
            # in the single source case (they can extract at least two sources)
            continue
        elif bss.is_single_source[name] and n_targets > 1:
            # doesn't work for multi source scenario
            continue
        elif bss.is_overdetermined[name] and n_targets == n_mics:
            # don't run the overdetermined stuff in determined case
            continue

        results.append(
            {
                "algorithm": full_name,
                "n_targets": n_targets,
                "n_interferers": n_interf,
                "n_mics": n_mics,
                "rt60": rt60,
                "dist_ratio": dist_ratio,
                "sinr": sinr,
                "seed": seed,
                "sdr": [],
                "sir": [],  # to store the result
                "cost": [],
                "runtime": np.nan,
                "eval_time": np.nan,
                "n_samples": n_samples,
            }
        )

        # this is used to keep track of time spent in the evaluation callback
        eval_time = []

        def cb(Y, source_model):
            convergence_callback(
                Y,
                source_model,
                X_mics,
                n_targets,
                results[-1]["sdr"],
                results[-1]["sir"],
                results[-1]["cost"],
                eval_time,
                refs,
                parameters["mix_params"]["ref_mic"],
                parameters["stft_params"],
                name,
                not bss.is_determined[name],
            )

        if "model" not in kwargs:
            local_model = bss.default.model
        else:
            local_model = kwargs["model"]

        cb(X_mics, local_model)

        try:
            t_start = time.perf_counter()

            bss.separate(
                X_mics,
                n_src=n_targets,
                algorithm=name,
                callback=cb,
                proj_back=False,
                **kwargs
            )

            t_finish = time.perf_counter()

            results[-1]["eval_time"] = np.sum(eval_time)
            results[-1]["runtime"] = t_finish - t_start - results[-1]["eval_time"]

        except Exception:

            # get the traceback
            tb = traceback.format_exc()

            report = {
                "algorithm": name,
                "n_src": n_targets,
                "kwargs": kwargs,
                "result": results[-1],
                "tb": tb,
            }

            pid = os.getpid()
            # report last sdr/sir as np.nan
            results[-1]["sdr"].append(np.nan)
            results[-1]["sir"].append(np.nan)
            # now write the problem to file
            fn_err = os.path.join(
                parameters["_results_dir"], "error_{}.json".format(pid)
            )
            with open(fn_err, "a") as f:
                f.write(json.dumps(report, indent=4))
                f.write(",\n")

            # skip to next iteration
            continue

    # restore RNG former state
    np.random.set_state(rng_state)

    return results
