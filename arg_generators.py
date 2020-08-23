# This code generates the parallel simulation arguments from a configuration file
#
# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import json
import random
import time
from pathlib import Path

import numpy as np

import pyroomacoustics as pra
from pyroomacoustics.bss.common import projection_back
from room_builder import (callback_noise_mixer, choose_target_locations,
                          convergence_callback, random_locations,
                          random_room_definition)
from samples.generate_samples import sampling


def exp1_gen_args(config):

    np.random.seed(config["seed"])

    # sub-seeds
    sub_seeds = []
    for r in range(config["repeat"]):
        sub_seeds.append(int(np.random.randint(2 ** 32)))

    # maximum number of sources and microphones in the simulation
    if config["n_targets"] == "determined":
        n_sources = np.max(config["n_mics"])
    else:
        n_sources = np.max(config["n_interferers"]) + np.max(config["n_targets"])
    n_mics = np.max(config["n_mics"])

    room_file = Path(config["room_cache_file"])

    regenerate_rooms = True  # assume we need to start over

    # now check the content of the cache if it exists
    if room_file.exists():

        with open(room_file, "r") as f:
            room_cache = json.load(f)

        # if the content of the config file changes, we should generate again
        if not (
            room_cache["seed"] != config["seed"]
            or room_cache["room_params"] != config["room_params"]
            or len(room_cache["rooms"]) != config["repeat"]
        ):
            regenerate_rooms = False

    if not regenerate_rooms:
        # use the content of the cache
        print("Use the rooms in the cache")
        rooms = room_cache["rooms"]
        rt60s = room_cache["rt60s"]

    else:
        # generate all the rooms
        print("Generate the rooms and measure rt60")

        # choose all the files in advance
        gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
        all_wav_files = sampling(
            config["repeat"],
            n_sources,
            config["samples_list"],
            gender_balanced=True,
            seed=gen_files_seed,
        )

        # generates all the rooms in advance too
        rooms = []
        rt60s = []
        for i in range(config["repeat"]):
            room_params, rt60 = random_room_definition(
                n_sources, n_mics, seed=i, **config["room_params"]
            )
            # add the speech signal files to use
            room_params["wav"] = all_wav_files[r]

            rt60s.append(rt60)
            rooms.append(room_params)

            # cache the rooms
            with open(room_file, "w") as f:
                json.dump(
                    {
                        "seed": config["seed"],
                        "room_params": config["room_params"],
                        "rooms": rooms,
                        "rt60s": rt60s,
                    },
                    f,
                )

        print("Done generating the rooms")

    # now generates all the other argument combinations
    args = []
    for sinr in config["sinr"]:
        for n_interf in config["n_interferers"]:
            for n_mics in config["n_mics"]:

                # in the determined case, n_mics == n_targets
                if config["n_targets"] == "determined":
                    n_targets_lst = [n_mics]
                else:
                    n_targets_lst = config["n_targets"]

                for n_targets in n_targets_lst:
                    for dist_ratio in config["dist_crit_ratio"]:
                        for r in range(config["repeat"]):

                            # bundle all the room parameters for the simulation
                            room_params = rooms[r]

                            args.append(
                                (
                                    sinr,
                                    n_targets,
                                    n_interf,
                                    n_mics,
                                    dist_ratio,
                                    room_params,
                                    sub_seeds[r],
                                )
                            )

    return args


def exp2_gen_args(config):

    # infer a few arguments
    room_dim = config["room"]["room_kwargs"]["p"]
    mic_array_center = np.array(config["room"]["mic_array_location_m"])
    mic_array = mic_array_center[None, :] + np.array(
        config["room"]["mic_array_geometry_m"]
    )
    mic_array = mic_array.T
    critical_distance = config["room"]["critical_distance_m"]

    # master seed
    np.random.seed(config["seed"])

    # choose all the files in advance
    gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
    all_wav_files = sampling(
        config["repeat"],
        np.max(config["n_targets"]) + np.max(config["n_interferers"]),
        config["samples_list"],
        gender_balanced=True,
        seed=gen_files_seed,
    )

    # sub-seeds
    sub_seeds = []
    for r in range(config["repeat"]):
        sub_seeds.append(int(np.random.randint(2 ** 32)))

    # create all distinct interferers locations
    interferers_locs = []
    for n in range(config["repeat"]):
        interferers_locs.append(
            random_locations(
                np.max(config["n_interferers"]),
                room_dim,
                mic_array_center,
                min_dist=critical_distance,
            ).tolist()
        )

    # create all distinct target locations
    target_locs = {}
    for r in range(config["repeat"]):
        # pick rotation
        random_rot = np.random.rand() * 2 * np.pi
        target_locs[r] = {}
        for n in config["n_targets"]:
            target_locs[r][n] = {}
            for dist_ratio in config["dist_crit_ratio"]:
                dist_mic_target = dist_ratio * critical_distance
                target_locs[r][n][dist_ratio] = choose_target_locations(
                    n, mic_array_center, dist_mic_target
                ).tolist()

    args = []
    for sinr in config["sinr"]:
        for n_targets in config["n_targets"]:
            for n_interf in config["n_interferers"]:
                for n_mics in config["n_mics"]:
                    for dist_ratio in config["dist_crit_ratio"]:
                        for r in range(config["repeat"]):

                            assert (
                                n_mics == mic_array.shape[1]
                            ), "n_mics and number of microphones used should match"

                            # bundle all the room parameters for the simulation
                            room_params = {
                                "room_kwargs": config["room"]["room_kwargs"],
                                "mic_array": mic_array.tolist(),
                                "sources": np.concatenate(
                                    (
                                        target_locs[r][n_targets][dist_ratio],
                                        interferers_locs[r],
                                    ),
                                    axis=1,
                                ).tolist(),
                                "wav": all_wav_files[r][: n_targets + n_interf],
                            }

                            args.append(
                                (
                                    sinr,
                                    n_targets,
                                    n_interf,
                                    n_mics,
                                    dist_ratio,
                                    room_params,
                                    sub_seeds[r],
                                )
                            )

    return args


def generate(config):
    if config["name"].startswith("speed_contest"):
        args = exp1_gen_args(config)
    elif config["name"].startswith("reverb_interf_performance"):
        args = exp2_gen_args(config)
    else:
        raise ValueError("Invalid experiment name in the configuration")

    # randomize the execution order
    random.shuffle(args)

    return args
