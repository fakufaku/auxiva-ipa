# This file loads the data from the simulation result file and creates a
# pandas data frame for further processing.
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
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# This table maps some of the labels we used in the
# simulation to labels we would like to use in the figures
# of the paper
substitutions = {
    "Algorithm": {
        "five_laplace": "FIVE",
        "overiva_ip_laplace": "OverIVA-IP",
        "overiva_ip2_laplace": "OverIVA-IP2",
        "overiva_ip_block_laplace": "OverIVA-IP-NP",
        "overiva_ip2_block_laplace": "OverIVA-IP2-NP",
        "overiva_demix_bg_laplace": "OverIVA-DX/BG",
        "ogive_laplace": "OGIVEs",
        "auxiva_laplace": "AuxIVA-IP (PCA)",
        "auxiva_laplace_nopca": "AuxIVA-IP",
        "auxiva_iss_laplace": "AuxIVA-ISS (PCA)",
        "auxiva_iss_laplace_nopca": "AuxIVA-ISS",
        "auxiva2_laplace": "AuxIVA-IP2 (PCA)",
        "auxiva2_laplace_nopca": "AuxIVA-IP2",
        "auxiva_pca": "PCA+AuxIVA-IP",
        "auxiva_demix_steer_nopca": "AuxIVA-IPA",
        "auxiva_demix_steer_pca": "AuxIVA-IPA (PCA)",
        "auxiva_ipa_nopca": "AuxIVA-IPA",
        "auxiva_ipa_pca": "AuxIVA-IPA (PCA)",
        "auxiva_ipa2_nopca": "AuxIVA-IPA2",
        "auxiva_ipa2_pca": "AuxIVA-IPA2 (PCA)",
        "auxiva_ipa2": "AuxIVA-IPA2 (PCA)",
        "overiva_demix_steer": "OverIVA-IPA",
        "fastiva": "FastIVA (PCA)",
        "fastiva_nopca": "FastIVA",
        "iva_ng_0.3": "IVA-NG (PCA)",
        "iva_ng_0.3_nopca": "IVA-NG",
        "pca": "PCA",
    }
}


def load_data(dirs, pickle=False):

    parameters = dict()
    algorithms = dict()
    df = None

    data_files = []

    parameters = None
    for i, data_dir in enumerate(dirs):

        print("Reading in", data_dir)

        # add the data file from this directory
        data_file = os.path.join(data_dir, "data.json")
        if os.path.exists(data_file):
            data_files.append(data_file)
        else:
            raise ValueError("File {} doesn" "t exist".format(data_file))

        # get the simulation config
        with open(os.path.join(data_dir, "parameters.json"), "r") as f:
            new_parameters = json.load(f)

        if parameters is None:
            parameters = new_parameters
        else:
            parameters["algorithm_kwargs"].update(new_parameters["algorithm_kwargs"])

    # algorithms to take in the plot
    algos = algorithms.keys()

    # check if a pickle file exists for these files
    pickle_file = f".{parameters['name']}.pickle"
    rt60_file = ".rt60.pickle"

    if os.path.isfile(pickle_file) and pickle:
        print("Reading existing pickle file...")
        # read the pickle file
        df = pd.read_pickle(pickle_file)
        rt60 = pd.read_pickle(rt60_file)

    else:

        # reading all data files in the directory
        records = []
        rt60_list = []
        for file in data_files:
            with open(file, "r") as f:
                content = json.load(f)
                for seg in content:
                    records += seg

        # build the data table line by line
        print("Building table")
        columns = [
            "Algorithm",
            "Sources",
            "Interferers",
            "Mics",
            "RT60",
            "Distance",
            "SINR",
            "seed",
            "Iteration",
            "Iteration_Index",
            "Runtime [s]",
            "SI-SDR [dB]",
            "SI-SIR [dB]",
            "\u0394SI-SDR [dB]",
            "\u0394SI-SIR [dB]",
            "Success",
        ]
        table = []
        num_sources = set()

        copy_fields = [
            "algorithm",
            "n_targets",
            "n_interferers",
            "n_mics",
            "rt60",
            "dist_ratio",
            "sinr",
            "seed",
        ]

        number_failed_records = 0

        for record in records:

            algo_kwargs = parameters["algorithm_kwargs"][record["algorithm"]]["kwargs"]
            if "callback_checkpoints" in algo_kwargs:
                checkpoints = algo_kwargs["callback_checkpoints"].copy()
                checkpoints.insert(0, 0)
                algo_n_iter = algo_kwargs["n_iter"]
            else:
                checkpoints = list(range(len(record["sdr"])))
                algo_n_iter = 1

            rt60_list.append(record["rt60"])

            try:
                fs = parameters["room_params"]["fs"]
            except KeyError:
                fs = parameters["room"]["room_kwargs"]["fs"]

            # runtime per iteration, per second of audio
            runtime = record["runtime"] / record["n_samples"] * fs / algo_n_iter
            evaltime = record["eval_time"] / record["n_samples"] * fs / algo_n_iter

            if len(record["sdr"]) == 2 and np.any(np.isnan(record["sdr"][-1])):
                number_failed_records += 1
                continue

            sdr_i = np.array(record["sdr"][0])  # Initial SDR
            sir_i = np.array(record["sir"][0])  # Initial SDR

            for i, (n_iter, sdr, sir) in enumerate(
                zip(checkpoints, record["sdr"], record["sir"])
            ):

                entry = [record[field] for field in copy_fields]
                entry.append(n_iter)
                entry.append(i)

                # seconds processing / second of audio
                entry.append(runtime * n_iter)

                try:
                    sdr_f = np.array(sdr)  # Final SDR
                    sir_f = np.array(sir)  # Final SDR

                    table.append(
                        entry
                        + [
                            np.mean(sdr_f),
                            np.mean(sir_f),
                            np.mean(sdr_f - sdr_i),
                            np.mean(sir_f - sir_i),
                            float(np.mean(sir_f > 0.0)),
                        ]
                    )
                except Exception:
                    continue

        # create a pandas frame
        print("Making PANDAS frame...")
        df = pd.DataFrame(table, columns=columns)
        rt60 = pd.DataFrame(rt60_list, columns=["RT60"])

        df.to_pickle(pickle_file)
        rt60.to_pickle(rt60_file)

        if number_failed_records > 0:
            import warnings

            warnings.warn(f"Number of failed record: {number_failed_records}")

    # apply the subsititutions
    df = df.replace(substitutions)

    return df, rt60, parameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Load simulation data into a pandas frame"
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Read the aggregated data table from a pickle cache",
    )
    parser.add_argument(
        "dirs",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="The directory containing the simulation output files.",
    )
    args = parser.parse_args()

    dirs = args.dirs
    pickle = args.pickle

    df, rt60, parameters = load_data(args.dirs, pickle=pickle)
