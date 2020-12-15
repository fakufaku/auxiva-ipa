import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data

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

    df, conv_tbl, rt60, parameters = load_data(args.dirs, pickle=pickle)

    col_order = [
        "AuxIVA-IP (PCA)",
        "AuxIVA-ISS (PCA)",
        "AuxIVA-IP2 (PCA)",
        "AuxIVA-IPA (PCA)",
    ]

    agg_func = np.mean

    print("Runtime Table:")
    pt_rt = conv_tbl.pivot_table(
        index=["SINR", "Mics"], columns=["Algorithm"], aggfunc=agg_func
    )["Runtime"]
    pt_rt = pt_rt.reindex(columns=col_order)
    print(pt_rt.to_latex(float_format="%.2f"))

    print()

    print("Iteration Table Table:")
    pt_it = conv_tbl.pivot_table(
        index=["SINR", "Mics"], columns=["Algorithm"], aggfunc=agg_func
    )["Iterations"]
    pt_it = pt_it.reindex(columns=col_order)
    print(pt_it.to_latex(float_format="%.2f"))
