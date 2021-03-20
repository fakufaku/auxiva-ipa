import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data

substitutions = {
    "Algorithm": {
        "AuxIVA-IP (PCA)": "IP",
        "AuxIVA-ISS (PCA)": "ISS",
        "AuxIVA-IP2 (PCA)": "IP2",
        "AuxIVA-IPA (PCA)": "IPA",
        "FastIVA (PCA)": "FIVA",
    }
}

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

    df, final_val_tbl, conv_tbl, rt60, parameters = load_data(args.dirs, pickle=pickle)

    conv_tbl = conv_tbl.replace(substitutions)

    col_order = [
        "IP",
        "ISS",
        "IP2",
        "IPA",
        "FIVA",
    ]

    agg_func = np.median
    flt_fmt = "%.1f"

    print("Runtime Table:")
    pt_rt = conv_tbl.pivot_table(
        index=["Mics"], columns=["SINR", "Algorithm"], aggfunc=agg_func
    )["Runtime"]
    pt_rt = pt_rt.reindex(level=1, columns=col_order)
    print(pt_rt.to_latex(float_format=flt_fmt))

    print()

    print("Speed-up Factors:")
    for snr in [5, 15, 25]:
        print(f"SNR {snr}")
        print(pt_rt[snr].div(pt_rt[snr, "IPA"], axis=0))

    print()

    print("Iteration Table Table:")
    pt_it = conv_tbl.pivot_table(
        index=["Mics"], columns=["SINR", "Algorithm"], aggfunc=agg_func
    )["Iterations"]
    pt_it = pt_it.reindex(level=1, columns=col_order)
    print(pt_it.to_latex(float_format="%.0f"))
