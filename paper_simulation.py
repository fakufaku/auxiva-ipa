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
This file contains the code to run the systematic simulation for evaluation
of auxiva-ipa and other algorithms.
"""
import os
import sys

import numpy as np

import pyroomacoustics as pra
import rrtools
from arg_generators import generate
from generate_samples import sampling, wav_read_center
# Get the data if needed
from get_data import get_data, samples_dir
from room_builder import callback_noise_mixer, random_room_builder

# Routines for manipulating audio samples
sys.path.append(samples_dir)

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])


def init(parameters):
    parameters["base_dir"] = base_dir


def one_loop(args):
    global parameters

    import sys

    sys.path.append(parameters["base_dir"])
    from simulation_loop import run

    return run(args, parameters)


if __name__ == "__main__":

    rrtools.run(
        one_loop,
        generate,
        func_init=init,
        base_dir=base_dir,
        results_dir="data/",
        description="Simulation for OverIVA",
    )
