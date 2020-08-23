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
This file contains a number of routines to generate random rooms and separation
scenarios for the simulation as well as some routines to mix signals at determined
levels of SNR.
"""
import itertools
import math
import time
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import rand

import bss
import pyroomacoustics as pra
from metrics import si_bss_eval
from pyroomacoustics.bss import projection_back


def random_locations(
    n_sources, room_dim, mic_array_location, min_dist=None, max_dist=None, seed=None
):

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    room_dim = np.array(room_dim)
    mic_loc = np.array(mic_array_location)

    sources = np.zeros((3, n_sources))
    for n in range(n_sources):

        good = False
        while not good:
            loc = np.random.rand(3) * room_dim
            d = np.linalg.norm(loc - mic_loc)

            if min_dist is not None and d < min_dist:
                continue

            elif max_dist is not None and d > max_dist:
                continue

            else:
                sources[:, n] = loc
                good = True

    if seed is not None:
        np.random.set_state(state)

    return sources


def choose_target_locations(n_sources, mic_array_location, distance, rot=None):
    # make the sources circular around the center of the array
    # in the horizontal plane

    def angle(n):
        return 2 * np.pi / n_sources * n

    circ_points = np.array(
        [[np.cos(angle(n)), np.sin(angle(n)), 0.0] for n in range(n_sources)]
    ).T

    # Apply the rotation in the x-y plane, if requested
    if rot is not None:
        rot_mat = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
        circ_points[:2, :] = rot_mat @ circ_points[:2, :]

    return mic_array_location[:, None] + circ_points * distance


def convergence_callback(
    Y,
    source_model,
    X,
    n_targets,
    SDR,
    SIR,
    cost_list,
    eval_time,
    ref_sig,
    ref_mic,
    stft_params,
    algo_name,
    algo_is_overdetermined,
):
    global id_wav
    # we will keep track of how long this routine takes
    t_in = time.perf_counter()

    # Compute the current value of the IVA cost function
    cost_list.append(bss.cost_iva(X, Y, model=source_model))

    # prepare STFT parameters
    framesize = stft_params["framesize"]
    hop = stft_params["hop"]
    if stft_params["window"] == "hamming":
        win_a = pra.hamming(framesize)
    else:  # default is Hann
        win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # projection back
    Y = bss.project_back(Y, X[:, :, ref_mic])

    if Y.shape[2] == 1:
        y = pra.transform.synthesis(Y[:, :, 0], framesize, hop, win=win_s)[:, None]
    else:
        y = pra.transform.synthesis(Y, framesize, hop, win=win_s)
    y = y[framesize - hop :, :].astype(np.float64)

    if not algo_is_overdetermined:
        new_ord = np.argsort(np.std(y, axis=0))[::-1]
        y = y[:, new_ord]

    m = np.minimum(y.shape[0], ref_sig.shape[1])

    synth = np.zeros_like(ref_sig)
    # in the overdetermined case, we also take into account the background for SIR computation
    synth[:n_targets, :m] = y[:m, :n_targets].T
    if synth.shape[0] > y.shape[1]:
        # here we copy the first source to fill the channel of the background
        synth[n_targets, :m] = y[:m, 0]

    if ref_sig.shape[0] > n_targets and np.sum(np.abs(ref_sig[n_targets, :])) < 1e-10:
        sdr, sir, sar, perm = si_bss_eval(ref_sig[:n_targets, :m].T, synth[:-1, :m].T)
    else:
        sdr, sir, sar, perm = si_bss_eval(ref_sig[:, :m].T, synth[:, :m].T)

    SDR.append(sdr[:n_targets].tolist())
    SIR.append(sir[:n_targets].tolist())

    t_out = time.perf_counter()
    eval_time.append(t_out - t_in)


def callback_noise_mixer(
    premix, sinr=0, diffuse_ratio=0, ref_mic=0, n_src=None, n_tgt=None, tgt_std=None
):
    """
    This callback function will rescale all the signals so that the SINR
    is fixed to a given value with a given ratio of diffuse noise
    """

    if tgt_std is None:
        tgt_std = np.ones(n_tgt)

    # first normalize all separate recording to have unit power at microphone one
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]
    premix[:n_tgt, :, :] *= tgt_std[:, None, None]

    # Total variance of noise components
    var_noise_tot = 10 ** (-sinr / 10) * np.sum(tgt_std ** 2)

    if n_src > n_tgt:
        # In this case, there are interferers

        # compute noise variance
        sigma_n = np.sqrt((1 - diffuse_ratio) * var_noise_tot)

        # now compute the power of interference signal needed to achieve desired SINR
        sigma_i = np.sqrt((diffuse_ratio / (n_src - n_tgt)) * var_noise_tot)
        premix[n_tgt:n_src, :, :] *= sigma_i

    else:
        # In this case, there is only uncorrelated noise
        sigma_n = np.sqrt(var_noise_tot)

    # the background
    bg = np.sum(premix[n_tgt:n_src, :, :], axis=0)

    # uncorrelated noise
    un = sigma_n * np.random.randn(*premix.shape[1:])

    # Mix down the recorded signals
    mix = np.sum(premix[:n_tgt, :], axis=0) + bg + un

    return mix


def inv_sabine(t60, room_dim, c):
    """
    given desired t60, (shoebox) room dimension and sound speed,
    computes the reflection coefficient (amplitude) and image source
    order needed. the speed of sound used is the package wide default
    (in :py:data:`pyroomacoustics.parameters.constants`).

    parameters
    ----------
    t60: float
        desired t60 (time it takes to go from full amplitude to 60 db decay) in seconds
    room_dim: list of floats
        list of length 2 or 3 of the room side lengths
    c: float
        speed of sound

    returns
    -------
    reflection: float
        the reflection coefficient (in amplitude domain, to be passed to
        room constructor)
    max_order: int
        the maximum image source order necessary to achieve the desired t60
    """

    # finding image sources up to a maximum order creates a (possibly 3d) diamond
    # like pile of (reflected) rooms. now we need to find the image source model order
    # so that reflections at a distance of at least up to ``c * rt60`` are included.
    # one possibility is to find the largest sphere (or circle in 2d) that fits in the
    # diamond. this is what we are doing here.
    R = []
    for l1, l2 in itertools.combinations(room_dim, 2):
        R.append(l1 * l2 / np.sqrt(l1 ** 2 + l2 ** 2))

    V = np.prod(room_dim)  # area (2d) or volume (3d)
    # "surface" computation is diff for 2d and 3d
    if len(room_dim) == 2:
        S = 2 * np.sum(room_dim)
        sab_coef = 12  # the sabine's coefficient needs to be adjusted in 2d
    elif len(room_dim) == 3:
        S = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
        sab_coef = 24

    a2 = sab_coef * np.log(10) * V / (c * S * t60)  # absorption in power (sabine)

    if a2 > 1.0:
        raise ValueError(
            "evaluation of parameters failed. room may be too large for required t60."
        )

    reflection = np.sqrt(1 - a2)  # convert to reflection coefficient

    max_order = math.ceil(c * t60 / np.min(R) - 1)

    return reflection, max_order


def random_room_builder(
    source_signals: List[np.ndarray],
    n_mics: int,
    mic_delta: Optional[float] = None,
    fs: float = 16000,
    t60_interval: Tuple[float, float] = (0.150, 0.500),
    room_width_interval: Tuple[float, float] = (6, 10),
    room_height_interval: Tuple[float, float] = (2.8, 4.5),
    source_zone_height: Tuple[float, float] = [1.0, 2.0],
    guard_zone_width: float = 0.5,
    seed: Optional[int] = None,
):
    """
    This function creates a random room within some parameters.

    The microphone array is circular with the distance between neighboring
    elements set to the maximal distance avoiding spatial aliasing.

    Parameters
    ----------
    source_signals: list of numpy.ndarray
        A list of audio signals for each source
    n_mics: int
        The number of microphones in the microphone array
    mic_delta: float, optional
        The distance between neighboring microphones in the array
    fs: float, optional
        The sampling frequency for the simulation
    t60_interval: (float, float), optional
        An interval where to pick the reverberation time
    room_width_interval: (float, float), optional
        An interval where to pick the room horizontal length/width
    room_height_interval: (float, float), optional
        An interval where to pick the room vertical length
    source_zone_height: (float, float), optional
        The vertical interval where sources and microphones are allowed
    guard_zone_width: float
        The minimum distance between a vertical wall and a source/microphone

    Returns
    -------
    ShoeBox object
        A randomly generated room according to the provided parameters
    float
        The measured T60 reverberation time of the room created
    """

    # save current numpy RNG state and set a known seed
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    n_sources = len(source_signals)

    # sanity checks
    assert source_zone_height[0] > 0
    assert source_zone_height[1] < room_height_interval[0]
    assert source_zone_height[0] <= source_zone_height[1]
    assert 2 * guard_zone_width < room_width_interval[1] - room_width_interval[0]

    def random_location(
        room_dim, n, ref_point=None, min_distance=None, max_distance=None
    ):
        """ Helper function to pick a location in the room """

        width = room_dim[0] - 2 * guard_zone_width
        width_intercept = guard_zone_width

        depth = room_dim[1] - 2 * guard_zone_width
        depth_intercept = guard_zone_width

        height = np.diff(source_zone_height)[0]
        height_intercept = source_zone_height[0]

        locs = rand(3, n)
        locs[0, :] = locs[0, :] * width + width_intercept
        locs[1, :] = locs[1, :] * depth + depth_intercept
        locs[2, :] = locs[2, :] * height + height_intercept

        if ref_point is not None:
            # Check condition
            d = np.linalg.norm(locs - ref_point, axis=0)

            if min_distance is not None and max_distance is not None:
                redo = np.where(np.logical_or(d < min_distance, max_distance < d))[0]
            elif min_distance is not None:
                redo = np.where(d < min_distance)[0]
            elif max_distance is not None:
                redo = np.where(d > max_distance)[0]
            else:
                redo = []

            # Recursively call this function on sources to redraw
            if len(redo) > 0:
                locs[:, redo] = random_location(
                    room_dim,
                    len(redo),
                    ref_point=ref_point,
                    min_distance=min_distance,
                    max_distance=max_distance,
                )

        return locs

    c = pra.constants.get("c")

    # Create the room
    # Sometimes the room dimension and required T60 are not compatible, then
    # we just try again with new random values
    retry = True
    while retry:
        try:
            room_dim = np.array(
                [
                    rand() * np.diff(room_width_interval)[0] + room_width_interval[0],
                    rand() * np.diff(room_width_interval)[0] + room_width_interval[0],
                    rand() * np.diff(room_height_interval)[0] + room_height_interval[0],
                ]
            )
            t60 = rand() * np.diff(t60_interval)[0] + t60_interval[0]
            reflection, max_order = inv_sabine(t60, room_dim, c)
            retry = False
        except ValueError:
            pass
    # Create the room based on the random parameters
    room = pra.ShoeBox(room_dim, fs=fs, absorption=1 - reflection, max_order=max_order)

    # The critical distance
    # https://en.wikipedia.org/wiki/Critical_distance
    d_critical = 0.057 * np.sqrt(np.prod(room_dim) / t60)

    # default intermic distance is set according to nyquist criterion
    # i.e. 1/2 wavelength corresponding to fs / 2 at given speed of sound
    if mic_delta is None:
        mic_delta = 0.5 * (c / (0.5 * fs))

    # the microphone array is uniformly circular with the distance between
    # neighboring elements of mic_delta
    mic_center = random_location(room_dim, 1)
    mic_rotation = rand() * 2 * np.pi
    mic_radius = 0.5 * mic_delta / np.sin(np.pi / n_mics)
    mic_array = pra.MicrophoneArray(
        np.vstack(
            (
                pra.circular_2D_array(
                    mic_center[:2, 0], n_mics, mic_rotation, mic_radius
                ),
                mic_center[2, 0] * np.ones(n_mics),
            )
        ),
        room.fs,
    )
    room.add_microphone_array(mic_array)

    # Now we will get the sources at random
    source_locs = []

    # Choose the target location at least as far as the critical distance
    # Then the other sources, yet one further meter away
    target_source = random_location(
        room_dim,
        1,
        ref_point=mic_center,
        min_distance=d_critical,
        max_distance=d_critical + 1,
    )
    interferers = random_location(
        room_dim, n_sources - 1, ref_point=mic_center, min_distance=d_critical + 1
    )
    source_locs = np.concatenate((target_source, interferers), axis=1)

    for s, signal in enumerate(source_signals):
        room.add_source(source_locs[:, s], signal=signal)

    # pre-compute the impulse responses
    room.compute_rir()
    t60_actual = pra.experimental.measure_rt60(room.rir[0][0], room.fs)

    # restore numpy RNG former state
    if seed is not None:
        np.random.set_state(rng_state)

    return room, t60_actual


def random_room_definition(
    n_sources: List[np.ndarray],
    n_mics: int,
    mic_delta: Optional[float] = None,
    fs: float = 16000,
    t60_interval: Tuple[float, float] = (0.150, 0.500),
    room_width_interval: Tuple[float, float] = (6, 10),
    room_height_interval: Tuple[float, float] = (2.8, 4.5),
    source_zone_height: Tuple[float, float] = [1.0, 2.0],
    guard_zone_width: float = 0.5,
    seed: Optional[int] = None,
):
    """
    This function creates a random room within some parameters.

    The microphone array is circular with the distance between neighboring
    elements set to the maximal distance avoiding spatial aliasing.

    Parameters
    ----------
    source_signals: list of numpy.ndarray
        A list of audio signals for each source
    n_mics: int
        The number of microphones in the microphone array
    mic_delta: float, optional
        The distance between neighboring microphones in the array
    fs: float, optional
        The sampling frequency for the simulation
    t60_interval: (float, float), optional
        An interval where to pick the reverberation time
    room_width_interval: (float, float), optional
        An interval where to pick the room horizontal length/width
    room_height_interval: (float, float), optional
        An interval where to pick the room vertical length
    source_zone_height: (float, float), optional
        The vertical interval where sources and microphones are allowed
    guard_zone_width: float
        The minimum distance between a vertical wall and a source/microphone

    Returns
    -------
    ShoeBox object
        A randomly generated room according to the provided parameters
    float
        The measured T60 reverberation time of the room created
    """

    # save current numpy RNG state and set a known seed
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    # sanity checks
    assert source_zone_height[0] > 0
    assert source_zone_height[1] < room_height_interval[0]
    assert source_zone_height[0] <= source_zone_height[1]
    assert 2 * guard_zone_width < room_width_interval[1] - room_width_interval[0]

    def random_location(
        room_dim, n, ref_point=None, min_distance=None, max_distance=None
    ):
        """ Helper function to pick a location in the room """

        width = room_dim[0] - 2 * guard_zone_width
        width_intercept = guard_zone_width

        depth = room_dim[1] - 2 * guard_zone_width
        depth_intercept = guard_zone_width

        height = np.diff(source_zone_height)[0]
        height_intercept = source_zone_height[0]

        locs = rand(3, n)
        locs[0, :] = locs[0, :] * width + width_intercept
        locs[1, :] = locs[1, :] * depth + depth_intercept
        locs[2, :] = locs[2, :] * height + height_intercept

        if ref_point is not None:
            # Check condition
            d = np.linalg.norm(locs - ref_point, axis=0)

            if min_distance is not None and max_distance is not None:
                redo = np.where(np.logical_or(d < min_distance, max_distance < d))[0]
            elif min_distance is not None:
                redo = np.where(d < min_distance)[0]
            elif max_distance is not None:
                redo = np.where(d > max_distance)[0]
            else:
                redo = []

            # Recursively call this function on sources to redraw
            if len(redo) > 0:
                locs[:, redo] = random_location(
                    room_dim,
                    len(redo),
                    ref_point=ref_point,
                    min_distance=min_distance,
                    max_distance=max_distance,
                )

        return locs

    c = pra.constants.get("c")

    # Create the room
    # Sometimes the room dimension and required T60 are not compatible, then
    # we just try again with new random values
    retry = True
    while retry:
        try:
            room_dim = np.array(
                [
                    rand() * np.diff(room_width_interval)[0] + room_width_interval[0],
                    rand() * np.diff(room_width_interval)[0] + room_width_interval[0],
                    rand() * np.diff(room_height_interval)[0] + room_height_interval[0],
                ]
            )
            t60 = rand() * np.diff(t60_interval)[0] + t60_interval[0]
            reflection, max_order = inv_sabine(t60, room_dim, c)
            retry = False
        except ValueError:
            pass

    # Create the room based on the random parameters
    room_kwargs = {
        "p": room_dim.tolist(),
        "fs": fs,
        "absorption": 1 - reflection,
        "max_order": max_order,
    }

    # The critical distance
    # https://en.wikipedia.org/wiki/Critical_distance
    d_critical = 0.057 * np.sqrt(np.prod(room_dim) / t60)

    # default intermic distance is set according to nyquist criterion
    # i.e. 1/2 wavelength corresponding to fs / 2 at given speed of sound
    if mic_delta is None:
        mic_delta = 0.5 * (c / (0.5 * fs))

    # the microphone array is uniformly circular with the distance between
    # neighboring elements of mic_delta
    mic_center = random_location(room_dim, 1)
    mic_rotation = rand() * 2 * np.pi
    mic_radius = 0.5 * mic_delta / np.sin(np.pi / n_mics)
    mic_locs = np.vstack(
        (
            pra.circular_2D_array(mic_center[:2, 0], n_mics, mic_rotation, mic_radius),
            mic_center[2, 0] * np.ones(n_mics),
        )
    )

    # Now we will get the sources at random
    source_locs = []

    # Choose the target location at least as far as the critical distance
    # Then the other sources, yet one further meter away
    source_locs = random_location(
        room_dim, n_sources, ref_point=mic_center, min_distance=d_critical
    )

    # order the sources from closes to the microphones to furthest
    dist = np.linalg.norm(source_locs - mic_locs[:, 0, None], axis=0)
    I_dist = np.argsort(dist)
    source_locs = source_locs[:, I_dist[::-1]]

    # restore numpy RNG former state
    if seed is not None:
        np.random.set_state(rng_state)

    room_params = {
        "room_kwargs": room_kwargs,
        "mic_array": mic_locs.tolist(),
        "sources": source_locs.tolist(),
    }

    return room_params, estimate_rt60(room_params)


def estimate_rt60(room_params):
    """ Estimate RT60 of room using simulation """
    # Create the room object
    room = pra.ShoeBox(**room_params["room_kwargs"])
    mic_loc = np.mean(room_params["mic_array"], axis=1, keepdims=True)
    S = np.array(room_params["sources"])
    room.add_microphone_array(pra.MicrophoneArray(mic_loc, room.fs))
    for n in range(S.shape[1]):
        room.add_source(S[:, n])

    # compute the impulse responses
    room.compute_rir()

    rt60 = np.median(
        [pra.experimental.measure_rt60(rir, fs=room.fs) for rir in room.rir[0]]
    )

    return rt60
