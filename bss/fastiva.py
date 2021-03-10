import numpy as np

from .projection_back import project_back
from .update_rules import (_block_ip, _ip_double, _ip_double_two_channels,
                           _ip_single, _ipa, _iss_single,
                           _joint_demix_background,
                           _parametric_background_update)
from .utils import TwoStepsIterator, cost_iva, demix, tensor_H

_eps = 1e-15

def score_laplace(Y):
    """
    Computes the score function for the Laplace prior

    Parameters
    ----------
    Y: ndarray (n_freq, n_chan, n_frames)
        The source signal
    """

    r = np.linalg.norm(Y, axis=0, keepdims=True)
    return Y / np.maximum(r, _eps)

def score_gauss(Y):
    """
    Computes the score function for the time-varying Gauss prior

    Parameters
    ----------
    Y: ndarray (n_freq, n_chan, n_frames)
        The source signal
    """

    r = np.linalg.norm(Y, axis=0, keepdims=True) ** 2 / Y.shape[0]
    return Y / np.maximum(r, _eps)

def fastiva(
    X,
    n_iter=20,
    step_size=100.,
    proj_back=True,
    model="laplace",
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):

    """
    This is an implementation of IVA using the natural gradient that separates
    the input signal into statistically independent sources. The separation is
    done in the time-frequency domain and the FFT length should be
    approximately equal to the reverberation time.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components. When
        ``n_src==nchannels``, the algorithms is identical to AuxIVA. When
        ``n_src==1``, then it is doing independent vector extraction.
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    update_rule: str
        The update rules to use for the algorithm, one of ``ip-param``, ``ip2-param``,
        ``demix-bg``, ``ip-block``
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence
    callback_checkpoints: list of int
        A list of epoch number when the callback should be called

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    if model not in ["laplace", "gauss"]:
        raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    # We will need the inverse square root of Cx
    e_val, e_vec = np.linalg.eigh(Cx)
    Q_H_inv = e_vec @ ((1.0 / np.sqrt(e_val[:, :, None])) * tensor_H(e_vec))
    # Q_H = e_vec[:, :, :] * np.sqrt(e_val[:, None, :])

    eps = 1e-10
    V = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    r_inv = np.zeros((n_chan, n_frames))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_chan, n_frames), dtype=X.dtype)
    X_original = X.copy()

    # pre-whiten the input signal
    X = Q_H_inv @ X.transpose([1, 2, 0])

    # initialize at identity
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W[:] = np.eye(n_chan)[None, ...]
    Y = X.copy()

    # Monitor the algorithm progression
    if callback is not None and 0 in callback_checkpoints:
        Y_tmp = Y.transpose([2, 0, 1])
        if "eval_demix_mat" in kwargs:
            callback(Y_tmp.copy(), W, model)
        else:
            callback(Y_tmp.copy(), model)

    for epoch in range(n_iter):

        # fixed-point update equations
        # shape: (n_freq, n_frames)
        Y_sq = np.abs(Y) ** 2

        # shape: (n_chan, n_frames)
        # r = 1. / np.mean(Y_sq, axis=0)
        r = np.sum(Y_sq, axis=0, keepdims=True)
        if model == "laplace":
            r_inv_1 = 0.5 / np.maximum(eps, np.sqrt(r))
            r_inv_2 = -0.25 / np.maximum(eps, r ** 1.5)
        elif model == "gauss":
            r_inv_1 = 1.0 / np.maximum(eps, r)
            r_inv_2 = -1.0 / np.maximum(eps, r ** 2)
            pass
        else:
            raise NotImplementedError()

        # shape: (n_freq, n_chan)
        a = np.mean(r_inv_1 + Y_sq * r_inv_2, axis=-1)
        # shape: (n_freq, n_chan, n_chan)
        b = -(r_inv_1 * Y) @ tensor_H(X) / n_frames

        # update the extraction filter
        W[:] = a[..., None, :] * W + b

        # symmetric decorrelation
        WHW = tensor_H(W) @ W
        WHW = 0.5 * (WHW + tensor_H(WHW))
        e_val, e_vec = np.linalg.eigh(WHW)
        L = (e_vec * np.reciprocal(np.maximum(np.sqrt(e_val[:, None, :]), 1e-15))) @ tensor_H(e_vec)
        np.matmul(W, L, out=W)

        # Update the output signal
        # note: eigenvalues are in ascending order, we use the smallest
        np.matmul(W, X, out=Y)

        # Monitor the algorithm progression
        if callback is not None and (epoch + 1) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1])
            if "eval_demix_mat" in kwargs:
                callback(Y_tmp.copy(), W @ Q_H_inv, model)
            else:
                callback(Y_tmp.copy(), model)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        W[:] = W * Q_H_inv
        return Y, W
    else:
        return Y
