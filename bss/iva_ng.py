import numpy as np

from .projection_back import project_back
from .update_rules import (_block_ip, _ip_double, _ip_double_two_channels,
                           _ip_single, _ipa, _iss_single,
                           _joint_demix_background,
                           _parametric_background_update)
from .utils import TwoStepsIterator, demix, tensor_H, cost_iva
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

def iva_ng(
    X,
    n_iter=20,
    step_size=0.3,
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

    # Things are more efficient when the frequencies are over the first axis
    X_original = X  # keep the original for the projection back
    X = X.transpose([1, 2, 0]).copy()

    # scale the signal so that all channels have unit variance
    scale = X[..., None, :] @ tensor_H(X[..., None, :]) / n_frames
    scale = scale[..., 0] / n_freq
    X = X / np.sqrt(scale)

    # allocate demixing matrix and take view to the target and background parts
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    eye = np.broadcast_to(np.eye(n_chan)[None, ...], W.shape)

    # initialize at identity
    W[:] = eye
    Y = X.copy()

    for epoch in range(n_iter):

        # update the demixing matrix
        if model == "laplace":
            score = score_laplace(Y)
        elif model == "gauss":
            score = score_gauss(Y)
        else:
            raise NotImplementedError()

        # natural gradient update
        delta = (eye - score @ tensor_H(Y) / n_frames) @ W
        W[:] = W + step_size * delta

        # update the demixed signal
        np.matmul(W, X, out=Y)

        # Monitor the algorithm progression
        if callback is not None and (epoch + 1) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1])
            if "eval_demix_mat" in kwargs:
                callback(Y_tmp.copy(), W.copy(), model)
            else:
                callback(Y_tmp, model)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        W[:] = W * np.sqrt(scale.swapaxes(-2, -1))
        return Y, W
    else:
        return Y
