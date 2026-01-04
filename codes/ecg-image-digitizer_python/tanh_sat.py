"""Saturate outlier samples using a tanh shape function."""
from __future__ import annotations

import numpy as np


def tanh_sat(x: np.ndarray, param: float | np.ndarray, mode: str = "ksigma") -> np.ndarray:
    """
    Saturate outlier samples using a tanh shape function.

    Args:
        x: Input data, vector or matrix (channels x time).
        param: Scaling factor for saturation level.
        mode: "ksigma" or "absolute".

    Returns:
        Saturated data with outliers replaced by the saturation level.
    """
    x = np.asarray(x, dtype=float)
    was_vector = False
    if x.ndim == 1:
        x = x[np.newaxis, :]
        was_vector = True

    if mode == "ksigma":
        alpha = param * np.std(x, axis=1, ddof=0)
    elif mode == "absolute":
        if np.isscalar(param):
            alpha = np.full((x.shape[0],), float(param))
        else:
            alpha = np.asarray(param, dtype=float)
            if alpha.shape[0] != x.shape[0]:
                raise ValueError(
                    "Parameter must be scalar or a vector with the same number of channels"
                )
    else:
        raise ValueError("Undefined mode")

    alpha = alpha.reshape(-1, 1)
    alpha = np.where(alpha == 0, np.finfo(float).eps, alpha)
    y = alpha * np.tanh(x / alpha)

    if was_vector:
        return y.flatten()
    return y
