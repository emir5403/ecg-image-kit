"""Extract a sequence/time-series from an image."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[..., :3]
        img = img.astype(float)
        img_gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
        return img_gray
    return img.astype(float)


def image_to_sequence(
    img: np.ndarray,
    mode: str,
    method: str,
    windowlen: int | None = None,
    plot_result: bool = False,
) -> np.ndarray:
    """
    Extract a sequence/time-series from an image.

    Args:
        img: Image array (grayscale or RGB).
        mode: "dark-foreground" or "bright-foreground".
        method: Filtering method.
        windowlen: Optional window length for smoothing.
        plot_result: Whether to plot the result.

    Returns:
        Extracted sequence from the image.
    """
    if windowlen is None:
        windowlen = 3

    img_gray = _to_gray(img)

    if mode == "dark-foreground":
        img_flipped = np.max(img_gray) - img_gray
    elif mode == "bright-foreground":
        img_flipped = img_gray
    else:
        raise ValueError("mode must be 'dark-foreground' or 'bright-foreground'")

    if method == "max_finder":
        img_filtered = img_flipped
    elif method == "moving_average":
        kernel = np.ones((windowlen, windowlen), dtype=float)
        kernel /= kernel.sum()
        img_filtered = ndimage.convolve(img_flipped, kernel, mode="nearest")
    elif method == "hor_smoothing":
        kernel = np.ones((1, windowlen), dtype=float)
        kernel /= kernel.sum()
        img_filtered = ndimage.convolve(img_flipped, kernel, mode="nearest")
    elif method == "all_left_right_neighbors":
        kernel = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=float)
        kernel /= kernel.sum()
        img_filtered = ndimage.convolve(img_flipped, kernel, mode="nearest")
    elif method == "combined_all_neighbors":
        kernel1 = np.array([[1, 1, 1]], dtype=float)
        kernel1 /= kernel1.sum()
        z1 = ndimage.convolve(img_flipped, kernel1, mode="nearest")

        kernel2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        kernel2 /= kernel2.sum()
        z2 = ndimage.convolve(img_flipped, kernel2, mode="nearest")

        kernel3 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        kernel3 /= kernel3.sum()
        z3 = ndimage.convolve(img_flipped, kernel3, mode="nearest")

        img_filtered = np.minimum(np.minimum(z1, z2), z3)
    else:
        raise ValueError(f"Unsupported method: {method}")

    indices = np.argmax(img_filtered, axis=0)
    img_height = img_filtered.shape[0]
    data = img_height - (indices + 1)

    if plot_result:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.plot(np.arange(img.shape[1]), img.shape[0] - data, "g", linewidth=3)
        plt.show()

    return data
