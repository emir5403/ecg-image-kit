"""ECG grid size estimation using matched filtering."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d, find_peaks

from tanh_sat import tanh_sat


def _to_gray_normalized(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[..., :3]
        img_gray = img.astype(float)
        img_gray = (
            0.2989 * img_gray[..., 0]
            + 0.5870 * img_gray[..., 1]
            + 0.1140 * img_gray[..., 2]
        )
        img_gray = img_gray / np.max(img_gray)
    else:
        img_gray = img.astype(float)
        img_gray = 1.0 - (img_gray / np.max(img_gray))
    return img_gray


def _boundary_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=float)
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    mask /= np.sum(mask)
    return mask


def ecg_gridest_matchedfilt(
    img: np.ndarray, params: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate grid size in ECG images using matched filtering.
    """
    if params is None:
        params = {}

    params.setdefault("blur_sigma_in_inch", 1.0)
    params.setdefault("paper_size_in_inch", [11, 8.5])
    params.setdefault("remove_shadows", True)
    params.setdefault("sat_pre_grid_det", True)
    params.setdefault("sat_level_pre_grid_det", 0.7)
    params.setdefault("num_seg_hor", 4)
    params.setdefault("num_seg_ver", 4)
    params.setdefault("tiling_method", "RANDOM_TILING")
    params.setdefault("total_segments", 16)
    params.setdefault("max_grid_size", 30)
    params.setdefault("min_grid_size", 2)
    params.setdefault("power_avg_prctile_th", 95.0)
    params.setdefault("detailed_plots", 0)

    width = img.shape[1]
    height = img.shape[0]

    img_gray = _to_gray_normalized(img)

    if params["remove_shadows"]:
        blur_sigma = np.mean(
            [
                width * params["blur_sigma_in_inch"] / params["paper_size_in_inch"][0],
                height * params["blur_sigma_in_inch"] / params["paper_size_in_inch"][1],
            ]
        )
        img_gray_blurred = ndimage.gaussian_filter(img_gray, blur_sigma, mode="nearest")
        img_gray_normalized = img_gray / img_gray_blurred
        img_gray_normalized = (img_gray_normalized - np.min(img_gray_normalized)) / (
            np.max(img_gray_normalized) - np.min(img_gray_normalized)
        )
    else:
        img_gray_blurred = img_gray
        img_gray_normalized = img_gray

    if params["sat_pre_grid_det"]:
        img_sat = tanh_sat(
            1.0 - img_gray_normalized.flatten(),
            params["sat_level_pre_grid_det"],
            "ksigma",
        )
        img_gray_normalized = img_sat.reshape(img_gray_normalized.shape)

    seg_width = int(np.floor(width / params["num_seg_hor"]))
    seg_height = int(np.floor(height / params["num_seg_ver"]))
    mask_sizes = np.arange(params["min_grid_size"], params["max_grid_size"] + 1)

    rng = np.random.default_rng()
    if params["tiling_method"] == "REGULAR_TILING":
        segments = []
        for i in range(params["num_seg_ver"]):
            for j in range(params["num_seg_hor"]):
                segment = img_gray_normalized[
                    i * seg_height : (i + 1) * seg_height,
                    j * seg_width : (j + 1) * seg_width,
                ]
                segments.append(segment)
    else:
        segments = []
        for _ in range(params["total_segments"]):
            start_hor = rng.integers(0, width - seg_width + 1)
            start_ver = rng.integers(0, height - seg_height + 1)
            segment = img_gray_normalized[
                start_ver : start_ver + seg_height,
                start_hor : start_hor + seg_width,
            ]
            segments.append(segment)

    matched_filter_powers = np.zeros((len(segments), len(mask_sizes)))
    for idx, segment in enumerate(segments):
        seg_mean = np.mean(segment)
        seg_std = np.std(segment)
        if seg_std == 0:
            seg_std = np.finfo(float).eps
        segment = (segment - seg_mean) / seg_std
        for g_idx, mask_size in enumerate(mask_sizes):
            mask = _boundary_mask(mask_size)
            mask = mask - np.mean(mask)
            matched_filtered = convolve2d(segment, mask, mode="same", boundary="symm")
            power = matched_filtered**2
            power_th = np.percentile(power, params["power_avg_prctile_th"])
            power_selected = power[power > power_th]
            if power_selected.size:
                matched_filter_powers[idx, g_idx] = 10 * np.log10(
                    np.mean(power_selected)
                )
            else:
                matched_filter_powers[idx, g_idx] = np.nan

    matched_filter_powers_avg = np.nanmean(matched_filter_powers, axis=0)
    peak_indices, peak_props = find_peaks(matched_filter_powers_avg)
    grid_size_prominences = peak_props.get("prominences", np.array([]))
    grid_sizes = mask_sizes[peak_indices] - 1

    if params["detailed_plots"] > 0:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(mask_sizes - 1, matched_filter_powers.T)
        plt.plot(mask_sizes - 1, matched_filter_powers_avg, "k", linewidth=3)
        if peak_indices.size:
            plt.plot(
                mask_sizes[peak_indices] - 1,
                matched_filter_powers_avg[peak_indices],
                "ro",
                markersize=12,
            )
        plt.grid(True)
        plt.xlabel("Grid size")
        plt.ylabel("Average power (dB)")
        plt.title("Average matched-filter output power vs grid size")

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("img")
        axes[0, 1].imshow(img_gray, cmap="gray")
        axes[0, 1].set_title("img_gray")
        axes[1, 0].imshow(img_gray_blurred, cmap="gray")
        axes[1, 0].set_title("img_gray_blurred")
        axes[1, 1].imshow(img_gray_normalized, cmap="gray")
        axes[1, 1].set_title("img_gray_normalized")
        fig.suptitle("Preprocessing stages")
        plt.show()

    return (
        grid_sizes,
        grid_size_prominences,
        mask_sizes,
        matched_filter_powers_avg,
        peak_indices,
    )
