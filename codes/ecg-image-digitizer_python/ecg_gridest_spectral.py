"""ECG grid size estimation using spectral analysis."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize

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


def _gaussian_kernel(shape: tuple[int, int], sigma: float) -> np.ndarray:
    y = np.linspace(-(shape[0] - 1) / 2.0, (shape[0] - 1) / 2.0, shape[0])
    x = np.linspace(-(shape[1] - 1) / 2.0, (shape[1] - 1) / 2.0, shape[1])
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def ecg_gridest_spectral(
    img: np.ndarray, params: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate grid size in ECG images using spectral analysis.
    """
    if params is None:
        params = {}

    params.setdefault("blur_sigma_in_inch", 1.0)
    params.setdefault("paper_size_in_inch", [11, 8.5])
    params.setdefault("remove_shadows", True)
    params.setdefault("apply_edge_detection", False)
    params.setdefault("post_edge_det_gauss_filt_std", 0.01)
    params.setdefault("post_edge_det_sat", True)
    params.setdefault("sat_level_upper_prctile", 99.0)
    params.setdefault("sat_level_lower_prctile", 1.0)
    params.setdefault("sat_pre_grid_det", True)
    params.setdefault("sat_level_pre_grid_det", 0.7)
    params.setdefault("num_seg_hor", 5)
    params.setdefault("num_seg_ver", 5)
    params.setdefault("spectral_tiling_method", "RANDOM_TILING")
    params.setdefault("total_segments", 100)
    params.setdefault("seg_width_rand_dev", 0.1)
    params.setdefault("seg_height_rand_dev", 0.1)
    params.setdefault("min_grid_resolution", 1)
    params.setdefault("min_grid_peak_prominence", 1.0)
    params.setdefault("detailed_plots", 0)
    params.setdefault("smooth_spectra", True)
    params.setdefault("gauss_win_sigma", 0.3)
    params.setdefault("patch_avg_method", "MEDIAN")

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

    if params["apply_edge_detection"]:
        edges = canny(img_gray_normalized)
        edges = skeletonize(edges)
        blur_sigma = np.mean(
            [
                width
                * params["post_edge_det_gauss_filt_std"]
                / params["paper_size_in_inch"][0],
                height
                * params["post_edge_det_gauss_filt_std"]
                / params["paper_size_in_inch"][1],
            ]
        )
        edges_blurred = ndimage.gaussian_filter(edges.astype(float), blur_sigma)

        edges_blurred_sat = edges_blurred.copy()
        if params["post_edge_det_sat"]:
            upper = np.percentile(edges_blurred, params["sat_level_upper_prctile"])
            lower = np.percentile(edges_blurred, params["sat_level_lower_prctile"])
            edges_blurred_sat = np.clip(edges_blurred_sat, lower, upper)

        edges_blurred_sat = edges_blurred_sat / np.max(edges_blurred_sat)
        img_gray_normalized = 1.0 - (
            (edges_blurred_sat - np.min(edges_blurred_sat))
            / (np.max(edges_blurred_sat) - np.min(edges_blurred_sat))
        )

    if params["sat_pre_grid_det"]:
        img_sat = tanh_sat(
            1.0 - img_gray_normalized.flatten(),
            params["sat_level_pre_grid_det"],
            "ksigma",
        )
        img_gray_normalized = img_sat.reshape(img_gray_normalized.shape)

    seg_width = int(np.floor(width / params["num_seg_hor"]))
    seg_height = int(np.floor(height / params["num_seg_ver"]))

    def compute_spectrum(patch: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
        return np.abs(np.fft.fft2(patch, s=out_shape)) ** 2 / (out_shape[0] * out_shape[1])

    spectra = []
    rng = np.random.default_rng()

    if params["spectral_tiling_method"] == "REGULAR_TILING":
        sigma = params["gauss_win_sigma"] * np.mean([seg_width, seg_height])
        mask = (
            _gaussian_kernel((seg_height, seg_width), sigma)
            if params["smooth_spectra"]
            else np.ones((seg_height, seg_width))
        )
        for i in range(params["num_seg_ver"]):
            for j in range(params["num_seg_hor"]):
                patch = img_gray_normalized[
                    i * seg_height : (i + 1) * seg_height,
                    j * seg_width : (j + 1) * seg_width,
                ]
                spectra.append(compute_spectrum(mask * patch, (seg_height, seg_width)))
    elif params["spectral_tiling_method"] == "RANDOM_VAR_SIZE_TILING":
        for _ in range(params["total_segments"]):
            seg_width_rand = min(
                width - 1,
                seg_width + rng.integers(1, max(2, int(np.ceil(params["seg_width_rand_dev"] * seg_width)) + 1)),
            )
            seg_height_rand = min(
                height - 1,
                seg_height + rng.integers(1, max(2, int(np.ceil(params["seg_height_rand_dev"] * seg_height)) + 1)),
            )
            sigma = params["gauss_win_sigma"] * np.mean([seg_width_rand, seg_height_rand])
            mask = (
                _gaussian_kernel((seg_height_rand, seg_width_rand), sigma)
                if params["smooth_spectra"]
                else np.ones((seg_height_rand, seg_width_rand))
            )
            start_hor = rng.integers(0, width - seg_width_rand + 1)
            start_ver = rng.integers(0, height - seg_height_rand + 1)
            patch = img_gray_normalized[
                start_ver : start_ver + seg_height_rand,
                start_hor : start_hor + seg_width_rand,
            ]
            spectra.append(compute_spectrum(mask * patch, (seg_height, seg_width)))
    else:
        sigma = params["gauss_win_sigma"] * np.mean([seg_width, seg_height])
        mask = (
            _gaussian_kernel((seg_height, seg_width), sigma)
            if params["smooth_spectra"]
            else np.ones((seg_height, seg_width))
        )
        for _ in range(params["total_segments"]):
            start_hor = rng.integers(0, width - seg_width + 1)
            start_ver = rng.integers(0, height - seg_height + 1)
            patch = img_gray_normalized[
                start_ver : start_ver + seg_height,
                start_hor : start_hor + seg_width,
            ]
            spectra.append(compute_spectrum(mask * patch, (seg_height, seg_width)))

    spectra_stacked = np.stack(spectra, axis=2) if spectra else np.empty((0, 0, 0))

    if params["patch_avg_method"] == "MEDIAN":
        spectral_avg = np.median(spectra_stacked, axis=2)
    else:
        spectral_avg = np.mean(spectra_stacked, axis=2)

    spectral_avg_hor = 10 * np.log10(np.mean(spectral_avg, axis=1))
    spectral_avg_ver = 10 * np.log10(np.mean(spectral_avg, axis=0))

    spectral_avg_hor_zm = spectral_avg_hor
    spectral_avg_ver_zm = spectral_avg_ver

    pk_locs_hor, pk_props_hor = find_peaks(
        spectral_avg_hor_zm,
        distance=params["min_grid_resolution"],
        prominence=params["min_grid_peak_prominence"],
    )
    pk_locs_ver, pk_props_ver = find_peaks(
        spectral_avg_ver_zm,
        distance=params["min_grid_resolution"],
        prominence=params["min_grid_peak_prominence"],
    )

    ff_hor = np.arange(len(spectral_avg_hor_zm)) / len(spectral_avg_hor_zm)
    ff_ver = np.arange(len(spectral_avg_ver_zm)) / len(spectral_avg_ver_zm)

    if pk_locs_hor.size:
        mask = ff_hor[pk_locs_hor] < 0.5
        pk_locs_hor = pk_locs_hor[mask]
        pk_prom_hor = pk_props_hor["prominences"][mask]
        sort_idx_hor = np.argsort(pk_prom_hor)[::-1]
        grid_size_hor = 1.0 / ff_hor[pk_locs_hor[sort_idx_hor]]
    else:
        grid_size_hor = np.array([])

    if pk_locs_ver.size:
        mask = ff_ver[pk_locs_ver] < 0.5
        pk_locs_ver = pk_locs_ver[mask]
        pk_prom_ver = pk_props_ver["prominences"][mask]
        sort_idx_ver = np.argsort(pk_prom_ver)[::-1]
        grid_size_ver = 1.0 / ff_ver[pk_locs_ver[sort_idx_ver]]
    else:
        grid_size_ver = np.array([])

    if params["detailed_plots"] > 0:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(ff_hor, spectral_avg_hor_zm)
        if pk_locs_hor.size:
            plt.plot(ff_hor[pk_locs_hor], spectral_avg_hor_zm[pk_locs_hor], "ro")
            plt.plot(
                ff_hor[pk_locs_hor[sort_idx_hor[0]]],
                spectral_avg_hor_zm[pk_locs_hor[sort_idx_hor[0]]],
                "rx",
            )
        plt.plot(ff_ver, spectral_avg_ver_zm)
        if pk_locs_ver.size:
            plt.plot(ff_ver[pk_locs_ver], spectral_avg_ver_zm[pk_locs_ver], "ko")
            plt.plot(
                ff_ver[pk_locs_ver[sort_idx_ver[0]]],
                spectral_avg_ver_zm[pk_locs_ver[sort_idx_ver[0]]],
                "kx",
            )
        plt.grid(True)
        plt.title("Average spectral estimate across image patches")
        plt.xlabel("Grid repetition frequency (inverse of grid period in pixels)")
        plt.ylabel("Amplitude (dB)")

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

    return grid_size_hor, grid_size_ver
