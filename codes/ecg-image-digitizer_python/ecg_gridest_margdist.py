"""ECG grid size estimation using marginal distributions."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

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


def _choose_k(features: np.ndarray, max_clusters: int) -> int:
    n_samples = features.shape[0]
    if n_samples < 2:
        return 1
    max_k = min(max_clusters, n_samples)
    if max_k < 2:
        return 1
    best_k = 2
    best_score = -np.inf
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(features)
        score = calinski_harabasz_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _grid_size_from_cluster(
    peak_amps: np.ndarray,
    peak_gaps: np.ndarray,
    params: dict,
    method_label: str,
) -> float:
    if peak_gaps.size == 0:
        return float("nan")

    features = np.column_stack([peak_amps, peak_gaps])
    optimal_k = _choose_k(features, params["max_clusters"])
    labels = KMeans(n_clusters=optimal_k, n_init=10, random_state=0).fit_predict(
        peak_amps.reshape(-1, 1)
    )

    if params["cluster_selection_method"] == "GAP_MIN_VAR":
        gap_vars = []
        for cluster_id in range(optimal_k):
            cluster_gaps = peak_gaps[labels == cluster_id]
            gap_vars.append(np.std(cluster_gaps) if cluster_gaps.size else np.inf)
        selected_cluster = int(np.argmin(gap_vars))
    elif params["cluster_selection_method"] == "MAX_AMP_PEAKS":
        cluster_medians = []
        for cluster_id in range(optimal_k):
            cluster_amps = peak_amps[labels == cluster_id]
            cluster_medians.append(
                np.median(cluster_amps) if cluster_amps.size else -np.inf
            )
        selected_cluster = int(np.argmax(cluster_medians))
    else:
        raise ValueError(f"Undefined cluster selection method: {method_label}")

    selected_gaps = peak_gaps[labels == selected_cluster]
    if selected_gaps.size == 0:
        return float("nan")

    lower, upper = np.percentile(
        selected_gaps,
        [50.0 - params["avg_quartile"] / 2, 50.0 + params["avg_quartile"] / 2],
    )
    within = selected_gaps[(selected_gaps >= lower) & (selected_gaps <= upper)]
    if within.size == 0:
        return float("nan")
    return float(np.mean(within))


def ecg_gridest_margdist(
    img: np.ndarray, params: dict | None = None
) -> tuple[
    float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Estimate grid size in ECG images using marginal distributions.
    """
    if params is None:
        params = {}

    params.setdefault("blur_sigma_in_inch", 1.0)
    params.setdefault("paper_size_in_inch", [11, 8.5])
    params.setdefault("remove_shadows", True)
    params.setdefault("apply_edge_detection", False)
    params.setdefault("cluster_peaks", True)
    params.setdefault("max_clusters", 3)
    params.setdefault("cluster_selection_method", "GAP_MIN_VAR")
    params.setdefault("avg_quartile", 50.0)
    params.setdefault("post_edge_det_gauss_filt_std", 0.01)
    params.setdefault("post_edge_det_sat", True)
    params.setdefault("sat_level_upper_prctile", 99.0)
    params.setdefault("sat_level_lower_prctile", 1.0)
    params.setdefault("sat_pre_grid_det", True)
    params.setdefault("sat_level_pre_grid_det", 0.7)
    params.setdefault("num_seg_hor", 4)
    params.setdefault("num_seg_ver", 4)
    params.setdefault("hist_grid_det_method", "RANDOM_TILING")
    params.setdefault("total_segments", 100)
    params.setdefault("min_grid_resolution", 1)
    params.setdefault("min_grid_peak_prom_prctile", 2.0)
    params.setdefault("detailed_plots", 0)

    if params["avg_quartile"] > 100.0:
        raise ValueError("avg_quartile parameter must be between 0 and 100")

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

    if params["hist_grid_det_method"] == "REGULAR_TILING":
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
        rng = np.random.default_rng()
        for _ in range(params["total_segments"]):
            start_hor = rng.integers(0, width - seg_width + 1)
            start_ver = rng.integers(0, height - seg_height + 1)
            segment = img_gray_normalized[
                start_ver : start_ver + seg_height,
                start_hor : start_hor + seg_width,
            ]
            segments.append(segment)

    peak_amps_hor = []
    peak_gaps_hor = []
    peak_amps_ver = []
    peak_gaps_ver = []

    for segment in segments:
        hist_hor = 1.0 - np.mean(segment, axis=1)
        min_prom = np.percentile(hist_hor, params["min_grid_peak_prom_prctile"]) - np.min(
            hist_hor
        )
        pk_amps_hor, pk_locs_hor = find_peaks(
            hist_hor,
            distance=params["min_grid_resolution"],
            prominence=min_prom,
        )
        if pk_locs_hor.size > 1:
            peak_amps_hor.extend(hist_hor[pk_locs_hor][1:])
            peak_gaps_hor.extend(np.diff(pk_locs_hor))

        hist_ver = 1.0 - np.mean(segment, axis=0)
        min_prom = np.percentile(hist_ver, params["min_grid_peak_prom_prctile"]) - np.min(
            hist_ver
        )
        pk_amps_ver, pk_locs_ver = find_peaks(
            hist_ver,
            distance=params["min_grid_resolution"],
            prominence=min_prom,
        )
        if pk_locs_ver.size > 1:
            peak_amps_ver.extend(hist_ver[pk_locs_ver][1:])
            peak_gaps_ver.extend(np.diff(pk_locs_ver))

    peak_amps_hor = np.asarray(peak_amps_hor, dtype=float)
    peak_gaps_hor = np.asarray(peak_gaps_hor, dtype=float)
    peak_amps_ver = np.asarray(peak_amps_ver, dtype=float)
    peak_gaps_ver = np.asarray(peak_gaps_ver, dtype=float)

    if not params["cluster_peaks"]:
        lower, upper = np.percentile(
            peak_gaps_hor,
            [50.0 - params["avg_quartile"] / 2, 50.0 + params["avg_quartile"] / 2],
        )
        subset = peak_gaps_hor[(peak_gaps_hor >= lower) & (peak_gaps_hor <= upper)]
        grid_size_hor = float(np.mean(subset)) if subset.size else float("nan")

        lower, upper = np.percentile(
            peak_gaps_ver,
            [50.0 - params["avg_quartile"] / 2, 50.0 + params["avg_quartile"] / 2],
        )
        subset = peak_gaps_ver[(peak_gaps_ver >= lower) & (peak_gaps_ver <= upper)]
        grid_size_ver = float(np.mean(subset)) if subset.size else float("nan")
    else:
        grid_size_hor = _grid_size_from_cluster(
            peak_amps_hor, peak_gaps_hor, params, "horizontal"
        )
        grid_size_ver = _grid_size_from_cluster(
            peak_amps_ver, peak_gaps_ver, params, "vertical"
        )

    if params["detailed_plots"] > 0:
        import matplotlib.pyplot as plt

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

        plt.figure()
        plt.hist(peak_gaps_hor)
        plt.title("Histogram of horizontal grid spacing estimate of all segments")

        plt.figure()
        plt.hist(peak_gaps_ver)
        plt.title("Histogram of vertical grid spacing estimate of all segments")

        plt.show()

    return (
        grid_size_hor,
        grid_size_ver,
        peak_gaps_hor,
        peak_gaps_ver,
        peak_amps_hor,
        peak_amps_ver,
    )
