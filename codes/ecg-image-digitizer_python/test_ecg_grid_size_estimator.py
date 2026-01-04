"""Test script for multiple ECG grid size estimation algorithms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ecg_grid_size_from_paper import ecg_grid_size_from_paper
from ecg_gridest_margdist import ecg_gridest_margdist
from ecg_gridest_spectral import ecg_gridest_spectral
from ecg_gridest_matchedfilt import ecg_gridest_matchedfilt


def main() -> None:
    data_path = Path(__file__).resolve().parent / "../../sample-data/ecg-images"
    all_files = sorted(data_path.iterdir())

    for file_path in all_files:
        if not file_path.is_file():
            continue
        try:
            img = np.array(Image.open(file_path))

            paper_size = [11.0, 8.5]
            _, fine_grid_size_paper_based = ecg_grid_size_from_paper(
                img, paper_size[0], "in"
            )

            params_margdist = {
                "blur_sigma_in_inch": 1.0,
                "paper_size_in_inch": paper_size,
                "remove_shadows": True,
                "apply_edge_detection": False,
                "post_edge_det_gauss_filt_std": 0.01,
                "post_edge_det_sat": True,
                "sat_level_upper_prctile": 99.0,
                "sat_level_lower_prctile": 1.0,
                "sat_pre_grid_det": False,
                "sat_level_pre_grid_det": 0.7,
                "num_seg_hor": 4,
                "num_seg_ver": 4,
                "hist_grid_det_method": "RANDOM_TILING",
                "total_segments": 100,
                "min_grid_resolution": 1,
                "min_grid_peak_prom_prctile": 2.0,
                "cluster_peaks": True,
                "max_clusters": 3,
                "cluster_selection_method": "GAP_MIN_VAR",
                "avg_quartile": 50.0,
                "detailed_plots": 1,
            }
            (
                gridsize_hor_margdist,
                gridsize_ver_margdist,
                _,
                _,
                _,
                _,
            ) = ecg_gridest_margdist(img, params_margdist)

            params_spectral = {
                "blur_sigma_in_inch": 1.0,
                "paper_size_in_inch": paper_size,
                "remove_shadows": True,
                "apply_edge_detection": False,
                "post_edge_det_gauss_filt_std": 0.01,
                "post_edge_det_sat": False,
                "sat_level_upper_prctile": 99.0,
                "sat_level_lower_prctile": 1.0,
                "sat_pre_grid_det": False,
                "sat_level_pre_grid_det": 0.7,
                "num_seg_hor": 4,
                "num_seg_ver": 4,
                "spectral_tiling_method": "RANDOM_TILING",
                "total_segments": 100,
                "min_grid_resolution": 1,
                "min_grid_peak_prominence": 1.0,
                "detailed_plots": 1,
            }
            gridsize_hor_spectral, gridsize_ver_spectral = ecg_gridest_spectral(
                img, params_spectral
            )

            if gridsize_hor_spectral.size:
                closest_ind_hor = int(
                    np.argmin(np.abs(gridsize_hor_spectral - fine_grid_size_paper_based))
                )
            else:
                closest_ind_hor = None
            if gridsize_ver_spectral.size:
                closest_ind_ver = int(
                    np.argmin(np.abs(gridsize_ver_spectral - fine_grid_size_paper_based))
                )
            else:
                closest_ind_ver = None

            params_matchfilt = params_margdist.copy()
            params_matchfilt["sat_pre_grid_det"] = True
            params_matchfilt["sat_level_pre_grid_det"] = 0.7
            params_matchfilt["total_segments"] = 10
            params_matchfilt["tiling_method"] = "RANDOM_TILING"
            (
                grid_sizes_matchedfilt,
                _,
                _,
                _,
                _,
            ) = ecg_gridest_matchedfilt(img, params_matchfilt)

            print(
                "Grid resolution estimate per 0.1mV x 40ms (paper size-based): "
                f"{fine_grid_size_paper_based} pixels"
            )
            print(
                "Grid resolution estimates per 0.1mV x 40ms (matched filter-based): "
                f"[{grid_sizes_matchedfilt}] pixels"
            )

            print(
                f"Horizontal grid resolution estimate (margdist): {gridsize_hor_margdist} pixels"
            )
            print(
                f"Vertical grid resolution estimate (margdist): {gridsize_ver_margdist} pixels"
            )

            print(
                f"Horizontal grid resolution estimate (spectral): [{gridsize_hor_spectral}] pixels"
            )
            print(
                f"Vertical grid resolution estimate (spectral): [{gridsize_ver_spectral}] pixels"
            )

            if closest_ind_hor is not None:
                print(
                    "Closest spectral horizontal grid resolution estimate from paper-based "
                    f"resolution (per 0.1mV x 40ms): {gridsize_hor_spectral[closest_ind_hor]} pixels"
                )
            if closest_ind_ver is not None:
                print(
                    "Closest spectral vertical grid resolution estimate from paper-based "
                    f"resolution (per 0.1mV x 40ms): {gridsize_ver_spectral[closest_ind_ver]} pixels"
                )

            print("---")
        except Exception:
            continue


if __name__ == "__main__":
    main()
