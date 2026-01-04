"""Estimate ECG grid size based on paper size."""
from __future__ import annotations

import numpy as np


def ecg_grid_size_from_paper(
    img: np.ndarray, paper_width: float, unit: str
) -> tuple[float, float]:
    """
    Estimate coarse and fine grid size for an ECG image.

    Args:
        img: Image array.
        paper_width: Paper width corresponding to image width.
        unit: "cm" or "in".

    Returns:
        coarse_grid_res, fine_grid_res in pixels.
    """
    width = img.shape[1]
    if unit.lower() == "cm":
        paper_width_in_inch = paper_width / 2.54
    else:
        paper_width_in_inch = paper_width

    pxls_per_inch = width / paper_width_in_inch
    coarse_grid_res = pxls_per_inch * 5 / 25.4
    fine_grid_res = coarse_grid_res / 5
    return coarse_grid_res, fine_grid_res
