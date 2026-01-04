"""Test script for image_to_sequence to convert ECG images into time-series."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from image_to_sequence import image_to_sequence


def main() -> None:
    data_path = Path(__file__).resolve().parent / "../../sample-data/ecg-images/sample-segments"

    for file_path in sorted(data_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            img = np.array(Image.open(file_path))

            z0 = image_to_sequence(img, "dark-foreground", "max_finder", None, False)
            z1 = image_to_sequence(img, "dark-foreground", "hor_smoothing", 3)
            z2 = image_to_sequence(img, "dark-foreground", "all_left_right_neighbors")
            z3 = image_to_sequence(img, "dark-foreground", "combined_all_neighbors")
            z4 = image_to_sequence(img, "dark-foreground", "moving_average", 3)

            z_combined = np.median(np.vstack([z0, z1, z2, z3, z4]), axis=0)

            nn = np.arange(img.shape[1])
            img_height = img.shape[0]

            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.plot(nn, img_height - z0, linewidth=3, label="max_finder")
            plt.plot(nn, img_height - z1, linewidth=3, label="hor_smoothing")
            plt.plot(nn, img_height - z2, linewidth=3, label="all_left_right_neighbors")
            plt.plot(nn, img_height - z3, linewidth=3, label="combined_all_neighbors")
            plt.plot(nn, img_height - z4, linewidth=3, label="moving_average")
            plt.plot(nn, img_height - z_combined, linewidth=3, label="combined methods")
            plt.legend()
            plt.title(f"Paper ECG vs recovered signal for: {file_path.name}")
            plt.gca().tick_params(labelsize=14)
            plt.close()
        except Exception:
            continue


if __name__ == "__main__":
    main()
