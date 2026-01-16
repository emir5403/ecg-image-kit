#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def parse_box(box_dict: Dict) -> Tuple[int, int, int, int]:
    points: List[Tuple[float, float]] = []
    for i in range(4):
        key = str(i) if str(i) in box_dict else i
        if key not in box_dict:
            raise KeyError(f"Bounding box missing point {i} in {box_dict}")
        y, x = box_dict[key]
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = int(max(0, min(xs)))
    min_y = int(max(0, min(ys)))
    max_x = int(max(xs))
    max_y = int(max(ys))
    return min_x, min_y, max_x, max_y


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def save_matrix(path: str, matrix: np.ndarray) -> None:
    np.save(path, matrix)


def draw_boxes(
    image: Image.Image,
    lead_box: Tuple[int, int, int, int],
    text_box: Tuple[int, int, int, int],
    lead_name: str,
) -> None:
    drawer = ImageDraw.Draw(image)
    drawer.rectangle(lead_box, outline=(255, 0, 0), width=3)
    if text_box is not None:
        drawer.rectangle(text_box, outline=(0, 128, 255), width=2)
    label_pos = (lead_box[0] + 4, max(0, lead_box[1] - 18))
    drawer.text(label_pos, lead_name, fill=(255, 0, 0))


def find_content_bounds(image: Image.Image, threshold: int) -> Tuple[int, int, int, int]:
    rgb = np.asarray(image)
    non_white = np.any(rgb < threshold, axis=-1)
    if not np.any(non_white):
        width, height = image.size
        return 0, 0, width, height
    ys, xs = np.where(non_white)
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    return min_x, min_y, max_x, max_y


def infer_lead_layout(
    image: Image.Image,
    lead_names: List[str],
    columns: int,
    include_full_strip: bool,
    threshold: int,
) -> List[Dict]:
    min_x, min_y, max_x, max_y = find_content_bounds(image, threshold)
    content_width = max_x - min_x
    content_height = max_y - min_y

    if columns <= 0:
        columns = 4 if len(lead_names) >= 12 else 1

    rows = int(np.ceil(len(lead_names) / columns))
    if include_full_strip:
        rows += 1

    cell_width = content_width / columns
    cell_height = content_height / rows

    leads = []
    for index, name in enumerate(lead_names):
        row = index // columns
        col = index % columns
        x1 = int(min_x + col * cell_width)
        x2 = int(min_x + (col + 1) * cell_width)
        y1 = int(min_y + row * cell_height)
        y2 = int(min_y + (row + 1) * cell_height)
        lead_box = {"0": [y1, x1], "1": [y1, x2], "2": [y2, x2], "3": [y2, x1]}

        text_x2 = int(x1 + cell_width * 0.25)
        text_y2 = int(y1 + cell_height * 0.25)
        text_box = {"0": [y1, x1], "1": [y1, text_x2], "2": [text_y2, text_x2], "3": [text_y2, x1]}

        leads.append(
            {
                "lead_name": name,
                "lead_bounding_box": lead_box,
                "text_bounding_box": text_box,
            }
        )

    if include_full_strip:
        full_y1 = int(min_y + (rows - 1) * cell_height)
        full_y2 = int(min_y + rows * cell_height)
        full_box = {"0": [full_y1, min_x], "1": [full_y1, max_x], "2": [full_y2, max_x], "3": [full_y2, min_x]}
        full_text_x2 = int(min_x + cell_width * 0.25)
        full_text_y2 = int(full_y1 + cell_height * 0.25)
        leads.append(
            {
                "lead_name": "full_strip",
                "lead_bounding_box": full_box,
                "text_bounding_box": {"0": [full_y1, min_x], "1": [full_y1, full_text_x2], "2": [full_text_y2, full_text_x2], "3": [full_text_y2, min_x]},
            }
        )

    return leads


def parse_lead_names(value: str) -> List[str]:
    if not value:
        return DEFAULT_LEADS
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names if names else DEFAULT_LEADS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract lead and lead-name bounding boxes from an ECG image using the "
            "generator config JSON, then save per-lead crops (matrices) and an "
            "annotated overview image."
        )
    )
    parser.add_argument("--image", required=True, help="Path to ECG image (.png/.jpg).")
    parser.add_argument(
        "--config",
        help=(
            "Optional path to config JSON containing lead_bounding_box and text_bounding_box "
            "entries (generated with --lead_bbox/--lead_name_bbox). If missing, the script "
            "will infer a layout from the image."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="lead_outputs",
        help="Directory to write cropped lead images, matrices, and annotated output.",
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
        help="Save per-lead cropped PNGs in addition to matrices.",
    )
    parser.add_argument(
        "--save_matrices",
        action="store_true",
        help="Save per-lead numpy matrices (.npy).",
    )
    parser.add_argument(
        "--lead_names",
        default="",
        help="Comma-separated lead names for layout inference (default: standard 12-lead order).",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=-1,
        help="Number of columns in the ECG layout (default: 4 for 12 leads, 1 otherwise).",
    )
    parser.add_argument(
        "--include_full_strip",
        action="store_true",
        help="Add a final row for a full-length lead strip when inferring layout.",
    )
    parser.add_argument(
        "--content_threshold",
        type=int,
        default=245,
        help="Threshold for detecting non-white pixels when inferring layout (0-255).",
    )

    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    image = Image.open(args.image).convert("RGB")
    config = load_config(args.config) if args.config else {}

    if "leads" not in config:
        lead_names = parse_lead_names(args.lead_names)
        config["leads"] = infer_lead_layout(
            image=image,
            lead_names=lead_names,
            columns=args.columns,
            include_full_strip=args.include_full_strip,
            threshold=args.content_threshold,
        )

    annotated = image.copy()
    leads_summary = []

    for index, lead in enumerate(config["leads"]):
        lead_name = lead.get("lead_name", f"lead_{index}")
        lead_box_dict = lead.get("lead_bounding_box")
        text_box_dict = lead.get("text_bounding_box")

        if lead_box_dict is None:
            raise KeyError(
                f"Lead {lead_name} missing lead_bounding_box. "
                "Regenerate the ECG image with --lead_bbox and --store_config or "
                "use --lead_names to infer a layout."
            )

        lead_box = parse_box(lead_box_dict)
        text_box = parse_box(text_box_dict) if text_box_dict else None

        crop = image.crop(lead_box)
        crop_matrix = np.asarray(crop)

        if args.save_crops:
            crop_path = os.path.join(args.output_dir, f"{lead_name}_lead.png")
            crop.save(crop_path)

        if args.save_matrices:
            matrix_path = os.path.join(args.output_dir, f"{lead_name}_matrix.npy")
            save_matrix(matrix_path, crop_matrix)

        draw_boxes(annotated, lead_box, text_box, lead_name)

        leads_summary.append(
            {
                "lead_name": lead_name,
                "lead_box": lead_box,
                "text_box": text_box,
                "matrix_shape": crop_matrix.shape,
            }
        )

    annotated_path = os.path.join(args.output_dir, "ecg_with_bounding_boxes.png")
    annotated.save(annotated_path)

    print("Extracted leads:")
    for summary in leads_summary:
        print(
            f"- {summary['lead_name']}: matrix shape {summary['matrix_shape']}, "
            f"lead box {summary['lead_box']}, text box {summary['text_box']}"
        )
    print(f"Annotated image saved to: {annotated_path}")


if __name__ == "__main__":
    main()
