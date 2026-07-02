#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute DC-HED-Contour-based Edge Precision, Recall, and F1.

The mask is used only to define the garment foreground region.
It is not used to add contour edges to the predicted structural map.

Example:
    python compute_edge_f1.py \
        --generated_sketch_dir ./results/palettegan_sketches \
        --reference_sketch_dir ./dataset/line_drawing_dc_hed_contour \
        --mask_dir ./dataset/mask_dc_hed_contour \
        --tolerance 2 \
        --output_csv ./results/edge_f1.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_images(folder: Path):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(image_paths)


def find_matching_file(folder: Path, stem: str):
    for ext in IMAGE_EXTS:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
        path_upper = folder / f"{stem}{ext.upper()}"
        if path_upper.exists():
            return path_upper
    return None


def normalize_stem(stem: str):
    suffixes = ["_fake", "_generated", "_output", "_result", "_sketch"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def load_mask(mask_path: Path, target_size=None, threshold=127):
    mask_img = Image.open(mask_path).convert("L")
    if target_size is not None and mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.NEAREST)
    mask_arr = np.asarray(mask_img)
    return mask_arr > threshold


def load_sketch_as_edges(sketch_path: Path, target_size=None, mode="auto", threshold=127):
    sketch_img = Image.open(sketch_path).convert("L")
    if target_size is not None and sketch_img.size != target_size:
        sketch_img = sketch_img.resize(target_size, Image.NEAREST)

    arr = np.asarray(sketch_img)

    if mode == "dark":
        edges = arr < threshold
    elif mode == "light":
        edges = arr > threshold
    elif mode == "auto":
        # Bright images are usually white background with dark lines.
        if arr.mean() > 127:
            edges = arr < threshold
        else:
            edges = arr > threshold
    else:
        raise ValueError("mode should be 'auto', 'dark', or 'light'.")

    return edges


def make_ellipse_kernel(radius: int):
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate_binary(edge_map: np.ndarray, radius: int):
    kernel = make_ellipse_kernel(radius)
    edge_uint8 = edge_map.astype(np.uint8) * 255
    dilated = cv2.dilate(edge_uint8, kernel, iterations=1)
    return dilated > 0


def compute_edge_f1(pred_edges, ref_edges, mask=None, tolerance=2):
    pred_edges = pred_edges.astype(bool)
    ref_edges = ref_edges.astype(bool)

    if mask is not None:
        mask = mask.astype(bool)
        pred_edges = pred_edges & mask
        ref_edges = ref_edges & mask

    pred_count = pred_edges.sum()
    ref_count = ref_edges.sum()

    if pred_count == 0 and ref_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or ref_count == 0:
        return 0.0, 0.0, 0.0

    ref_dilated = dilate_binary(ref_edges, tolerance)
    pred_dilated = dilate_binary(pred_edges, tolerance)

    matched_pred = pred_edges & ref_dilated
    precision = matched_pred.sum() / pred_count

    matched_ref = ref_edges & pred_dilated
    recall = matched_ref.sum() / ref_count

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)


def compute_one_sample(
    generated_sketch_path: Path,
    reference_sketch_path: Path,
    mask_path: Path,
    mask_threshold=127,
    sketch_line_mode="auto",
    sketch_threshold=127,
    tolerance=2,
):
    generated_img = Image.open(generated_sketch_path).convert("L")
    target_size = generated_img.size

    pred_edges = load_sketch_as_edges(
        generated_sketch_path,
        target_size=target_size,
        mode=sketch_line_mode,
        threshold=sketch_threshold,
    )

    ref_edges = load_sketch_as_edges(
        reference_sketch_path,
        target_size=target_size,
        mode=sketch_line_mode,
        threshold=sketch_threshold,
    )

    mask = load_mask(mask_path, target_size=target_size, threshold=mask_threshold)

    precision, recall, f1 = compute_edge_f1(
        pred_edges=pred_edges,
        ref_edges=ref_edges,
        mask=mask,
        tolerance=tolerance,
    )

    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "generated_edge_pixels": int((pred_edges & mask).sum()),
        "reference_edge_pixels": int((ref_edges & mask).sum()),
        "garment_pixels": int(mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute DC-HED-Contour-based Edge F1.")
    parser.add_argument("--generated_sketch_dir", required=True, type=Path, help="Directory of DC-HED-Contour sketches extracted from generated images.")
    parser.add_argument("--reference_sketch_dir", required=True, type=Path, help="Directory of input/reference DC-HED-Contour sketches.")
    parser.add_argument("--mask_dir", required=True, type=Path, help="Directory of garment foreground masks.")
    parser.add_argument("--mask_threshold", default=127, type=int, help="Mask threshold. Pixels > threshold are foreground.")
    parser.add_argument("--sketch_line_mode", default="auto", choices=["auto", "dark", "light"], help="How to detect edge pixels from sketch images.")
    parser.add_argument("--sketch_threshold", default=127, type=int, help="Threshold for binarizing sketch images.")
    parser.add_argument("--tolerance", default=2, type=int, help="Edge matching tolerance in pixels.")
    parser.add_argument("--output_csv", default="edge_f1.csv", type=Path, help="Path to save per-image CSV results.")
    args = parser.parse_args()

    generated_paths = find_images(args.generated_sketch_dir)
    if not generated_paths:
        raise FileNotFoundError(f"No generated sketches found in {args.generated_sketch_dir}")

    rows = []
    missing = []

    for generated_path in generated_paths:
        stem = normalize_stem(generated_path.stem)
        reference_path = find_matching_file(args.reference_sketch_dir, stem)
        mask_path = find_matching_file(args.mask_dir, stem)

        if reference_path is None:
            missing.append((generated_path.name, "reference_sketch", stem))
            continue
        if mask_path is None:
            missing.append((generated_path.name, "mask", stem))
            continue

        metrics = compute_one_sample(
            generated_sketch_path=generated_path,
            reference_sketch_path=reference_path,
            mask_path=mask_path,
            mask_threshold=args.mask_threshold,
            sketch_line_mode=args.sketch_line_mode,
            sketch_threshold=args.sketch_threshold,
            tolerance=args.tolerance,
        )

        row = {
            "stem": stem,
            "generated_sketch": generated_path.name,
            "reference_sketch": reference_path.name,
            "mask": mask_path.name,
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid samples were processed. Please check file names and paths.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem", "generated_sketch", "reference_sketch", "mask",
        "edge_precision", "edge_recall", "edge_f1",
        "generated_edge_pixels", "reference_edge_pixels", "garment_pixels",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    precision_values = np.array([r["edge_precision"] for r in rows], dtype=np.float64)
    recall_values = np.array([r["edge_recall"] for r in rows], dtype=np.float64)
    f1_values = np.array([r["edge_f1"] for r in rows], dtype=np.float64)

    print(f"Processed samples: {len(rows)}")
    print(f"Mean Edge Precision: {precision_values.mean():.4f}")
    print(f"Mean Edge Recall: {recall_values.mean():.4f}")
    print(f"Mean Edge F1: {f1_values.mean():.4f}")
    print(f"Saved CSV: {args.output_csv}")

    if missing:
        print(f"Missing matched files: {len(missing)}")
        for item in missing[:10]:
            print("  ", item)


if __name__ == "__main__":
    main()
