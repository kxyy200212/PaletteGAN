#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Palette Distance for generated fashion images.

For each foreground garment pixel, this metric computes the nearest CIELAB
distance to the six colors in the input palette, and reports the foreground
average. Lower values indicate better palette adherence.

Example:
    python compute_palette_distance.py \
        --generated_dir ./results/palettegan \
        --palette_dir ./dataset/color_palette \
        --mask_dir ./dataset/mask_dc_hed_contour \
        --output_csv ./results/palette_distance.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
PALETTE_EXTS = [".npy"]


def find_images(folder: Path):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(image_paths)


def find_matching_file(folder: Path, stem: str, exts):
    for ext in exts:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
        path_upper = folder / f"{stem}{ext.upper()}"
        if path_upper.exists():
            return path_upper
    return None


def normalize_stem(stem: str):
    suffixes = ["_fake", "_generated", "_output", "_result"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def rgb_to_lab(rgb):
    """
    Convert RGB values in [0, 1] to CIELAB.
    Input shape: (..., 3)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, 1.0)

    mask = rgb > 0.04045
    rgb_linear = np.empty_like(rgb)
    rgb_linear[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb_linear[~mask] = rgb[~mask] / 12.92

    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    xyz = rgb_linear @ matrix.T
    xyz = xyz * 100.0

    white = np.array([95.047, 100.000, 108.883])
    xyz_scaled = xyz / white

    epsilon = 0.008856
    kappa = 903.3

    f = np.where(
        xyz_scaled > epsilon,
        np.cbrt(xyz_scaled),
        (kappa * xyz_scaled + 16) / 116,
    )

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def normalize_palette_rgb(palette):
    """
    Convert palette to RGB [0, 1].
    Supports [0, 255], [0, 1], and [-1, 1].
    """
    palette = np.asarray(palette, dtype=np.float64)

    if palette.shape != (6, 3):
        raise ValueError(f"Palette shape should be (6, 3), but got {palette.shape}")

    p_min = palette.min()
    p_max = palette.max()

    if p_min < 0 and p_max <= 1:
        palette = (palette + 1.0) / 2.0
    elif p_max <= 1.0:
        pass
    else:
        palette = palette / 255.0

    return np.clip(palette, 0.0, 1.0)


def load_image_rgb(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def load_mask(mask_path: Path, target_size=None, threshold=127):
    mask_img = Image.open(mask_path).convert("L")
    if target_size is not None and mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.NEAREST)
    mask_arr = np.asarray(mask_img)
    return mask_arr > threshold


def compute_palette_distance_for_one(image_path: Path, palette_path: Path, mask_path: Path, mask_threshold=127):
    rgb = load_image_rgb(image_path)
    h, w = rgb.shape[:2]

    mask = load_mask(mask_path, target_size=(w, h), threshold=mask_threshold)

    palette = np.load(palette_path)
    palette_rgb = normalize_palette_rgb(palette)

    fg_pixels_rgb = rgb[mask]
    if fg_pixels_rgb.shape[0] == 0:
        raise ValueError(f"No garment pixels found in mask: {mask_path}")

    fg_pixels_lab = rgb_to_lab(fg_pixels_rgb)
    palette_lab = rgb_to_lab(palette_rgb)

    distances = np.linalg.norm(
        fg_pixels_lab[:, None, :] - palette_lab[None, :, :],
        axis=-1,
    )
    min_distances = distances.min(axis=1)

    return float(min_distances.mean()), int(fg_pixels_rgb.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Compute Palette Distance for generated images.")
    parser.add_argument("--generated_dir", required=True, type=Path, help="Directory of generated images.")
    parser.add_argument("--palette_dir", required=True, type=Path, help="Directory containing six-color palette .npy files.")
    parser.add_argument("--mask_dir", required=True, type=Path, help="Directory containing garment foreground masks.")
    parser.add_argument("--mask_threshold", default=127, type=int, help="Mask threshold. Pixels > threshold are foreground.")
    parser.add_argument("--output_csv", default="palette_distance.csv", type=Path, help="Path to save per-image CSV results.")
    args = parser.parse_args()

    generated_paths = find_images(args.generated_dir)
    if not generated_paths:
        raise FileNotFoundError(f"No generated images found in {args.generated_dir}")

    rows = []
    missing = []
    scores = []

    for image_path in generated_paths:
        stem = normalize_stem(image_path.stem)
        palette_path = find_matching_file(args.palette_dir, stem, PALETTE_EXTS)
        mask_path = find_matching_file(args.mask_dir, stem, IMAGE_EXTS)

        if palette_path is None:
            missing.append((image_path.name, "palette", stem))
            continue
        if mask_path is None:
            missing.append((image_path.name, "mask", stem))
            continue

        score, num_pixels = compute_palette_distance_for_one(
            image_path=image_path,
            palette_path=palette_path,
            mask_path=mask_path,
            mask_threshold=args.mask_threshold,
        )

        rows.append({
            "stem": stem,
            "generated_image": image_path.name,
            "palette_file": palette_path.name,
            "mask_file": mask_path.name,
            "palette_distance": score,
            "foreground_pixels": num_pixels,
        })
        scores.append(score)

    if not rows:
        raise RuntimeError("No valid samples were processed. Please check file names and paths.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["stem", "generated_image", "palette_file", "mask_file", "palette_distance", "foreground_pixels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed samples: {len(rows)}")
    print(f"Mean Palette Distance: {np.mean(scores):.4f}")
    print(f"Std Palette Distance: {np.std(scores):.4f}")
    print(f"Saved CSV: {args.output_csv}")

    if missing:
        print(f"Missing matched files: {len(missing)}")
        for item in missing[:10]:
            print("  ", item)


if __name__ == "__main__":
    main()
