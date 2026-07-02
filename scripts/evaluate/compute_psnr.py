#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute PSNR between reference images and generated images.

Example:
    python compute_psnr.py \
        --reference_dir ./dataset/groundtruth \
        --generated_dir ./results/palettegan \
        --target_size 300 \
        --output_csv ./results/psnr_results.csv \
        --output_npz ./results/psnr_results.npz
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_images(folder: Path):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(image_paths)


def normalize_stem(stem: str):
    suffixes = ["_fake", "_generated", "_output", "_result"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def build_stem_map(folder: Path):
    paths = find_images(folder)
    mapping = {}
    for path in paths:
        mapping[normalize_stem(path.stem)] = path
    return mapping


def load_rgb(path: Path, target_size: int = None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0


def compute_psnr(ref_rgb: np.ndarray, gen_rgb: np.ndarray):
    mse = np.mean((ref_rgb - gen_rgb) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR for paired reference/generated images.")
    parser.add_argument("--reference_dir", required=True, type=Path, help="Directory of reference/ground-truth images.")
    parser.add_argument("--generated_dir", required=True, type=Path, help="Directory of generated images.")
    parser.add_argument("--target_size", default=300, type=int, help="Resize images to target_size x target_size. Use 0 to disable resizing.")
    parser.add_argument("--output_csv", default="psnr_results.csv", type=Path, help="Path to save per-image CSV results.")
    parser.add_argument("--output_npz", default="psnr_results.npz", type=Path, help="Path to save NPZ results.")
    args = parser.parse_args()

    target_size = None if args.target_size == 0 else args.target_size

    ref_map = build_stem_map(args.reference_dir)
    gen_map = build_stem_map(args.generated_dir)

    common_stems = sorted(set(ref_map.keys()) & set(gen_map.keys()))
    if not common_stems:
        raise RuntimeError("No matched image pairs found. Please check file names and directories.")

    rows = []
    scores = []
    filenames = []

    for stem in common_stems:
        ref_path = ref_map[stem]
        gen_path = gen_map[stem]
        ref_rgb = load_rgb(ref_path, target_size)
        gen_rgb = load_rgb(gen_path, target_size)

        if ref_rgb.shape != gen_rgb.shape:
            raise ValueError(f"Image shape mismatch for {stem}: {ref_rgb.shape} vs {gen_rgb.shape}")

        score = compute_psnr(ref_rgb, gen_rgb)
        rows.append({
            "stem": stem,
            "reference": ref_path.name,
            "generated": gen_path.name,
            "psnr": score,
        })
        scores.append(score)
        filenames.append(stem)

    finite_scores = np.asarray([s for s in scores if np.isfinite(s)], dtype=np.float64)
    mean_psnr = float(np.mean(finite_scores)) if len(finite_scores) > 0 else float("inf")
    std_psnr = float(np.std(finite_scores)) if len(finite_scores) > 0 else 0.0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "reference", "generated", "psnr"])
        writer.writeheader()
        writer.writerows(rows)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, filenames=np.array(filenames, dtype=object), psnr=np.array(scores, dtype=np.float64))

    print(f"Matched pairs: {len(common_stems)}")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Std PSNR: {std_psnr:.4f}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved NPZ: {args.output_npz}")


if __name__ == "__main__":
    main()
