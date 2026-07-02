#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute SSIM between reference images and generated images.

The default setting follows the notebook logic: images are resized to 300x300,
converted to grayscale, and evaluated with skimage.metrics.structural_similarity.

Example:
    python compute_ssim.py \
        --reference_dir ./dataset/groundtruth \
        --generated_dir ./results/palettegan \
        --target_size 300 \
        --threshold 0.85 \
        --output_csv ./results/ssim_results.csv \
        --output_npz ./results/ssim_results.npz
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


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


def load_image(path: Path, target_size: int = None, mode: str = "gray"):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.BICUBIC)

    if mode == "gray":
        img = img.convert("L")
        return np.asarray(img, dtype=np.uint8)

    if mode == "rgb":
        return np.asarray(img, dtype=np.uint8)

    raise ValueError("mode must be 'gray' or 'rgb'.")


def compute_ssim_score(ref_img: np.ndarray, gen_img: np.ndarray, mode: str = "gray"):
    if ref_img.shape != gen_img.shape:
        raise ValueError(f"Image shape mismatch: {ref_img.shape} vs {gen_img.shape}")

    if mode == "gray":
        return float(ssim(ref_img, gen_img, win_size=11, gaussian_weights=True, sigma=1.5, data_range=255))

    return float(ssim(ref_img, gen_img, channel_axis=-1, win_size=11, gaussian_weights=True, sigma=1.5, data_range=255))


def main():
    parser = argparse.ArgumentParser(description="Compute SSIM for paired reference/generated images.")
    parser.add_argument("--reference_dir", required=True, type=Path, help="Directory of reference/ground-truth images.")
    parser.add_argument("--generated_dir", required=True, type=Path, help="Directory of generated images.")
    parser.add_argument("--target_size", default=300, type=int, help="Resize images to target_size x target_size. Use 0 to disable resizing.")
    parser.add_argument("--mode", default="gray", choices=["gray", "rgb"], help="Compute SSIM on grayscale or RGB images.")
    parser.add_argument("--threshold", default=0.85, type=float, help="Threshold for counting low-SSIM samples.")
    parser.add_argument("--output_csv", default="ssim_results.csv", type=Path, help="Path to save per-image CSV results.")
    parser.add_argument("--output_npz", default="ssim_results.npz", type=Path, help="Path to save NPZ results.")
    parser.add_argument("--low_ssim_txt", default=None, type=Path, help="Optional path to save filenames with SSIM below threshold.")
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
    low_stems = []

    for stem in common_stems:
        ref_path = ref_map[stem]
        gen_path = gen_map[stem]
        ref_img = load_image(ref_path, target_size, args.mode)
        gen_img = load_image(gen_path, target_size, args.mode)
        score = compute_ssim_score(ref_img, gen_img, args.mode)

        if score < args.threshold:
            low_stems.append(stem)

        rows.append({
            "stem": stem,
            "reference": ref_path.name,
            "generated": gen_path.name,
            "ssim": score,
        })
        scores.append(score)
        filenames.append(stem)

    scores_np = np.asarray(scores, dtype=np.float64)
    mean_ssim = float(np.mean(scores_np))
    std_ssim = float(np.std(scores_np))
    low_count = int(np.sum(scores_np < args.threshold))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "reference", "generated", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, filenames=np.array(filenames, dtype=object), ssim=scores_np)

    if args.low_ssim_txt is not None:
        args.low_ssim_txt.parent.mkdir(parents=True, exist_ok=True)
        args.low_ssim_txt.write_text("\n".join(low_stems), encoding="utf-8")

    print(f"Matched pairs: {len(common_stems)}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Std SSIM: {std_ssim:.4f}")
    print(f"SSIM < {args.threshold}: {low_count}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved NPZ: {args.output_npz}")


if __name__ == "__main__":
    main()
