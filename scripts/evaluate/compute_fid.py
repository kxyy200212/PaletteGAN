#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute FID between real images and generated images using Inception-v3 pool features.

Example:
    python compute_fid.py \
        --real_dir ./dataset/groundtruth \
        --generated_dir ./results/palettegan \
        --batch_size 32
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_images(folder: Path):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(image_paths)


class ImageDataset(Dataset):
    def __init__(self, image_dir: Path, transform):
        self.image_paths = find_images(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def build_inception(device):
    try:
        weights = models.Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights, transform_input=False)
    except AttributeError:
        model = models.inception_v3(pretrained=True, transform_input=False)

    # Replace final classification layer with identity so output is 2048-d pool feature.
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def get_activations(image_dir: Path, batch_size: int, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_inception(device)

    activations = []
    for batch in loader:
        batch = batch.to(device)
        features = model(batch)
        features = features.detach().cpu().numpy()
        activations.append(features)

    return np.concatenate(activations, axis=0)


def compute_statistics(activations: np.ndarray):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def main():
    parser = argparse.ArgumentParser(description="Compute FID between real and generated images.")
    parser.add_argument("--real_dir", required=True, type=Path, help="Directory of real/reference images.")
    parser.add_argument("--generated_dir", required=True, type=Path, help="Directory of generated images.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for Inception feature extraction.")
    parser.add_argument("--device", default=None, type=str, help="cuda or cpu. If omitted, use cuda when available.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    real_acts = get_activations(args.real_dir, args.batch_size, device)
    gen_acts = get_activations(args.generated_dir, args.batch_size, device)

    mu_real, sigma_real = compute_statistics(real_acts)
    mu_gen, sigma_gen = compute_statistics(gen_acts)

    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    print(f"Real images: {len(real_acts)}")
    print(f"Generated images: {len(gen_acts)}")
    print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
