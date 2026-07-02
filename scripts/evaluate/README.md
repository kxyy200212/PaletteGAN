# Evaluation Scripts

These Python scripts were converted and cleaned from the original evaluation notebooks.

## Files

```text
compute_fid.py
compute_psnr.py
compute_ssim.py
compute_palette_distance.py
compute_edge_f1.py
```

## Example commands

### FID

```bash
python compute_fid.py \
  --real_dir ./dataset/groundtruth \
  --generated_dir ./results/palettegan \
  --batch_size 32
```

### PSNR

```bash
python compute_psnr.py \
  --reference_dir ./dataset/groundtruth \
  --generated_dir ./results/palettegan \
  --target_size 300 \
  --output_csv ./results/psnr_results.csv \
  --output_npz ./results/psnr_results.npz
```

### SSIM

```bash
python compute_ssim.py \
  --reference_dir ./dataset/groundtruth \
  --generated_dir ./results/palettegan \
  --target_size 300 \
  --threshold 0.85 \
  --output_csv ./results/ssim_results.csv \
  --output_npz ./results/ssim_results.npz
```

### Palette Distance

```bash
python compute_palette_distance.py \
  --generated_dir ./results/palettegan \
  --palette_dir ./dataset/color_palette \
  --mask_dir ./dataset/mask_dc_hed_contour \
  --output_csv ./results/palette_distance.csv
```

### DC-HED-Contour Edge F1

```bash
python compute_edge_f1.py \
  --generated_sketch_dir ./results/palettegan_sketches \
  --reference_sketch_dir ./dataset/line_drawing_dc_hed_contour \
  --mask_dir ./dataset/mask_dc_hed_contour \
  --tolerance 2 \
  --output_csv ./results/edge_f1.csv
```
