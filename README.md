# PaletteGAN: Palette-Guided Controllable Fashion Sketch Colorization

PaletteGAN is a palette-guided framework for controllable fashion sketch colorization. Given a fashion sketch and a six-color palette, the model generates a colorized fashion image while preserving garment structure.

![Generated Result](./images/encoder.png)

This repository contains the implementation and reproducibility materials for the paper:

**PaletteGAN: Palette-Guided Controllable Fashion Sketch Colorization**

The repository provides training code, preprocessing scripts, evaluation scripts, a training configuration file, and fixed data split files for reproducible experiments.

---

## Overview

Fashion sketch colorization is a design-oriented image generation task. It converts fashion sketches and designer-specified palettes into colorized fashion renderings while preserving garment structures such as contours, folds, hemlines, cuffs, and local decorative details.

PaletteGAN takes two inputs:

1. a fashion sketch;
2. a six-color palette.

The model then generates a colorized fashion image. DC-HED-Contour sketches are used as structural conditions, and six-color palettes are used as explicit color conditions.

---

## Repository Structure

```text
PaletteGAN/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── palettegan.yaml
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── train_palettegan.py
├── scripts/
│   ├── preprocess/
│   │   ├── hue_shift_augmentation.ipynb
│   │   ├── add_contour.py
│   │   ├── dc-hed.py
│   │   ├── maxminc_palette.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   └── utils.py
│   └── evaluate/
│       ├── compute_fid.py
│       ├── compute_psnr.py
│       ├── compute_ssim.py
│       ├── compute_palette_distance.py
│       └── compute_edge_f1.py
├── docs/
│   └── dataset_preparation.md
├── checkpoints/
│   └── README.md
└── images/
    └── encoder.png
```

---

## Environment

The experiments were conducted on a 64-bit Linux high-performance computing server with an NVIDIA A30 GPU.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
Python
PyTorch
Torchvision
NumPy
OpenCV
Pillow
SciPy
Scikit-image
Scikit-learn
Matplotlib
Tqdm
```

---

## Dataset

The original fashion images are from the Dresses category of the Cleaned Maryland Dataset provided by the AiDLab-fAshIon-Data project:

```text
https://github.com/AemikaChow/AiDLab-fAshIon-Data/blob/main/Datasets/cleaned-maryland.md
```

The Cleaned Maryland Dataset is a cleaned and reorganized version of the Maryland Polyvore dataset.

Due to dataset redistribution restrictions, the original images are not included in this repository. Users should download the original dataset from the official source and follow the corresponding dataset license and access policy.

In this project, the processed dataset should be organized as:

```text
dataset/
├── line_drawing_dc_hed_contour/
├── color_palette/
└── groundtruth/
```

where:

```text
line_drawing_dc_hed_contour/ : DC-HED-Contour fashion sketches
color_palette/              : six-color palette files in .npy format
groundtruth/                 : real color fashion images
```

More details about dataset construction are provided in:

```text
docs/dataset_preparation.md
```

---

## Data Splits

The dataset partition used in the paper is fixed rather than randomly generated. Therefore, no random seed was used for dataset splitting.

The exact split files are provided in the `splits/` folder:

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

The split sizes are:

```text
Training set:   10,076 samples
Validation set:    593 samples
Test set:        1,778 samples
```

The test images are original, non-enhanced samples and do not contain repeated color variants of the 561 source images used to construct the training set.

Each split file contains one sample name per line.

---

## Preprocessing

The preprocessing stage includes:

1. brightness-preserving color enhancement;
2. DC-HED-Contour sketch extraction;
3. mask-based foreground processing;
4. six-color palette extraction using K-means clustering and MAXMIN color selection.

The preprocessing scripts are provided in:

```text
scripts/preprocess/
```

Main preprocessing files include:

```text
scripts/preprocess/hue_shift_augmentation.ipynb
scripts/preprocess/add_contour.py
scripts/preprocess/dc-hed.py
scripts/preprocess/maxminc_palette.py
scripts/preprocess/dataset.py
scripts/preprocess/models.py
scripts/preprocess/utils.py
```

The brightness-preserving color enhancement step generates color variants by changing hue while preserving the luminance channel. The DC-HED-Contour extraction step generates fashion sketches with enhanced garment structure. The palette extraction step constructs a fixed six-color palette for each fashion image.

---

## Training

The main training script is:

```text
train_palettegan.py
```

The training configuration used in the paper is provided in:

```text
configs/palettegan.yaml
```

The main training settings are:

```text
Epochs:                100
Batch size:            8
Initial learning rate: 0.0002
Optimizer:             Adam
Beta1:                 0.5
Beta2:                 0.999
Input resolution:      256 × 256
Palette size:          6 colors
Random seed:           2024
```

Run training with:

```bash
python train_palettegan.py
```

By default, the full PaletteGAN setting should use:

```python
SELECTED_ABLATION_ID = 0
```

Different ablation settings can be selected by modifying `SELECTED_ABLATION_ID` in `train_palettegan.py`.

---

## Reproducibility

All reported model training runs, including PaletteGAN, the ablation variants, and the adapted comparison baselines, used a fixed random seed of 2024 to improve reproducibility.

The seed was applied to:

```text
Python random number generator
NumPy random number generator
PyTorch random number generator
CUDA random number generator
DataLoader shuffling
```

Deterministic cuDNN behavior was enabled where applicable.

The dataset partition was fixed rather than randomly generated, and the exact split files are provided in the `splits/` folder.

---

## Evaluation

The paper reports the following metrics:

```text
FID
PSNR
SSIM
Palette Distance
DC-HED-Contour Edge F1
```

Evaluation scripts are provided in:

```text
scripts/evaluate/
```

Run FID evaluation with:

```bash
python scripts/evaluate/compute_fid.py \
  --real_dir ./dataset/groundtruth \
  --generated_dir ./results/palettegan \
  --batch_size 32
```

Run PSNR evaluation with:

```bash
python scripts/evaluate/compute_psnr.py \
  --reference_dir ./dataset/groundtruth \
  --generated_dir ./results/palettegan \
  --target_size 300 \
  --output_csv ./results/psnr_results.csv \
  --output_npz ./results/psnr_results.npz
```

Run SSIM evaluation with:

```bash
python scripts/evaluate/compute_ssim.py \
  --reference_dir ./dataset/groundtruth \
  --generated_dir ./results/palettegan \
  --target_size 300 \
  --threshold 0.85 \
  --output_csv ./results/ssim_results.csv \
  --output_npz ./results/ssim_results.npz
```

Run Palette Distance evaluation with:

```bash
python scripts/evaluate/compute_palette_distance.py \
  --generated_dir ./results/palettegan \
  --palette_dir ./dataset/color_palette \
  --mask_dir ./dataset/mask_dc_hed_contour \
  --output_csv ./results/palette_distance.csv
```

Run DC-HED-Contour Edge F1 evaluation with:

```bash
python scripts/evaluate/compute_edge_f1.py \
  --generated_sketch_dir ./results/palettegan_sketches \
  --reference_sketch_dir ./dataset/line_drawing_dc_hed_contour \
  --mask_dir ./dataset/mask_dc_hed_contour \
  --tolerance 2 \
  --output_csv ./results/edge_f1.csv
```

---

## Checkpoints

Large model checkpoint files are not included in this GitHub repository.

All checkpoint files used in this project are provided through Quark Cloud Drive:

https://pan.quark.cn/s/8013790cb21a

Please download the checkpoint files from the link above and place them in:

checkpoints/

The checkpoint package includes the following files:

checkpoints/bsds500_pascal_model.pth
checkpoints/palettegan.pth
checkpoints/vgg16-397923af.pth

File usage:

bsds500_pascal_model.pth is the DC-HED pretrained model used for DC-HED-Contour sketch extraction during preprocessing.

palettegan.pth is the trained PaletteGAN model used for inference and testing.

vgg16-397923af.pth is the VGG16 pretrained model used for perceptual loss, style loss, and perceptual color loss during training.

Users can also reproduce the model by running the training script with the fixed data splits and configuration provided in this repository.

---

## Outputs

Training outputs are saved in:

```text
ablation_experiments/
```

This folder may include:

```text
model checkpoints
generated images
loss curves
training logs
```

Generated images for evaluation can be organized as:

```text
results/
└── palettegan/
```

Large generated results are not included in this repository.

---

## Notes on Large Files

The following files and folders are not included in the Git repository:

```text
dataset/
data/
testdata/
*.pth
*.pt
*.ckpt
results/
outputs/
generated_images/
ablation_experiments/
```

These files are excluded to avoid redistributing large datasets, model checkpoints, and generated result folders.

---

## License

This project is released for academic research purposes. Please see the `LICENSE` file for details.

The original dataset is subject to its own license and access policy.

---

## Citation

If you use this code or find this work helpful, please cite the corresponding paper:

```text
PaletteGAN: Palette-Guided Controllable Fashion Sketch Colorization
```
