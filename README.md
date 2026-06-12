# PaletteGAN

PaletteGAN: Palette-guided fashion sketch coloring.

This repository contains the code for **PaletteGAN**, a palette-guided fashion sketch coloring project based on PyTorch.

## Dependencies

This project requires Python 3 and the following dependencies:

```bash
pip install torch torchvision numpy pillow matplotlib
```

## Downloading Model

This project uses the VGG16 pretrained model to compute perceptual loss and style loss.

Please place the VGG16 model file in the project directory and modify the path in `palettegan.py`:

```python
vgg_weights_path = "./vgg16-397923af.pth"
```

## Dataset

The fashion dataset used in this project is downloaded from **Cleaned Maryland Dataset**:

https://github.com/AemikaChow/AiDLab-fAshIon-Data/blob/main/Datasets/cleaned-maryland.md

The original fashion images are processed into line drawings, color palette files, and ground truth images. The dataset should be organized as follows:

```text
dataset/
├── line_drawing_dc_hed_contour/
├── color_palette/
└── groundtruth/
```

The filenames of line drawings, color palettes, and ground truth images should correspond to each other.

For example:

```text
line_drawing_dc_hed_contour/000001.jpg
color_palette/000001.npy
groundtruth/000001.jpg
```

## Execute

Run the training script:

```bash
python palettegan.py
```

You can select different ablation experiments by changing `SELECTED_ABLATION_ID` in `palettegan.py`:

```python
SELECTED_ABLATION_ID = 0
```

## Testing

You can use `model_visualization.ipynb` to load the trained generator model and visualize the coloring results.

