# Dataset Preparation

## Source Dataset

The original images are from the Dresses category of the Cleaned Maryland Dataset provided by the AiDLab-fAshIon-Data project.

Due to dataset redistribution restrictions, the original images are not included in this repository. Users should download the original dataset from the official source and follow the corresponding dataset license and access policy.

## Training Set

We selected 561 original fashion images with complete garment bodies, minimal background interference, clear structures, and rich colors.

A brightness-preserving color enhancement method was applied to generate 18 color variants for each selected image by changing the hue in HSV space while preserving the original luminance channel in Lab space.

This process produced 10,098 candidate images. After removing 22 images with implausible colors, 10,076 images remained for training.

## Validation and Test Sets

The validation and test sets were independently selected from other original images in the Dresses category. No color enhancement was applied to either set.

The final split sizes are:

```text
Training set: 10,076 samples
Validation set: 593 samples
Test set: 1,778 samples
