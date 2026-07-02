# Checkpoints

Large model checkpoint files are not included in this GitHub repository.

All checkpoint files used in this project are provided through Quark Cloud Drive:

https://pan.quark.cn/s/8013790cb21a

Please download the checkpoint files from the link above and place them in this folder:

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
