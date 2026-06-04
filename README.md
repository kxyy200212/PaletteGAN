# PaletteGAN

PaletteGAN：基于调色板引导的服装草图着色项目。
![生成结果](./images/encoder.png)

本仓库包含 **“PaletteGAN：基于调色板引导的服装草图着色”** 项目的相关代码。

## 依赖环境

本项目需要 Python 3 环境，并安装以下依赖：

```bash
pip install torch torchvision numpy pillow matplotlib
```

## 下载模型

本项目在计算感知损失和风格损失时需要使用 VGG16 预训练模型。

请将 VGG16 模型文件放到项目目录中，并在 `palettegan.py` 中修改模型路径：

```python
vgg_weights_path = "./vgg16-397923af.pth"
```

## 数据集

数据集需要按照以下格式组织：

```text
dataset/
├── line_drawing_dc_hed_contour/
├── color_palette/
└── groundtruth/
```

线稿图、调色板文件和真实图像需要保持文件名对应。

例如：

```text
line_drawing_dc_hed_contour/000001.jpg
color_palette/000001.npy
groundtruth/000001.jpg
```

## 运行

运行训练脚本：

```bash
python palettegan.py
```

可以通过修改 `palettegan.py` 中的 `SELECTED_ABLATION_ID` 来选择不同的消融实验：

```python
SELECTED_ABLATION_ID = 0
```

## 测试

可以使用 `model_visualization.ipynb` 加载训练好的生成器模型，并对着色结果进行可视化展示。

## 结果

训练结果会保存在：

```text
ablation_experiments/
```

其中包括模型权重、生成图像和损失曲线。
