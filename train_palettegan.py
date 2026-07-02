import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
import shutil
from datetime import datetime
import torchvision.models as models
import math
import random


# ---------------------- 随机种子设置 ----------------------
def set_seed(seed=2024):
    """
    固定 Python、NumPy、PyTorch 和 CUDA 的随机种子，尽量提高训练复现性。
    注意：不同 CUDA/cuDNN 版本和不同硬件上，仍可能存在极小的非确定性差异。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    固定 DataLoader 每个 worker 的随机种子，用于配合 shuffle=True 时提高复现性。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------- 消融实验配置 ----------------------
def get_ablation_configs():
    """定义所有消融实验组的配置（可手动选择序号）"""
    # 基础损失权重（与原模型一致）
    base_weights = {
        'gan': 1.0,
        'l1': 100.0,
        'mse': 100.0,  # 用于L1/MSE替换实验
        'perceptual_color': 10.0,
        'ssim': 20.0,
        'style': 50.0,
        'perceptual': 30.0
    }

    return [
        # 0: 基准组（所有损失都启用）
        {
            'id': 0,
            'name': 'baseline_all_losses',
            'enabled_losses': ['gan', 'l1', 'perceptual_color', 'ssim', 'style', 'perceptual'],
            'weights': base_weights,
            'desc': '基准组：启用所有损失项'
        },
        # 1: 移除感知颜色损失
        {
            'id': 1,
            'name': 'ablation_no_perceptual_color',
            'enabled_losses': ['gan', 'l1', 'ssim', 'style', 'perceptual'],
            'weights': base_weights,
            'desc': '消融组：移除感知颜色损失'
        },
        # 2: 移除SSIM损失
        {
            'id': 2,
            'name': 'ablation_no_ssim',
            'enabled_losses': ['gan', 'l1', 'perceptual_color', 'style', 'perceptual'],
            'weights': base_weights,
            'desc': '消融组：移除SSIM损失'
        },
        # 3: 移除风格损失
        {
            'id': 3,
            'name': 'ablation_no_style',
            'enabled_losses': ['gan', 'l1', 'perceptual_color', 'ssim', 'perceptual'],
            'weights': base_weights,
            'desc': '消融组：移除风格损失'
        },
        # 4: 移除感知损失
        {
            'id': 4,
            'name': 'ablation_no_perceptual',
            'enabled_losses': ['gan', 'l1', 'perceptual_color', 'ssim', 'style'],
            'weights': base_weights,
            'desc': '消融组：移除感知损失'
        },
        # 5: 仅基础损失（GAN+L1）
        {
            'id': 5,
            'name': 'ablation_only_gan_l1',
            'enabled_losses': ['gan', 'l1'],
            'weights': base_weights,
            'desc': '消融组：仅保留GAN+L1基础损失'
        },
        # 6: L1替换为MSE
        {
            'id': 6,
            'name': 'ablation_l1_to_mse',
            'enabled_losses': ['gan', 'mse', 'perceptual_color', 'ssim', 'style', 'perceptual'],
            'weights': base_weights,
            'desc': '消融组：将L1损失替换为MSE损失'
        }
    ]


# 自定义数据集类
class FashionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.ids = [f.split('.')[0] for f in os.listdir(os.path.join(root_dir, 'line_drawing_dc_hed_contour'))
                    if f.endswith('.jpg')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.ids[idx]

        # 加载线条图
        line_path = os.path.join(self.root_dir, 'line_drawing_dc_hed_contour', f'{img_id}.jpg')
        line_drawing = Image.open(line_path).convert('L')

        # 加载调色板
        palette_path = os.path.join(self.root_dir, 'color_palette', f'{img_id}.npy')
        color_palette = np.load(palette_path)

        # 加载真实图像
        gt_path = os.path.join(self.root_dir, 'groundtruth', f'{img_id}.jpg')
        groundtruth = Image.open(gt_path).convert('RGB')

        # 应用数据变换
        if self.transform:
            line_drawing = self.transform['line'](line_drawing)
            groundtruth = self.transform['gt'](groundtruth)

        # 调色板归一化
        palette_tensor = torch.from_numpy(color_palette).float() / 255.0

        return {
            'line': line_drawing,
            'palette': palette_tensor,
            'gt': groundtruth,
            'id': img_id
        }


# 生成器模型
class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3):
        super(Generator, self).__init__()

        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 调色板特征处理
        self.palette_fc = nn.Sequential(
            nn.Linear(3 * 6, 256),  # 6色调色板
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True)
        )

        # 解码器部分
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, line, palette):
        # 编码器前向传播
        e1 = self.enc1(line)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 处理调色板特征
        batch_size = palette.size(0)
        palette_flat = palette.view(batch_size, -1)
        p_feat = self.palette_fc(palette_flat)
        p_feat = p_feat.view(batch_size, 512, 1, 1)
        p_feat = F.interpolate(p_feat, size=e4.size()[2:], mode='nearest')

        # 解码器前向传播（带跳跃连接）
        d1 = self.dec1(torch.cat([e4, p_feat], 1))
        d2 = self.dec2(torch.cat([d1, e3], 1))
        d3 = self.dec3(torch.cat([d2, e2], 1))
        output = self.dec4(torch.cat([d3, e1], 1))

        return output


# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3):
        super(Discriminator, self).__init__()

        # 线条图特征提取
        self.line_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 图像特征提取
        self.img_conv = nn.Sequential(
            nn.Conv2d(output_channels, 3, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 调色板特征处理
        self.palette_fc = nn.Sequential(
            nn.Linear(3 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

        # 主判别器网络
        self.main = nn.Sequential(
            nn.Conv2d(32 + 3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 + 256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, line, palette, img):
        # 提取线条和图像特征
        line_feat = self.line_conv(line)
        img_feat = self.img_conv(img)

        # 处理调色板特征
        batch_size = palette.size(0)
        palette_flat = palette.view(batch_size, -1)
        p_feat = self.palette_fc(palette_flat)
        p_feat = p_feat.view(batch_size, 256, 1, 1)
        p_feat = F.interpolate(p_feat, size=line_feat.size()[2:], mode='nearest')

        # 融合特征并判别
        x = torch.cat([line_feat, img_feat], 1)
        x = self.main[0](x)
        x = self.main[1](x)

        p_feat = F.interpolate(p_feat, size=x.size()[2:], mode='nearest')
        x = torch.cat([x, p_feat], 1)

        for layer in self.main[2:]:
            x = layer(x)

        return x


# 颜色空间转换工具
def rgb_to_hsv(rgb):
    """将RGB图像（[-1,1]或[0,1]）转为HSV空间（H:[0,1], S:[0,1], V:[0,1]）"""
    # 归一化到[0,1]范围
    if rgb.max() > 1.0:
        rgb = (rgb + 1.0) / 2.0  # 转换Tanh输出的[-1,1]到[0,1]

    r, g, b = rgb.unbind(dim=1)  # 分离通道
    max_rgb, _ = torch.max(rgb, dim=1, keepdim=True)  # 最大值
    min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)  # 最小值
    delta = max_rgb - min_rgb  # 亮度差

    # 计算H通道（色调）
    h = torch.zeros_like(max_rgb)
    mask_r = (max_rgb == r) & (delta > 1e-6)
    h[mask_r] = (((g[mask_r.squeeze()] - b[mask_r.squeeze()]) / delta[mask_r]) % 6) / 6.0
    mask_g = (max_rgb == g) & (delta > 1e-6)
    h[mask_g] = (((b[mask_g.squeeze()] - r[mask_g.squeeze()]) / delta[mask_g]) + 2) / 6.0
    mask_b = (max_rgb == b) & (delta > 1e-6)
    h[mask_b] = (((r[mask_b.squeeze()] - g[mask_b.squeeze()]) / delta[mask_b]) + 4) / 6.0

    # 计算S通道（饱和度）
    s = torch.where(delta > 1e-6, delta / max_rgb, torch.zeros_like(delta))

    # 计算V通道（亮度）
    v = max_rgb

    return torch.cat([h, s, v], dim=1)


# 感知网络（修复权重加载问题）
class PerceptualNetwork(nn.Module):
    def __init__(self, vgg_weights_path=None):
        super().__init__()
        # 加载VGG16的特征提取部分（不含分类器）
        self.vgg = models.vgg16(weights=None).features  # 修复pretrained deprecated警告
        # 加载并处理本地权重
        if vgg_weights_path:
            if not os.path.exists(vgg_weights_path):
                raise FileNotFoundError(f"VGG权重文件不存在: {vgg_weights_path}")

            # 加载原始权重
            state_dict = torch.load(vgg_weights_path, map_location=torch.device('cpu'))

            # 处理权重键名：过滤分类器层并移除features.前缀
            processed_state_dict = {}
            for key, value in state_dict.items():
                # 跳过分类器层（不需要）
                if key.startswith('classifier.'):
                    continue
                # 移除features.前缀以匹配模型结构
                if key.startswith('features.'):
                    new_key = key[len('features.'):]  # 例如 "features.0.weight" -> "0.weight"
                    processed_state_dict[new_key] = value

            # 加载处理后的权重
            self.vgg.load_state_dict(processed_state_dict)
            print(f"成功加载并处理VGG权重: {vgg_weights_path}")

        else:
            # 尝试从默认缓存路径加载
            default_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
            if os.path.exists(default_path):
                state_dict = torch.load(default_path, map_location=torch.device('cpu'))
                processed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('classifier.'):
                        continue
                    if key.startswith('features.'):
                        new_key = key[len('features.'):]
                        processed_state_dict[new_key] = value
                self.vgg.load_state_dict(processed_state_dict)
                print(f"成功从默认路径加载VGG权重: {default_path}")
            else:
                raise FileNotFoundError(
                    f"未找到VGG权重文件，请手动下载并指定路径。\n"
                    f"下载地址: https://download.pytorch.org/models/vgg16-397923af.pth\n"
                    f"建议保存路径: {default_path}"
                )

        # 选择用于不同损失计算的层
        self.color_layer = nn.Sequential(*list(self.vgg)[:3])  # 用于颜色感知损失
        self.perceptual_layers = nn.Sequential(*list(self.vgg)[:23])  # 用于整体感知损失
        self.style_layers = [3, 8, 15, 22]  # 用于风格损失的层索引

        # 冻结所有参数（仅作为特征提取器）
        for param in self.parameters():
            param.requires_grad = False

        # VGG预训练模型的归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def normalize(self, x):
        """将输入图像归一化以适应VGG网络"""
        x = (x + 1) / 2  # 将[-1, 1]转换为[0, 1]
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, output_type='perceptual'):
        """根据需要返回不同类型的特征"""
        x_norm = self.normalize(x)

        if output_type == 'color':
            return self.color_layer(x_norm)
        elif output_type == 'perceptual':
            return self.perceptual_layers(x_norm)
        elif output_type == 'style':
            features = []
            x_current = x_norm
            for i, layer in enumerate(self.perceptual_layers):
                x_current = layer(x_current)
                if i in self.style_layers:
                    features.append(x_current)
            return features


# 损失函数定义
def perceptual_color_loss(fake, gt, perceptual_net):
    """感知颜色损失：利用VGG早期层约束颜色感知一致性"""
    fake_feat = perceptual_net(fake, output_type='color')
    gt_feat = perceptual_net(gt, output_type='color')
    return F.mse_loss(fake_feat, gt_feat)


def ssim_loss(fake, gt, window_size=11, size_average=True):
    """自定义SSIM损失（不依赖torchmetrics）"""
    device = fake.device
    C = fake.size(1)  # 通道数

    # 创建高斯窗口
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * (window_size // 2) ** 2))
                          for x in range(window_size)]).to(device)
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    # 计算均值
    mu1 = F.conv2d(fake, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=C)

    # 计算方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(fake * fake, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(fake * gt, window, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM公式（输入范围[-1,1]，动态范围=2）
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)


def gram_matrix(x):
    """计算特征图的Gram矩阵（用于风格损失）"""
    B, C, H, W = x.size()
    features = x.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2)) / (C * H * W)  # 归一化
    return gram


def style_loss(fake, gt, perceptual_net):
    """风格损失：匹配生成图与真实图的风格特征"""
    fake_features = perceptual_net(fake, output_type='style')
    gt_features = perceptual_net(gt, output_type='style')

    total_loss = 0.0
    for fake_feat, gt_feat in zip(fake_features, gt_features):
        fake_gram = gram_matrix(fake_feat)
        gt_gram = gram_matrix(gt_feat)
        total_loss += F.mse_loss(fake_gram, gt_gram)

    return total_loss


def perceptual_loss(fake, gt, perceptual_net):
    """感知损失：匹配生成图与真实图的高层视觉特征"""
    fake_feat = perceptual_net(fake, output_type='perceptual')
    gt_feat = perceptual_net(gt, output_type='perceptual')
    return F.mse_loss(fake_feat, gt_feat)


# 实验目录创建
def create_directories(exp_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("ablation_experiments", f"{exp_name}_{timestamp}")  # 改为消融实验专用目录
    model_dir = os.path.join(exp_dir, "checkpoint03")
    images_dir = os.path.join(exp_dir, "generated_images")
    logs_dir = os.path.join(exp_dir, "logs03")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return exp_dir, model_dir, images_dir, logs_dir


# ---------------------- 主训练函数（修复变量未定义问题） ----------------------
def train_model(
        dataset_path,
        ablation_config,  # 新增：消融实验配置
        exp_name="fashion_pix2pix",
        epochs=100,
        batch_size=8,
        lr=0.0002,
        vgg_weights_path=None,
        load_gen_weight_path=None,
        seed=2024
):
    # 固定随机种子
    if seed is not None:
        set_seed(seed)
        print(f"✅ 已固定随机种子: {seed}")
    else:
        print("⚠️ 当前训练未固定随机种子")

    # 实验名称添加消融组标识
    exp_name = f"{exp_name}_ablation_{ablation_config['id']}_{ablation_config['name']}"
    exp_dir, model_dir, images_dir, logs_dir = create_directories(exp_name)

    # 保存消融配置到实验目录
    with open(os.path.join(exp_dir, "ablation_config.txt"), 'w', encoding='utf-8') as f:
        f.write(f"实验组ID: {ablation_config['id']}\n")
        f.write(f"实验名称: {ablation_config['name']}\n")
        f.write(f"实验描述: {ablation_config['desc']}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"启用损失项: {ablation_config['enabled_losses']}\n")
        f.write(f"损失权重: {ablation_config['weights']}\n")

    print(f"=== 启动消融实验 ===")
    print(f"实验配置：{ablation_config['desc']}")
    print(f"实验目录: {exp_dir}\n")

    # 数据变换（不变）
    transform = {
        'line': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'gt': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 数据加载
    dataset = FashionDataset(dataset_path, transform)

    # 使用 torch.Generator 固定 DataLoader 的 shuffle 顺序
    if seed is not None:
        data_generator = torch.Generator()
        data_generator.manual_seed(seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=data_generator
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

    # 设备配置（不变）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    if load_gen_weight_path is not None and os.path.exists(load_gen_weight_path):
        generator.load_state_dict(torch.load(load_gen_weight_path, map_location=device))
        print(f"✅ 成功加载已有生成器权重：{load_gen_weight_path}")
    else:
        print(f"ℹ️  未加载已有权重，将从头训练")

    discriminator = Discriminator().to(device)
    perceptual_net = PerceptualNetwork(vgg_weights_path=vgg_weights_path).to(device)

    # 损失函数与优化器（新增MSE损失）
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()  # 用于L1/MSE替换实验
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 损失记录（不变）
    loss_history = {
        'loss_d': [], 'loss_g': [], 'loss_g_gan': [], 'loss_g_l1': [], 'loss_g_mse': [],
        'loss_g_perceptual_color': [], 'loss_g_ssim': [],
        'loss_g_style': [], 'loss_g_perceptual': []
    }

    # 训练循环
    for epoch in range(epochs):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_loss_g_gan = 0.0
        epoch_loss_g_l1 = 0.0
        epoch_loss_g_mse = 0.0
        epoch_loss_g_perceptual_color = 0.0
        epoch_loss_g_ssim = 0.0
        epoch_loss_g_style = 0.0
        epoch_loss_g_perceptual = 0.0

        for i, batch in enumerate(dataloader):
            line = batch['line'].to(device)
            palette = batch['palette'].to(device)
            gt = batch['gt'].to(device)
            batch_size = line.size(0)

            # 真实/伪造标签
            real_label = torch.ones(batch_size, 1, 15, 15).to(device)
            fake_label = torch.zeros(batch_size, 1, 15, 15).to(device)

            # ---------------------- 训练判别器（不变） ----------------------
            optimizer_d.zero_grad()

            # 真实样本损失
            output_real = discriminator(line, palette, gt)
            loss_d_real = criterion_gan(output_real, real_label)

            # 伪造样本损失
            fake = generator(line, palette)
            output_fake = discriminator(line, palette, fake.detach())
            loss_d_fake = criterion_gan(output_fake, fake_label)

            # 总判别器损失
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()

            # ---------------------- 训练生成器（修复变量未定义问题） ----------------------
            optimizer_g.zero_grad()

            output = discriminator(line, palette, fake)
            loss_g_total = 0.0
            weights = ablation_config['weights']

            # 初始化所有损失变量为0（关键修复）
            loss_g_gan = torch.tensor(0.0).to(device)
            loss_g_l1 = torch.tensor(0.0).to(device)
            loss_g_mse = torch.tensor(0.0).to(device)
            loss_g_pcolor = torch.tensor(0.0).to(device)
            loss_g_ssim_val = torch.tensor(0.0).to(device)
            loss_g_style_val = torch.tensor(0.0).to(device)
            loss_g_perceptual_val = torch.tensor(0.0).to(device)

            # 1. GAN损失（根据配置启用）
            if 'gan' in ablation_config['enabled_losses']:
                loss_g_gan = criterion_gan(output, real_label)
                loss_g_total += weights['gan'] * loss_g_gan
                epoch_loss_g_gan += loss_g_gan.item()

            # 2. 像素级损失（L1或MSE，根据配置选择）
            if 'l1' in ablation_config['enabled_losses']:
                loss_g_l1 = criterion_l1(fake, gt)
                loss_g_total += weights['l1'] * loss_g_l1
                epoch_loss_g_l1 += loss_g_l1.item()
            elif 'mse' in ablation_config['enabled_losses']:
                loss_g_mse = criterion_mse(fake, gt)
                loss_g_total += weights['mse'] * loss_g_mse
                epoch_loss_g_mse += loss_g_mse.item()

            # 3. 感知颜色损失（根据配置启用）
            if 'perceptual_color' in ablation_config['enabled_losses']:
                loss_g_pcolor = perceptual_color_loss(fake, gt, perceptual_net)
                loss_g_total += weights['perceptual_color'] * loss_g_pcolor
                epoch_loss_g_perceptual_color += loss_g_pcolor.item()

            # 4. SSIM损失（根据配置启用）
            if 'ssim' in ablation_config['enabled_losses']:
                loss_g_ssim_val = ssim_loss(fake, gt)
                loss_g_total += weights['ssim'] * loss_g_ssim_val
                epoch_loss_g_ssim += loss_g_ssim_val.item()

            # 5. 风格损失（根据配置启用）
            if 'style' in ablation_config['enabled_losses']:
                loss_g_style_val = style_loss(fake, gt, perceptual_net)
                loss_g_total += weights['style'] * loss_g_style_val
                epoch_loss_g_style += loss_g_style_val.item()

            # 6. 感知损失（根据配置启用）
            if 'perceptual' in ablation_config['enabled_losses']:
                loss_g_perceptual_val = perceptual_loss(fake, gt, perceptual_net)
                loss_g_total += weights['perceptual'] * loss_g_perceptual_val
                epoch_loss_g_perceptual += loss_g_perceptual_val.item()

            # 反向传播
            loss_g_total.backward()
            optimizer_g.step()

            # 累计损失
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g_total.item()

            # 打印中间结果（现在变量都已定义）
            if i % 50 == 0:
                print(f"轮次 [{epoch + 1}/{epochs}], 步骤 [{i + 1}/{len(dataloader)}], "
                      f"判别器损失: {loss_d.item():.4f}, 生成器损失: {loss_g_total.item():.4f}, "
                      f"GAN: {loss_g_gan.item():.4f}, L1/MSE: {loss_g_l1.item():.4f}/{loss_g_mse.item():.4f}, "
                      f"颜色感知: {loss_g_pcolor.item():.4f}, SSIM: {loss_g_ssim_val.item():.4f}")

        # 计算平均损失
        avg_loss_d = epoch_loss_d / len(dataloader)
        avg_loss_g = epoch_loss_g / len(dataloader)
        avg_loss_g_gan = epoch_loss_g_gan / len(dataloader)
        avg_loss_g_l1 = epoch_loss_g_l1 / len(dataloader)
        avg_loss_g_mse = epoch_loss_g_mse / len(dataloader)
        avg_loss_g_perceptual_color = epoch_loss_g_perceptual_color / len(dataloader)
        avg_loss_g_ssim = epoch_loss_g_ssim / len(dataloader)
        avg_loss_g_style = epoch_loss_g_style / len(dataloader)
        avg_loss_g_perceptual = epoch_loss_g_perceptual / len(dataloader)

        # 保存损失历史
        loss_history['loss_d'].append(avg_loss_d)
        loss_history['loss_g'].append(avg_loss_g)
        loss_history['loss_g_gan'].append(avg_loss_g_gan)
        loss_history['loss_g_l1'].append(avg_loss_g_l1)
        loss_history['loss_g_mse'].append(avg_loss_g_mse)
        loss_history['loss_g_perceptual_color'].append(avg_loss_g_perceptual_color)
        loss_history['loss_g_ssim'].append(avg_loss_g_ssim)
        loss_history['loss_g_style'].append(avg_loss_g_style)
        loss_history['loss_g_perceptual'].append(avg_loss_g_perceptual)

        # 打印轮次总结
        print(f"\n轮次 [{epoch + 1}/{epochs}] 总结:")
        print(f"平均判别器损失: {avg_loss_d:.4f}")
        print(f"平均生成器损失: {avg_loss_g:.4f}")
        print(f"  GAN损失: {avg_loss_g_gan:.4f}")
        print(f"  L1损失: {avg_loss_g_l1:.4f}")
        print(f"  MSE损失: {avg_loss_g_mse:.4f}")
        print(f"  颜色感知损失: {avg_loss_g_perceptual_color:.4f}")
        print(f"  SSIM损失: {avg_loss_g_ssim:.4f}")
        print(f"  风格损失: {avg_loss_g_style:.4f}")
        print(f"  感知损失: {avg_loss_g_perceptual:.4f}\n")

        # 定期保存模型和生成图像
        if (epoch + 1) % 10 == 0:
            # 保存模型（不变）
            torch.save(generator.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f'discriminator_epoch_{epoch + 1}.pth'))
            print(f"模型已保存至 {model_dir}")

            # 生成并保存样本图像（不变）
            with torch.no_grad():
                sample_line = line[:3]
                sample_palette = palette[:3]
                sample_gt = gt[:3]
                sample_fake = generator(sample_line, sample_palette)

                torch.save(
                    {"line": sample_line, "palette": sample_palette},
                    f"train_input_epoch_{epoch + 1}.pth"
                )
                print(f"✅ 训练输入数据已保存：train_input_epoch_{epoch + 1}.pth")

                for j in range(min(3, batch_size)):
                    plt.figure(figsize=(15, 5))
                    plt.subplot(131)
                    plt.imshow(sample_line[j].cpu().squeeze(), cmap='gray')
                    plt.title('Line Drawing')
                    plt.axis('off')
                    plt.subplot(132)
                    plt.imshow((sample_fake[j].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
                    plt.title('Generated Image')
                    plt.axis('off')
                    plt.subplot(133)
                    plt.imshow((sample_gt[j].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
                    plt.title('Ground Truth')
                    plt.axis('off')

                    img_path = os.path.join(images_dir, f'generated_epoch_{epoch + 1}_sample_{j}.png')
                    plt.savefig(img_path, bbox_inches='tight')
                    plt.close()
                print(f"样本图像已保存至 {images_dir}")

        # 绘制并保存损失曲线
        plot_loss_curves(loss_history, logs_dir, epoch + 1)

    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(model_dir, 'generator_final_hed_contour_new.pth'))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, 'discriminator_final.pth'))
    print(f"训练完成。最终模型已保存至 {model_dir}")
    return generator, discriminator, exp_dir


# 损失曲线绘制（新增MSE损失曲线显示）
def plot_loss_curves(loss_history, save_dir, epoch):
    plt.figure(figsize=(16, 12))

    # 1. 总损失对比
    plt.subplot(3, 2, 1)
    plt.plot(loss_history['loss_d'], label='Discriminator Loss')
    plt.plot(loss_history['loss_g'], label='Generator Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. GAN与像素级损失（新增MSE）
    plt.subplot(3, 2, 2)
    plt.plot(loss_history['loss_g_gan'], label='Generator GAN Loss')
    plt.plot(loss_history['loss_g_l1'], label='Generator L1 Loss')
    plt.plot(loss_history['loss_g_mse'], label='Generator MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN and Pixel-wise Losses')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. 颜色感知与SSIM损失
    plt.subplot(3, 2, 3)
    plt.plot(loss_history['loss_g_perceptual_color'], label='Perceptual Color Loss')
    plt.plot(loss_history['loss_g_ssim'], label='SSIM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Color Perception Losses')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. 风格损失
    plt.subplot(3, 2, 4)
    plt.plot(loss_history['loss_g_style'], label='Style Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Style Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 5. 感知损失
    plt.subplot(3, 2, 5)
    plt.plot(loss_history['loss_g_perceptual'], label='Perceptual Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Perceptual Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    combined_path = os.path.join(save_dir, f'loss_curves_epoch_{epoch}.png')
    plt.savefig(combined_path)
    plt.close()

    # 保存单独损失曲线（新增MSE）
    plot_individual_loss(loss_history['loss_d'], 'Discriminator Loss', save_dir, epoch, 'orange')
    plot_individual_loss(loss_history['loss_g'], 'Generator Total Loss', save_dir, epoch, 'blue')
    plot_individual_loss(loss_history['loss_g_gan'], 'Generator GAN Loss', save_dir, epoch, 'red')
    plot_individual_loss(loss_history['loss_g_l1'], 'Generator L1 Loss', save_dir, epoch, 'cyan')
    plot_individual_loss(loss_history['loss_g_mse'], 'Generator MSE Loss', save_dir, epoch, 'magenta')
    plot_individual_loss(loss_history['loss_g_perceptual_color'], 'Perceptual Color Loss', save_dir, epoch, 'magenta')
    plot_individual_loss(loss_history['loss_g_ssim'], 'SSIM Loss', save_dir, epoch, 'brown')
    plot_individual_loss(loss_history['loss_g_style'], 'Style Loss', save_dir, epoch, 'purple')
    plot_individual_loss(loss_history['loss_g_perceptual'], 'Perceptual Loss', save_dir, epoch, 'green')


def plot_individual_loss(loss_data, title, save_dir, epoch, color):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_data, label=title, color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()


# 主函数入口（修改：添加实验组选择）
if __name__ == "__main__":
    # ---------------------- 关键：手动选择要训练的实验组序号 ----------------------
    SELECTED_ABLATION_ID = 0  # 在这里修改序号（0-6）选择实验组

    # 固定训练随机种子
    TRAINING_SEED = 2024
    # -----------------------------------------------------------------------------

    # 获取所有消融配置
    ablation_configs = get_ablation_configs()

    # 验证选择的序号是否有效
    if SELECTED_ABLATION_ID < 0 or SELECTED_ABLATION_ID >= len(ablation_configs):
        raise ValueError(f"无效的实验组序号！请选择 0-{len(ablation_configs) - 1} 之间的序号")

    # 获取选中的实验组配置
    selected_config = ablation_configs[SELECTED_ABLATION_ID]
    print(f"当前选择的实验组：")
    print(f"ID: {selected_config['id']}")
    print(f"名称: {selected_config['name']}")
    print(f"描述: {selected_config['desc']}")
    print(f"启用的损失项: {selected_config['enabled_losses']}\n")

    # 配置路径（请根据实际情况修改）
    dataset_path = "./dataset"
    vgg_weights_path = "./checkpoints/vgg16-397923af.pth"
    load_gen_weight_path = None  # 可选：预训练权重路径

    # 启动训练（传入选中的消融配置）
    generator, discriminator, exp_dir = train_model(
        dataset_path=dataset_path,
        ablation_config=selected_config,  # 传入消融配置
        exp_name="palettegan",
        epochs=100,
        batch_size=8,
        vgg_weights_path=vgg_weights_path,
        load_gen_weight_path=load_gen_weight_path,
        seed=TRAINING_SEED
    )
    print(f"实验完成。结果存储在: {exp_dir}")
