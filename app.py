import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from skimage.color import rgb2lab as ski_rgb2lab, lab2rgb as ski_lab2rgb
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial
import os
from PIL import Image
import streamlit as st
from glob import glob

# ======================= 页面配置 =======================
st.set_page_config(
    page_title="AI面料重着色系统",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎨 AI 面料智能重着色工具")
st.markdown("---")


# ======================= 模型定义 =======================
class Convertor:
    @staticmethod
    def rgb2lab(rgb):
        return ski_rgb2lab(rgb)

    @staticmethod
    def lab2rgb(lab):
        if isinstance(lab, torch.Tensor):
            lab = lab.detach().cpu().numpy()
        return ski_lab2rgb(lab)


convertor = Convertor()


class PalletNetDataset(Dataset):
    def __init__(self, root, out5=False):
        with open(root + 'Arg.json', 'r') as outfile:
            self.Arg = json.load(outfile)
        with open(root + 'Org.json', 'r') as outfile:
            self.Org = json.load(outfile)
        self.out5 = out5
        self.ToArr = False
        self.ToTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.Arg)

    def Changeto5(self):
        self.out5 = False if self.out5 else True

    def ChangetoArray(self):
        self.ToArr = False if self.ToArr else True

    def Normalise_only(self, arr):
        return (arr - [50, 0, 0]) / [50, 128, 128]

    def Lab_Normalise_and_tensor(self, arr):
        return self.ToTensor(self.Normalise_only(arr)).float()

    def __getitem__(self, idx):
        im = io.imread(self.Arg[idx]["name"])
        im_lab = convertor.rgb2lab(im)
        im_tar_idx = self.Arg[idx]["siblings"][random.randint(0, 19)]
        im_tar = io.imread(self.Arg[im_tar_idx]["name"])
        im_tar_lab = convertor.rgb2lab(im_tar)
        pl = np.array([self.Arg[im_tar_idx]["palette"]]) / 255
        palette = convertor.rgb2lab(pl)

        OG = 0
        OGPal = 0
        if self.out5:
            OG_idx = random.randint(0, len(self.Org) - 1)
            im_OG = io.imread(self.Org[OG_idx]["name"])
            im_OG_lab = convertor.rgb2lab(im_OG)
            OGpl = np.array([self.Org[OG_idx]["palette"]]) / 255
            OGpalette = convertor.rgb2lab(OGpl)
            OG = im_OG_lab
            OGPal = OGpalette

        if self.ToArr:
            im_lab = self.Lab_Normalise_and_tensor(im_lab)
            palette = torch.tensor(self.Normalise_only(palette).flatten()).float()
            im_tar_lab = self.Lab_Normalise_and_tensor(im_tar_lab)[1:, :, :]
            if self.out5:
                OG = self.Lab_Normalise_and_tensor(OG)
                OGPal = torch.tensor(self.Normalise_only(OGPal)).float()
                if OG.shape[1] == 256:
                    OG = OG.permute(0, 2, 1)
            if im.shape[0] == 256:
                im_lab = im_lab.permute(0, 2, 1)
                im_tar_lab = im_tar_lab.permute(0, 2, 1)

        res = {
            "source": im_lab, "TPal": palette, "tar": im_tar_lab, "OG": OG, "OGPal": OGPal
        }
        return res


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.activate = activation_func(activation)

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=2, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False, padding=0),
                'bn': nn.InstanceNorm2d(self.expanded_channels)
            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.InstanceNorm2d(out_channels)}))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(negative_slope=0.02),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class FeatureEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeatureEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.res1 = ResNetBasicBlock(64, 128)
        self.res2 = ResNetBasicBlock(128, 256)
        self.res3 = ResNetBasicBlock(256, 512)

    def forward(self, x):
        x = F.relu(self.norm(self.conv(x)))
        c4 = self.pool(x)
        c3 = self.res1(c4)
        c2 = self.res2(c3)
        c1 = self.res3(c2)
        return c1, c2, c3, c4


def de_conv(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.02, inplace=True)
    )


class RecoloringDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_up_4 = de_conv(18 + 512, 256)
        self.dconv_up_3 = de_conv(256 + 256, 128)
        self.dconv_up_2 = de_conv(18 + 128 + 128, 64)
        self.dconv_up_1 = de_conv(18 + 64 + 64, 64)
        self.conv_last = nn.Conv2d(1 + 64, 2, kernel_size=3, padding=1)

    def forward(self, c1, c2, c3, c4, target_palettes_1d, illu):
        bz, h, w = c1.shape[0], c1.shape[2], c1.shape[3]
        tp_reshpaed = target_palettes_1d.reshape(bz, 18, 1, 1)
        tp_c1 = tp_reshpaed.repeat(1, 1, h, w)
        x = torch.cat((c1, tp_c1), 1)
        x = self.dconv_up_4(x)
        x = torch.cat([c2, x], dim=1)
        x = self.dconv_up_3(x)
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        tp_c3 = tp_reshpaed.repeat(1, 1, h, w)
        x = torch.cat([tp_c3, c3, x], dim=1)
        x = self.dconv_up_2(x)
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        tp_c4 = tp_reshpaed.repeat(1, 1, h, w)
        x = torch.cat([tp_c4, c4, x], dim=1)
        x = self.dconv_up_1(x)
        illu = illu.view(illu.size(0), 1, illu.size(2), illu.size(3))
        x = torch.cat((x, illu), dim=1)
        x = self.conv_last(x)
        x = torch.tanh(x)
        return x


# ======================= 加载模型 =======================
@st.cache_resource
def load_model():
    FE = FeatureEncoder()
    RD = RecoloringDecoder()
    FE.load_state_dict(torch.load("Models/FE.state_dict.pt", map_location="cpu"))
    RD.load_state_dict(torch.load("Models/RD.state_dict.pt", map_location="cpu"))
    FE.eval()
    RD.eval()
    return FE, RD


FE, RD = load_model()
device = "cpu"
FE.to(device)
RD.to(device)


# ======================= 工具函数 =======================
def get_all_palettes(npy_path="palette_list_maxminc.npy"):
    return np.load(npy_path)


def visualize_palette(palette_np):
    palette_np = palette_np.reshape(6, 3)
    fig, ax = plt.subplots(1, 6, figsize=(10, 1))
    for i, color in enumerate(palette_np):
        ax[i].set_facecolor(color / 255)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    return fig


@st.cache_data
def get_all_fabrics(folder_path="./mianliao/"):
    os.makedirs(folder_path, exist_ok=True)
    fabric_paths = sorted(glob(os.path.join(folder_path, "*.[jp][pn]g")) + glob(os.path.join(folder_path, "*.png")))
    fabric_names = [os.path.basename(p) for p in fabric_paths]
    return fabric_paths, fabric_names


def run_colorize(img_pil, palette_idx, palettes_np):
    img = np.array(img_pil.convert("RGB"))
    z = ((convertor.rgb2lab(img)) - [50, 0, 0]) / [50, 127, 127]
    img_tensor = torch.Tensor(z).permute(2, 0, 1)

    h = 16 * int(img_tensor.shape[1] / 16)
    w = 16 * int(img_tensor.shape[2] / 16)
    img_tensor = transforms.Resize((h, w))(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    target_palette = palettes_np[palette_idx]
    pal_np = np.array(target_palette).reshape(1, 6, 3) / 255
    pal = torch.Tensor((convertor.rgb2lab(pal_np) - [50, 0, 0]) / [50, 128, 128]).unsqueeze(0).to(device)

    illu = img_tensor[:, 0:1, :, :]

    with torch.no_grad():
        c1, c2, c3, c4 = FE(img_tensor)
        out = RD(c1, c2, c3, c4, pal, illu)
        final = torch.cat([(illu + 1) * 50, out * 128], axis=1).permute(0, 2, 3, 1)[0]

    return convertor.lab2rgb(final)


# ======================= 加载数据 =======================
palettes = get_all_palettes()
fabric_paths, fabric_names = get_all_fabrics()

# ======================= 侧边栏 =======================
with st.sidebar:
    st.header("📁 操作区")
    st.subheader("🧵 选择面料")
    selected_fabric_idx = st.selectbox(
        "本地面料库",
        options=range(len(fabric_names)),
        format_func=lambda x: fabric_names[x],
        index=0
    )

    st.markdown("---")
    st.subheader("🎨 选择调色板")
    palette_idx = st.selectbox(
        f"共 {len(palettes)} 个调色板",
        range(len(palettes)),
        index=0
    )

    st.markdown("---")
    run_btn = st.button("▶️ 开始AI重着色", type="primary")

# ======================= 主界面 =======================
selected_fabric_path = fabric_paths[selected_fabric_idx]
selected_fabric_name = fabric_names[selected_fabric_idx]

col1, col2, col3 = st.columns([1, 1, 1])
st.subheader(f"✅ 当前选择：调色板 #{palette_idx}")
st.pyplot(visualize_palette(palettes[palette_idx]))
st.markdown("---")

# 会话状态，缓存结果
if "result_img" not in st.session_state:
    st.session_state.result_img = None

# 运行着色
if run_btn:
    fabric_img = Image.open(selected_fabric_path).convert("RGB")
    with st.spinner("AI 正在着色..."):
        result = run_colorize(fabric_img, palette_idx, palettes)
        st.session_state.result_img = result  # 存入缓存

    with col1:
        st.markdown("### 🧵 原图")
        st.image(fabric_img)
    with col2:
        st.markdown("### 🎨 调色板")
        st.pyplot(visualize_palette(palettes[palette_idx]))
    with col3:
        st.markdown("### ✨ 结果")
        st.image(result)

# 显示结果 + 手动保存按钮
elif st.session_state.result_img is not None:
    with col1:
        st.markdown("### 🧵 原图")
        st.image(Image.open(selected_fabric_path))
    with col2:
        st.markdown("### 🎨 调色板")
        st.pyplot(visualize_palette(palettes[palette_idx]))
    with col3:
        st.markdown("### ✨ 结果")
        st.image(st.session_state.result_img)

    # ✅ 手动保存按钮（只有点击才保存！）
    if st.button("💾 保存结果到本地"):
        os.makedirs("./result", exist_ok=True)
        base = os.path.splitext(selected_fabric_name)[0]
        save_path = f"./result/{base}_调色板{palette_idx}.png"
        result_pil = Image.fromarray((st.session_state.result_img * 255).astype(np.uint8))
        result_pil.save(save_path)
        st.success(f"✅ 保存成功！路径：{save_path}")

else:
    st.info("👈 选择面料+调色板 → 点击【开始AI重着色】")
    st.image(Image.open(selected_fabric_path), caption=f"当前：{selected_fabric_name}", width=400)

# ======================= 全量预览 =======================
st.markdown("---")
st.subheader("🎨 全部调色板")
p_cols = st.columns(4)
for i, p in enumerate(palettes):
    with p_cols[i % 4]:
        st.caption(f"#{i}")
        st.pyplot(visualize_palette(p))

st.markdown("---")
st.subheader("🧵 全部面料")
f_cols = st.columns(3)
for i, path in enumerate(fabric_paths):
    with f_cols[i % 3]:
        st.caption(fabric_names[i])
        st.image(Image.open(path))