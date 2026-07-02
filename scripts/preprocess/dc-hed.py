import os

# 如果你电脑有多张 NVIDIA GPU，这里指定第 0 张
# 注意：这几行最好放在 import torch 之前
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import os.path as osp
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
import glob

from models import RCF


# =========================
# 直接在这里指定路径
# =========================

IMG_DIR = r"../../../pix2pix_code04/generated_results_dc_hed_contour_clean_"

SAVE_DIR = r"../../../pix2pix_code04/generated_results_dc_hed_contour_clean_/sketch"

CHECKPOINT = r"bsds500_pascal_model.pth"
# 这里改成你本地电脑上的模型权重路径，例如：
# CHECKPOINT = r"D:\project\RCF\checkpoints\RCFcheckpoint_epoch12.pth"


# =========================
# 自动选择 GPU / CPU
# =========================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. The code will run on CPU.")


def single_scale_test(model, test_loader, test_list, save_dir):
    model.eval()

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, image in enumerate(test_loader):
            image = image.to(DEVICE, non_blocking=True)

            _, _, H, W = image.shape
            results = model(image)

            all_res = torch.zeros((len(results), 1, H, W), device=DEVICE)

            for i in range(len(results)):
                all_res[i, 0, :, :] = results[i]

            filename = osp.splitext(test_list[idx])[0]

            torchvision.utils.save_image(
                1 - all_res,
                osp.join(save_dir, f"{filename}.jpg")
            )

            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)

            cv2.imwrite(
                osp.join(save_dir, f"{filename}_ss.png"),
                fuse_res
            )

    print("Running single-scale test done")


def multi_scale_test(model, test_loader, test_list, save_dir):
    model.eval()

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    scale = [0.5, 1.0, 1.5]

    with torch.no_grad():
        for idx, image in enumerate(test_loader):
            # image shape: [1, C, H, W]
            in_ = image[0].numpy().transpose((1, 2, 0))

            _, _, H, W = image.shape
            ms_fuse = np.zeros((H, W), np.float32)

            for s in scale:
                im_ = cv2.resize(
                    in_,
                    None,
                    fx=s,
                    fy=s,
                    interpolation=cv2.INTER_LINEAR
                )

                im_ = im_.transpose((2, 0, 1))
                im_tensor = torch.from_numpy(im_).float().unsqueeze(0).to(DEVICE)

                results = model(im_tensor)

                fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
                fuse_res = cv2.resize(
                    fuse_res,
                    (W, H),
                    interpolation=cv2.INTER_LINEAR
                )

                ms_fuse += fuse_res

            ms_fuse = ms_fuse / len(scale)

            filename = osp.splitext(test_list[idx])[0]
            ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)

            save_path = osp.join(save_dir, f"{filename}.jpg")
            cv2.imwrite(save_path, ms_fuse)

            print(f"Saved: {save_path}")

    print("Running multi-scale test done")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mean):
        self.img_paths = img_paths
        self.mean = mean

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.img_paths[idx]}")

        img = np.array(img, dtype=np.float32)

        # cv2.imread 读取的是 BGR，和原始 RCF/HED 预处理一致
        img = img - self.mean
        img = img.transpose((2, 0, 1))

        return img


if __name__ == "__main__":

    if not osp.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 读取 JPG / JPEG / PNG，兼容大小写
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        img_paths.extend(glob.glob(osp.join(IMG_DIR, ext)))

    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in: {IMG_DIR}")

    test_list = [osp.basename(path) for path in img_paths]

    mean = np.array(
        [104.00698793, 116.66876762, 122.67891434],
        dtype=np.float32
    )

    test_dataset = CustomDataset(img_paths, mean)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    model = RCF().to(DEVICE)

    if CHECKPOINT is not None and osp.isfile(CHECKPOINT):
        print(f"=> loading checkpoint from: {CHECKPOINT}")

        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

        # 兼容两种保存方式：
        # 1. torch.save(model.state_dict(), path)
        # 2. torch.save({"state_dict": model.state_dict()}, path)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint)
        print("=> checkpoint loaded")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print("Performing the testing...")

    # 单尺度测试
    # single_scale_test(model, test_loader, test_list, SAVE_DIR)

    # 多尺度测试
    multi_scale_test(model, test_loader, test_list, SAVE_DIR)