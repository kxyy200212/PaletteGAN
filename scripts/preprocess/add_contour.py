import cv2
import numpy as np
import os


def process_single_image(hed_path, mask_path):
    """处理单张图片（外轮廓最细，保留内部细节）"""
    # 读取输入
    hed = cv2.imread(hed_path, 0)
    mask = cv2.imread(mask_path, 0)
    if hed is None or mask is None:
        return None  # 读取失败返回None

    # 统一形状
    hed = np.squeeze(hed).astype(np.uint8)
    mask = np.squeeze(mask).astype(np.uint8)
    if hed.shape != mask.shape:
        mask = cv2.resize(mask, (hed.shape[1], hed.shape[0]), cv2.INTER_NEAREST)

    # 划分内部区域（窄外轮廓带）
    kernel = np.ones((1, 1), np.uint8)
    inner_mask = cv2.erode(mask, kernel, iterations=1)
    inner_result = cv2.bitwise_and(hed, inner_mask)

    # 外轮廓区域
    outer_mask = cv2.subtract(mask, inner_mask)

    # 生成精细外轮廓（1像素抗锯齿）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    clean_contour = np.full_like(hed, 255)
    for cnt in contours:
        cv2.polylines(clean_contour, [cnt], isClosed=True, color=0, thickness=1, lineType=cv2.LINE_AA)

    # 外轮廓结果
    outer_result = cv2.bitwise_and(clean_contour, outer_mask)

    # 非服装区域补白
    non_cloth_mask = 255 - mask
    non_cloth_white = np.full_like(hed, 255)
    non_cloth_result = cv2.bitwise_and(non_cloth_white, non_cloth_mask)

    # 融合
    final_result = cv2.bitwise_or(inner_result, outer_result)
    final_result = cv2.bitwise_or(final_result, non_cloth_result)

    return final_result


def batch_process(hed_dir, mask_dir, output_dir):
    """批量处理所有图片（无进度条）"""
    # 创建输出文件夹（若不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取HED文件夹中所有图片文件名
    hed_extensions = ('.jpg', '.jpeg', '.png')
    hed_filenames = [f for f in os.listdir(hed_dir) if f.lower().endswith(hed_extensions)]
    total = len(hed_filenames)
    print(f"共发现{total}个HED文件，开始批量处理...")

    # 遍历所有HED图片
    processed = 0
    for hed_filename in hed_filenames:
        base_name = os.path.splitext(hed_filename)[0]  # 提取文件名（不含扩展名）

        # 构建文件路径
        hed_path = os.path.join(hed_dir, hed_filename)
        mask_filename = f"{base_name}.png"  # 掩码文件名（与HED同名，.png格式）
        mask_path = os.path.join(mask_dir, mask_filename)

        # 检查掩码是否存在
        if not os.path.exists(mask_path):
            print(f"跳过：掩码不存在 -> {mask_path}")
            continue

        # 处理单张图片
        result = process_single_image(hed_path, mask_path)
        if result is None:
            print(f"跳过：处理失败 -> {hed_path}")
            continue

        # 保存结果
        output_filename = f"{base_name}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, result)
        processed += 1

    print(f"批量处理完成！成功处理{processed}/{total}个文件，结果保存至：{output_dir}")


# 主程序
if __name__ == "__main__":
    # 路径配置
    HED_DIR = "../Comicolorization/output/sketch"
    MASK_DIR = "../dataset/mask_dc_hed_contour"
    OUTPUT_DIR = "../Comicolorization/output/sketch"

    # 执行批量处理
    batch_process(hed_dir=HED_DIR, mask_dir=MASK_DIR, output_dir=OUTPUT_DIR)