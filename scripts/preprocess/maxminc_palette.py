#修改聚类数的our方法
# our method 色差评价模型
import random
import matplotlib.pyplot as plt
from skfuzzy import maxmin_composition
# from sklearn.cluster import KMeans  # 已改为PyTorch GPU版KMeans，可不再使用sklearn
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import datetime
import numpy as np
from colorsys import rgb_to_hsv  # 用于RGB转HSV，提取饱和度
import cv2
import matplotlib.pyplot as plt
import os
import torch
start = datetime.datetime.now()
#from sklearn.cluster import KMeans,BisectingKMeans
from skimage.color import rgb2lab,lab2rgb

# CIEDE2000: 精确计算两个CIE Lab*颜色的色差。
# calculate_MICD: 计算两组颜色的最小个体色差。
# calculate_MECD: 计算两组颜色的平均色差。
# Hungarian_hit: 使用匈牙利算法匹配两个调色板，并计算总色差和命中率。
# find_and_prioritize_color: 按目标颜色优先级重新排列颜色。
# max_De_Colors: 找到最大平均色差的颜色组合。
# reorder_palette: 将目标调色板按参考调色板重新排列。

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab,lab2rgb
import math
from datetime import datetime
from colormath.color_objects import LabColor

# target_colors = [
#     [255, 0, 0],    # 红色
#     [255, 165, 0],  # 橙色
#     [255, 255, 0],  # 黄色  #[0, 128, 0],    # 绿色
#     [0, 255, 255],  # 青色
#     [0, 0, 255],    # 蓝色
#     [128, 0, 128]   # 紫色
# ]
# 计算最小个体色差（Minimum Individual Color Difference，MICD）
def calculate_MICD(palete_k_labs,x_labs):
    m1=[]
    m2=[]
    m=[]
    index = [-1,-1,-1,-1,-1,-1]
    for i in range(6):
        ep_min=100
        for j in range(6):
            ep = CIEDE2000(palete_k_labs[i],x_labs[j])
            #print("m1:",i,j,ep,ep_min)
            if ep<ep_min:
                ep_min = ep
                index[i] = j
        m1.append(ep_min)

    for i in range(6):
            ep_min=100
            for j in range(6):
                ep = CIEDE2000(x_labs[i],palete_k_labs[j])
                #print("m2:",i,j,ep,ep_min)
                if ep<ep_min:
                    ep_min = ep

            m2.append(ep_min)
    minmal_ep = sum(m1+m2)/len(m1+m2)
    return minmal_ep

# 计算平均色差（Mean Color Difference，MECD）
def calculate_MECD(palete_k_labs,x_labs):

    m=[]

    for i in range(6):

        for j in range(6):
            ep = CIEDE2000(palete_k_labs[i],x_labs[j])
            #print("m1:",i,j,ep,ep_min)
            m.append(ep)

    avg_ep = sum(m)/len(m)
    return avg_ep

target = ['红','橙','黄','青','蓝','紫']

# 使用匈牙利算法计算两个调色板之间的色差，并返回总色差、命中率及匹配后的调色板颜色
def Hungarian_hit(palette1_lab, palette2_lab):



    # Calculate the cost matrix (color differences)
    cost_matrix = np.zeros((len(palette1_lab), len(palette2_lab)))
    for i in range(len(palette1_lab)):
        for j in range(len(palette2_lab)):
            cost_matrix[i, j] = CIEDE2000(palette1_lab[i], palette2_lab[j])

    # Apply the Hungarian algorithm to find the minimum cost matching
    #print(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(row_ind,col_ind)
    matched_palette2_lab = [palette2_lab[j] for j in col_ind]

    cost = cost_matrix[row_ind, col_ind].sum()/6
    score = 0
    for j in range(6):
        if CIEDE2000(palette1_lab[j],matched_palette2_lab[j]) < 20:
            score += 1
            print(j,' goal')

    return cost,score,matched_palette2_lab

# 寻找与目标颜色接近的颜色并结束程序# 寻找与目标颜色最接近的颜色并重新排列颜色列表，返回重新排列后的颜色列表
# def find_and_prioritize_color(colors, target_colors=target_colors):
#     min_de = 100
#     #print(colors)
#     res = colors
#     for i,target_color in enumerate(target_colors):
#         target_color = np.array(target_color).astype('uint8')
#         target_color = rgb2lab(target_color)
#         for index , color in enumerate(colors):
#
#             color = rgb2lab(color)
#
#             de = CIEDE2000(color,target_color)
#             #print(target[i],target_color,color,de,min_de,index,colors.shape)
#             if  de < min_de:
#                 min_de = de
#
#                 colors[[0,index]] =  colors[[index,0]]
#                 res = colors
#                 #print(colors)
#                 ##return colors
#     #print(res)
#     return res

def maxsumc_selection(rgb_15colors, sample_num=6):
    """
    从15个RGB颜色中选择6个代表性颜色（MAXSUMC逻辑）
    直接处理uint8类型的RGB数据（0-255），无需转换LAB

    参数:
        rgb_15colors: np.ndarray, 15×3的RGB矩阵（uint8，每行一个颜色）
        sample_num: int, 待选择的颜色数（默认6）

    返回:
        selected_6colors: np.ndarray, 6×3的RGB矩阵（uint8，选中的6个颜色）
    """
    # 转置为3×15矩阵（按列处理每个颜色的RGB通道）
    rgb_transposed = rgb_15colors.T  # 形状：(3, 15)，uint8类型

    # 1. 选择第一个样本：方差最大的颜色（饱和度最高）
    # 计算每个颜色的RGB通道方差（先转float避免溢出，不影响相对大小）
    var_rgb = np.var(rgb_transposed.astype(np.float32), axis=0)
    selected_indices = [np.argmax(var_rgb)]  # 方差最大的颜色索引

    # 2. 选择第二个样本：与第一个样本欧氏距离最大
    distances = np.zeros(15)  # 存储每个颜色到第一个样本的距离
    for i in range(15):
        # 计算欧氏距离（转float计算，结果不影响排序）
        distances[i] = np.linalg.norm(
            rgb_transposed[:, i].astype(np.float32) - rgb_transposed[:, selected_indices[0]].astype(np.float32)
        )
    distances[selected_indices] = 0  # 排除已选中样本
    selected_indices.append(np.argmax(distances))  # 距离最大的颜色索引

    # 3. 选择第3-6个样本：与已选所有样本的距离和最大
    for _ in range(2, sample_num):
        dist_sum = np.zeros(15)  # 存储每个颜色到已选样本的距离和
        for i in range(15):
            if i not in selected_indices:  # 只计算未选中颜色
                sum_d = 0.0
                for s_idx in selected_indices:
                    # 累加欧氏距离（转float计算）
                    sum_d += np.linalg.norm(
                        rgb_transposed[:, i].astype(np.float32) - rgb_transposed[:, s_idx].astype(np.float32)
                    )
                dist_sum[i] = sum_d
        dist_sum[selected_indices] = 0  # 排除已选中样本
        selected_indices.append(np.argmax(dist_sum))  # 距离和最大的颜色索引

    # 提取选中的6个颜色（保持uint8类型）
    selected_6colors = rgb_15colors[selected_indices]
    return selected_6colors

def maxminc_selection(rgb_15colors, sample_num=6):
    """
    实现MAXMINC选色逻辑（还原MATLAB原代码）
    1. 第1个：方差最大（饱和度最高）
    2. 第2个：与第1个欧氏距离最大
    3. 第3-6个：对每个未选颜色，计算它与所有已选样本的距离→取最小值（最近距离），选择该最小值最大的颜色
    （即“最远最近点”，保证样本均匀分布）

    参数:
        rgb_15colors: np.ndarray, 15×3的RGB矩阵（uint8，每行一个颜色）
        sample_num: int, 待选择的颜色数（默认6）

    返回:
        selected_6colors: np.ndarray, 6×3的RGB矩阵（uint8，选中的6个颜色）
    """
    # 转置为3×15矩阵（与MATLAB RGB=RGB' 一致，按列处理每个颜色）
    rgb_transposed = rgb_15colors.T  # 形状：(3, 15)，uint8类型
    m = rgb_15colors.shape[0]  # 总颜色数（15）

    # 初始化选中样本的索引列表
    selected_indices = []

    # 1. 第1个样本：方差最大的颜色（与MATLAB逻辑一致）
    var_rgb = np.var(rgb_transposed.astype(np.float32), axis=0)  # 每个颜色的RGB方差
    ind1 = np.argmax(var_rgb)
    selected_indices.append(ind1)

    # 2. 第2个样本：与第1个样本欧氏距离最大（与MATLAB逻辑一致）
    coor = np.zeros(m)
    for i in range(m):
        # 计算欧氏距离（转float32避免溢出）
        coor[i] = np.linalg.norm(
            rgb_transposed[:, i].astype(np.float32) - rgb_transposed[:, ind1].astype(np.float32)
        )
    coor[selected_indices] = 0  # 排除已选中样本
    ind2 = np.argmax(coor)
    selected_indices.append(ind2)

    # 3. 第3-6个样本：MAXMINC核心逻辑（最近距离最大）
    for j in range(2, sample_num):  # j是当前要选的第j+1个样本（索引从2开始，对应第3个样本）
        # dist矩阵：(已选样本数, 总颜色数)，存储每个已选样本到每个颜色的距离
        dist = np.zeros((j, m), dtype=np.float32)  # j=2时（选第3个样本），已选2个样本，dist为(2,15)

        for i in range(m):  # 遍历所有颜色
            for q in range(j):  # 遍历所有已选样本（共j个）
                # 计算已选样本q到颜色i的欧氏距离
                dist[q, i] = np.linalg.norm(
                    rgb_transposed[:, selected_indices[q]].astype(np.float32) - rgb_transposed[:, i].astype(np.float32)
                )

        # 排除已选中的样本（将它们的距离设为0，后续min时不会被选中）
        dist[:, selected_indices] = 0

        # 对每个颜色，取与已选样本的最小距离（min_dist(i) = 颜色i到已选集合的最近距离）
        min_dist = np.min(dist, axis=0)

        # 选择最小距离最大的颜色（最远最近点）
        ind_next = np.argmax(min_dist)
        selected_indices.append(ind_next)

    # 提取选中的6个颜色（保持uint8类型，与输入格式一致）
    selected_6colors = rgb_15colors[selected_indices]
    return selected_6colors

def find_and_prioritize_color(colors):
    """
    按颜色饱和度（Saturation）从高到低排序输入颜色列表
    :param colors: 待排序的颜色列表/数组，元素为RGB格式（0-255 uint8），形状为(n, 3)
    :return: 按饱和度降序排列后的颜色列表/数组（保持原数据类型）
    """
    # 确保输入是numpy数组，方便处理（兼容列表输入）
    colors = np.asarray(colors).astype('uint8')
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("输入颜色格式错误，需为形状为(n, 3)的RGB颜色列表/数组")

    # 存储每个颜色的「饱和度」和「原始索引」（用于后续排序）
    saturation_list = []
    for idx, rgb in enumerate(colors):
        # 1. RGB颜色转为0-1浮点数（colorsys要求输入范围）
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        # 2. RGB转HSV（hue:0-1, saturation:0-1, value:0-1）
        h, s, v = rgb_to_hsv(r, g, b)
        # 3. 记录饱和度和原始索引（避免排序后丢失对应关系）
        saturation_list.append((-s, idx))  # 用负号实现「降序排序」（默认升序）

    # 按饱和度降序排序（排序键是-s，值越小表示s越大，排在前面）
    saturation_list.sort()

    # 根据排序结果重新组织颜色列表
    sorted_colors = [colors[idx] for (neg_s, idx) in saturation_list]

    # 转回原数据格式（数组/列表）并返回
    return np.array(sorted_colors) if isinstance(colors, np.ndarray) else sorted_colors

# 将RGB颜色转换为色相值（Hue)
def rgb_to_hue(rgb):
    bgr = np.uint8([[rgb]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    hue = hsv[0][0][0]  # 取出色相值
    return hue


# 找到一组颜色中使得最大平均色差最大的颜色组合及对应的最大平均色差
def max_De_Colors(colors):
    first_color = colors[0]
    other_colors = colors[1:]
    color_combinations = combinations(other_colors, 5)
    #print(len(color_combinations))

# 初始化最大平均色差和对应的颜色组合
    max_avg_delta_e = 0
    max_avg_delta_e_combination = None

# 遍历所有颜色组合并计算平均色差
    for index,combination in enumerate(color_combinations):

        combination = list(combination)
        combination.append(first_color)
        #print(combination,len(combination))
        avg_delta_e = 0
        for color1, color2 in combinations(combination, 2):
            if CIEDE2000(color1, color2) < 20:
                break
            avg_delta_e +=    CIEDE2000(color1, color2)


    # 更新最大平均色差和对应的颜色组合
        if avg_delta_e > max_avg_delta_e:
            max_avg_delta_e = avg_delta_e
            max_avg_delta_e_combination = combination
    return max_avg_delta_e_combination , max_avg_delta_e

# 计算两个CIE L*a*b*颜色之间的CIEDE2000色差
def CIEDE2000(Lab_1, Lab_2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    C_25_7 = 6103515625 # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)

    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi
    else: h_ave = h1_ + h2_

    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))

    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00

import math
def reorder_palette(reference_palette, target_palette):
    index = [-1,-1,-1,-1,-1,-1]
    for j in range(6):
            ep_min=100
            for k in range(6):
                ep = CIEDE2000(reference_palette[j],target_palette[k])
                #print("m1:",i,j,ep,ep_min)
                if ep<ep_min and k not in index:
                    ep_min = ep
                    #ep_ori[j] =  ep_min
                    index[j] = k
    print(index)
    sorted_palette = target_palette
    sorted_palette[:6] = [sorted_palette[i] for i in index]
    score = 0
    for n in range(6):
        if CIEDE2000(reference_palette[n],target_palette[n]) < 18:
            score += 1
            print(n,' goal')

    return sorted_palette,score



#需要计算的代码块
def checkClrs(clrs):
    height = 50
    width = 50
    init = 0
    for color in clrs:
        for i in range(width):
            if init == 0:
                data = np.array([color])
                init = 1
            else:
                data = np.concatenate((data,np.array([color])), axis = 0)

    data = data.reshape(1,len(data),3)
    data2 = data
    for i in range(height):
        data = np.concatenate((data,data2), axis = 0)

    return(data)

#
# image_files = [f for f in os.listdir("./img1-60") if f.endswith(('.jpg', '.jpeg', '.png'))]
# palette_list = []
# index = 0
# # Iterate through each image file
# for index,image_file in enumerate(image_files):
#     # Read the image
#     original_image = cv2.imread(os.path.join("./img1-60", image_file))
#     print(image_file)
#     #img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#     #original_image = cv2.imread("../N.jpg")
#     np.random.seed(0)
#     random.seed(0)
#
#     img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
#     vectorized = img.reshape((-1,3))
#     #vectorized = rgb2lab(vectorized)
#     vectorized = np.float32(vectorized)
#     np.random.seed(0)
#
#     # 进行K均值聚类
#     kmeans = KMeans(n_clusters=15, random_state=0)  # 设置随机数种子为0
#     kmeans.fit(vectorized)
#     labels = kmeans.predict(vectorized)
#     centers = kmeans.cluster_centers_
#     center_rgb = np.uint8(centers)
#     center_copy = center_rgb
#     center_sorted = find_and_prioritize_color(center_copy)
#     hues = [rgb_to_hue(rgb) for rgb in center_copy]
#
# # # 按照色相值对RGB值进行排序
#     sorted_indices = np.argsort(hues)
#     sorted_rgb_matrix = center_copy[sorted_indices]
#     center_sorted = find_and_prioritize_color(sorted_rgb_matrix)
#     #print(center_rgb,center_sorted)
#     check = checkClrs(center_rgb)
#     check_sorted = checkClrs(center_sorted)
#
#
#     center_sorted_lab = rgb2lab(center_sorted)
#     center_6 , max_De = max_De_Colors(center_sorted_lab)
#     #print(center_6,type(center_6),max_De)
#     #处理k=6和k=7的异常
#     # 新增代码开始
#     # 获取当前聚类数（从KMeans参数中提取）
#     center_6 = np.array(center_6)
#     k = 15
#     if k in [6, 7]:
#         # 确保数组是2维结构 (n, 3)
#         if center_6.ndim != 2:
#             if center_6.size % 3 == 0:
#                 center_6 = center_6.reshape(-1, 3)
#             else:
#                 center_6 = center_sorted_lab[:6, :3]
#
#         # 确保通道数为3
#         if center_6.shape[-1] != 3:
#             center_6 = center_6[:, :3] if center_6.shape[-1] > 3 else \
#                       np.pad(center_6, ((0,0), (0, 3-center_6.shape[-1])), mode='constant')
#
#         # 确保有6个颜色中心
#         if len(center_6) < 6:
#             supplement_count = 6 - len(center_6)
#             center_6 = np.vstack([center_6, center_sorted_lab[:supplement_count, :3]])
#
#         # 过滤LAB空间异常值
#         center_6[:, 0] = np.clip(center_6[:, 0], 0, 100)
#         center_6[:, 1:] = np.clip(center_6[:, 1:], -128, 127)
#     # 新增代码结束
#
#     center_6_ori = (lab2rgb(center_6) *255).astype('uint8')
#
#
#     palette_list.append(center_6_ori)
#     fig = checkClrs(center_6_ori)
#
#     plt.subplot(1,1,1)
#     plt.axis('off')
#     plt.imshow(fig)
#     plt.show()
#
# palette_list = np.array(palette_list)
# np.save('./img1-60/palette_data_clusters_6-25/palette_list_saturation.npy',palette_list)


# ---------------------- GPU版KMeans：使用PyTorch在CUDA上聚类 ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前KMeans计算设备: {DEVICE}")

def _init_kmeans_plus_plus_torch(x, n_clusters=15, seed=0):
    """
    使用PyTorch实现KMeans++初始化，尽量接近sklearn KMeans的初始化逻辑。
    x: torch.Tensor, shape=(N, 3), float32, device=cuda/cpu
    """
    n_samples = x.shape[0]
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    centers = torch.empty((n_clusters, x.shape[1]), dtype=x.dtype, device=x.device)

    # 第一个中心随机选择
    first_idx = torch.randint(0, n_samples, (1,), generator=generator, device=x.device)
    centers[0] = x[first_idx]

    # 后续中心按距离平方加权抽样
    closest_dist_sq = torch.cdist(x, centers[0:1]).squeeze(1).pow(2)

    for c in range(1, n_clusters):
        total_dist = closest_dist_sq.sum()
        if total_dist.item() == 0:
            next_idx = torch.randint(0, n_samples, (1,), generator=generator, device=x.device)
        else:
            probs = closest_dist_sq / total_dist
            next_idx = torch.multinomial(probs, 1, generator=generator)
        centers[c] = x[next_idx]
        new_dist_sq = torch.cdist(x, centers[c:c+1]).squeeze(1).pow(2)
        closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

    return centers

def kmeans_torch_rgb(vectorized_np, n_clusters=15, max_iter=60, tol=1e-3, seed=0, device=DEVICE):
    """
    GPU版KMeans，用于替代 sklearn.cluster.KMeans。
    输入:
        vectorized_np: numpy数组，shape=(像素数, 3)，RGB，float32或uint8
    输出:
        centers_np: numpy数组，shape=(n_clusters, 3)，RGB颜色中心，float32，已回到CPU
    """
    if vectorized_np.ndim != 2 or vectorized_np.shape[1] != 3:
        raise ValueError("vectorized_np 必须是形状为 (N, 3) 的RGB像素矩阵")

    x = torch.as_tensor(vectorized_np, dtype=torch.float32, device=device)

    # 防止图像颜色数量少于聚类数时报错
    n_clusters = min(n_clusters, x.shape[0])

    centers = _init_kmeans_plus_plus_torch(x, n_clusters=n_clusters, seed=seed)

    for _ in range(max_iter):
        # 计算每个像素到各中心的距离，并分配标签
        distances = torch.cdist(x, centers)
        labels = torch.argmin(distances, dim=1)

        new_centers = []
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any():
                new_centers.append(x[mask].mean(dim=0))
            else:
                # 空簇时随机补一个中心，避免程序中断
                rand_idx = torch.randint(0, x.shape[0], (1,), device=device)
                new_centers.append(x[rand_idx].squeeze(0))

        new_centers = torch.stack(new_centers, dim=0)
        shift = torch.norm(new_centers - centers, dim=1).max()
        centers = new_centers

        if shift.item() < tol:
            break

    return centers.clamp(0, 255).detach().cpu().numpy()

# ---------------------- 主流程（清理后，无冗余逻辑）----------------------maxsumc

# 新增：单个调色板npy保存目录
SINGLE_PALETTE_DIR = "../../../CGAN_Sketch_coloring/FlexIcon/test_output/batch/palettenpy"
os.makedirs(SINGLE_PALETTE_DIR, exist_ok=True)
palette_list = []
image_files = os.listdir("../../../CGAN_Sketch_coloring/FlexIcon/test_output/batch")
# 过滤图像文件（避免非图像文件报错）
image_files = [f for f in image_files if f.endswith(('.jpg', '.png', '.jpeg'))]

for index, image_file in enumerate(image_files):
    # 读取图像
    original_image = cv2.imread(os.path.join("../../../CGAN_Sketch_coloring/FlexIcon/test_output/batch", image_file))
    if original_image is None:
        print(f"警告：无法读取图像 {image_file}，跳过")
        continue
    print(f"处理图像：{image_file}")

    # 固定随机种子（确保结果可复现）
    np.random.seed(0)
    random.seed(0)

    # 图像格式转换+向量化（用于KMeans）
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    np.random.seed(0)

    # GPU版KMeans聚类（得到15个颜色中心）
    # 注意：如果torch.cuda.is_available()为False，会自动退回CPU
    centers = kmeans_torch_rgb(
        vectorized,
        n_clusters=15,
        max_iter=60,
        tol=1e-3,
        seed=0,
        device=DEVICE
    )
    center_rgb = np.uint8(centers)  # 15×3 RGB矩阵（uint8）

    # （可选保留）按色相排序15色（不影响MAXSUMC选色，仅调整输入顺序）
    hues = [rgb_to_hue(rgb) for rgb in center_rgb]
    sorted_indices = np.argsort(hues)
    sorted_rgb_matrix = center_rgb[sorted_indices]  # 按色相排序后的15色

    # ---------------------- 核心：调用MAXSUMC选6色 ----------------------
    center_6_ori = maxminc_selection(sorted_rgb_matrix, sample_num=6)  # 直接得到6色RGB

    # ---------------------- （可选删除）异常处理（因n_clusters=15，实际用不上）----------------------
    # 如需删除，直接删掉以下if块即可
    k = 15
    if k in [6, 7]:
        if center_6_ori.ndim != 2:
            if center_6_ori.size % 3 == 0:
                center_6_ori = center_6_ori.reshape(-1, 3)
            else:
                center_6_ori = sorted_rgb_matrix[:6, :3]
        if center_6_ori.shape[-1] != 3:
            center_6_ori = center_6_ori[:, :3] if center_6_ori.shape[-1] > 3 else \
                          np.pad(center_6_ori, ((0,0), (0, 3-center_6_ori.shape[-1])), mode='constant')
        if len(center_6_ori) < 6:
            supplement_count = 6 - len(center_6_ori)
            center_6_ori = np.vstack([center_6_ori, sorted_rgb_matrix[:supplement_count, :3]])

    # 保存结果+绘图预览
    palette_list.append(center_6_ori)
    # 新增：保存单个调色板npy
    palette_name = os.path.splitext(image_file)[0] + ".npy"
    np.save(os.path.join(SINGLE_PALETTE_DIR, palette_name), center_6_ori)
    fig = checkClrs(center_6_ori)

    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(fig)
    save_name='../CGAN_Sketch_coloring/FlexIcon/test_output/batch/palette/' + image_file
    plt.savefig(save_name, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.title(f"Palette of {image_file}")
    plt.close()

# 保存所有调色板结果
palette_list = np.array(palette_list)
# 确保保存目录存在（避免报错）
os.makedirs('../../../CGAN_Sketch_coloring/FlexIcon/test_output/batch/palette', exist_ok=True)
np.save('../../../CGAN_Sketch_coloring/FlexIcon/test_output/batch/palette/palette.npy', palette_list)
print(f"\n所有调色板已保存，共 {len(palette_list)} 个图像的6色调色板")