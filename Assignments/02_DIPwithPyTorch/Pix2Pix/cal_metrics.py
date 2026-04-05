import numpy as np
from PIL import Image

# ==========================================
# 1. 定义你的类别颜色字典 (需根据你的数据集实际情况修改)
# 以下颜色参考了类似 Cityscapes 数据集的常见配色
# 格式为: 类别名称 -> (R, G, B)
# ==========================================
COLOR_DICT = {
    'background/unlabeled': (0, 0, 0),       # 黑色
    'road': (128, 64, 128),                  # 紫色
    'sidewalk': (244, 35, 232),              # 粉色
    'vegetation': (107, 142, 35),            # 绿色
    'vehicle': (0, 0, 142),                  # 深蓝色
    'person/pedestrian': (220, 20, 60),      # 红色
    'pole/traffic_light': (255, 255, 0),     # 黄色
    'building': (70, 70, 70),                # 灰色
}

# 提取类别名称和颜色矩阵
CLASS_NAMES = list(COLOR_DICT.keys())
PALETTE = np.array(list(COLOR_DICT.values()), dtype=np.int32) # 形状: (num_classes, 3)
NUM_CLASSES = len(CLASS_NAMES)

def rgb_to_class_indices(rgb_image, palette):
    """
    将 RGB 图像映射为单通道的类别索引图。
    利用欧式距离寻找每个像素最接近的调色板颜色，以应对预测图中的颜色模糊。
    """
    # 展平图像: (H, W, 3) -> (H*W, 3)
    flat_image = rgb_image.reshape(-1, 3).astype(np.float32)
    
    # 计算每个像素与调色板中所有颜色的距离
    # flat_image[:, None, :] 形状: (H*W, 1, 3)
    # palette[None, :, :] 形状: (1, num_classes, 3)
    # 广播相减后求 L2 范数，得到形状 (H*W, num_classes) 的距离矩阵
    distances = np.linalg.norm(flat_image[:, None, :] - palette[None, :, :], axis=2)
    
    # 获取距离最小的索引作为类别标签
    class_indices = np.argmin(distances, axis=1)
    
    # 恢复为 (H, W) 形状
    return class_indices.reshape(rgb_image.shape[:2])

def calculate_miou(pred_mask, true_mask, num_classes):
    """利用混淆矩阵计算 Mean IoU (沿用之前的高效方法)"""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # 混淆矩阵
    confusion_matrix = np.bincount(
        num_classes * true_flat + pred_flat, 
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    ious = {}
    valid_ious = []
    
    for i in range(num_classes):
        if union[i] == 0:
            ious[CLASS_NAMES[i]] = float('nan')
        else:
            class_iou = intersection[i] / union[i]
            ious[CLASS_NAMES[i]] = class_iou
            # 如果该类别是背景（索引为0），你可以选择是否将其计入 Mean IoU
            # 这里默认所有出现的类别都计入平均值
            valid_ious.append(class_iou)

    mean_iou = np.nanmean(valid_ious)
    return mean_iou, ious

def process_and_evaluate(image_path):
    # 1. 读取拼接图像 (确保是以 RGB 模式读取)
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # 2. 图像裁剪 (假设总尺寸为 768x256，宽度方向拼接)
    # 原始图: 0~256, 真实标签: 256~512, 预测结果: 512~768
    # 注意 numpy 切片格式为 [高度, 宽度, 通道]
    true_rgb = img_array[:, 256:512, :]
    pred_rgb = img_array[:, 512:768, :]
    
    # 3. 将 RGB 转换为类别索引 (0, 1, 2...)
    true_mask = rgb_to_class_indices(true_rgb, PALETTE)
    pred_mask = rgb_to_class_indices(pred_rgb, PALETTE)
    
    # 4. 计算 IoU
    mIoU, class_ious = calculate_miou(pred_mask, true_mask, NUM_CLASSES)
    
    print(f"Mean IoU: {mIoU:.4f}")
    print("-" * 30)
    for cls_name, iou_val in class_ious.items():
        if np.isnan(iou_val):
            print(f"{cls_name}: 未在图像中出现 (忽略)")
        else:
            print(f"{cls_name}: {iou_val:.4f}")

# ==========================================
# 运行脚本
# ==========================================
if __name__ == "__main__":
    # 请将 'image_f476a2.png' 替换为你实际的图片路径
    image_path = 'val_results/epoch_295/result_4.png' 
    process_and_evaluate(image_path)