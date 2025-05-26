"""
工具函数模块，提供共享功能
"""

import cv2
import numpy as np
import torch
from PIL import Image

def tensor_to_pil(tensor):
    """将tensor转换为PIL图像"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    # 转换为numpy数组并缩放到0-255
    img_np = tensor.cpu().numpy().transpose(1, 2, 0) * 255
    img_np = img_np.clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def tensor_to_cv2(tensor):
    """将tensor转换为OpenCV图像"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img_np = tensor.cpu().numpy()
    # 转换通道顺序 (C,H,W) -> (H,W,C)
    img_np = img_np.transpose(1, 2, 0)
    # 缩放到0-255
    img_np = (img_np * 255).astype(np.uint8)
    # 转换颜色空间 RGB -> BGR
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2

def pil_to_tensor(pil_image):
    """将PIL图像转换为tensor"""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    # 确保图像有3个通道
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=2)
    # 转换通道顺序 (H,W,C) -> (C,H,W)
    img_np = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_np).unsqueeze(0)

def cv2_to_tensor(img_cv2):
    """将OpenCV图像转换为tensor"""
    # 转换颜色空间 BGR -> RGB
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # 缩放到0-1
    img_np = img_rgb.astype(np.float32) / 255.0
    # 转换通道顺序 (H,W,C) -> (C,H,W)
    img_np = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_np).unsqueeze(0)

def draw_score_label(image, score, label, position=(20, 40), color=(255, 255, 255)):
    """绘制带有分数的标签"""
    cv2.putText(
        image,
        f"{label}: {score:.2f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )
    return image
