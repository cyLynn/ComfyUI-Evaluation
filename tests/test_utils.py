"""
测试工具函数，用于测试脚本
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_test_image(image_path=None):
    """
    加载测试图像，如果未提供路径则创建一个简单的测试图像
    
    Args:
        image_path: 图像路径，如果为None则创建一个测试图像
        
    Returns:
        torch.Tensor: 形状为(3, H, W)的图像张量
    """
    if image_path and os.path.exists(image_path):
        try:
            # 使用PIL加载图像
            pil_image = Image.open(image_path).convert("RGB")
            # 转换为numpy数组
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            # 转换为torch tensor
            tensor_image = torch.from_numpy(np_image.transpose(2, 0, 1))
            return tensor_image
        except Exception as e:
            print(f"加载图像出错: {str(e)}")
    
    # 创建一个简单的测试图像 (随机噪声)
    print("创建随机测试图像...")
    img = np.random.rand(512, 512, 3).astype(np.float32)
    tensor_image = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor_image

def save_test_result(image_tensor, filename):
    """
    保存测试结果图像
    
    Args:
        image_tensor: 形状为(C, H, W)或(1, C, H, W)的图像张量
        filename: 保存的文件名
    """
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    
    # 转换为numpy数组
    np_image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    # 缩放到0-255
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
    
    # 确保tests/outputs目录存在
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    print(f"结果已保存至: {output_path}")
    
    return output_path

def print_score(module_name, score):
    """
    打印评分结果
    """
    print("="*50)
    print(f"{module_name} 评分结果: {score:.2f}")
    print("="*50)
