"""
图像结构评估模块测试脚本
测试图像纹理和结构评估功能
"""

import os
import sys
import torch
from test_utils import load_test_image, save_test_result, print_score

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from nodes.texture_structure import TextureStructureNode

def test_texture_structure(image_path=None, ref_image_path=None, method="DISTS"):
    """测试图像结构评估模块"""
    print("\n开始测试 图像结构评估 模块...")
    
    # 加载测试图像
    image = load_test_image(image_path)
    
    # 加载参考图像（如果有）
    ref_image = load_test_image(ref_image_path) if ref_image_path else None
    
    # 创建模块实例
    structure_node = TextureStructureNode()
    
    # 运行评估
    score, annotated_image = structure_node.evaluate_structure(image, method, ref_image)
    
    # 打印结果
    print_score("图像结构", score)
    print(f"使用的方法: {method}")
    print(f"是否使用参考图像: {'是' if ref_image_path else '否'}")
    
    # 保存结果图像
    output_path = save_test_result(annotated_image, f"texture_structure_{method.lower()}_result.png")
    
    print(f"图像结构评估测试完成。得分: {score:.2f}")
    return score, output_path

if __name__ == "__main__":
    # 如果命令行提供了图像路径和方法，则使用它们
    import argparse
    parser = argparse.ArgumentParser(description='测试图像结构评估模块')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--reference', type=str, help='参考图像的路径', default=None)
    parser.add_argument('--method', type=str, choices=['DISTS', 'SSIM', 'LPIPS'], help='结构评估方法', default="DISTS")
    args = parser.parse_args()
    
    test_texture_structure(args.image, args.reference, args.method)
