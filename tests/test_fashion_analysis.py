"""
服装结构分析模块测试脚本
测试服装细节和结构评估功能
"""

import os
import sys
import torch
from test_utils import load_test_image, save_test_result, print_score

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from nodes.fashion_analysis import FashionAnalysisNode

def test_fashion_analysis(image_path=None, detail_level="中"):
    """测试服装结构分析模块"""
    print("\n开始测试 服装结构分析 模块...")
    
    # 加载测试图像
    image = load_test_image(image_path)
    
    # 创建模块实例
    fashion_node = FashionAnalysisNode()
    
    # 运行评估
    score, annotated_image = fashion_node.analyze_fashion(image, detail_level)
    
    # 打印结果
    print_score("服装结构", score)
    print(f"使用的详细程度: {detail_level}")
    
    # 保存结果图像
    output_path = save_test_result(annotated_image, f"fashion_analysis_result.png")
    
    print(f"服装结构分析测试完成。得分: {score:.2f}")
    return score, output_path

if __name__ == "__main__":
    # 如果命令行提供了图像路径和详细程度，则使用它们
    import argparse
    parser = argparse.ArgumentParser(description='测试服装结构分析模块')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--detail', type=str, choices=['低', '中', '高'], help='分析详细程度', default="中")
    args = parser.parse_args()
    
    test_fashion_analysis(args.image, args.detail)
