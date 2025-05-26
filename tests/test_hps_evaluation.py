"""
人类感知评分(HPS)模块测试脚本
测试人类感知评分功能
"""

import os
import sys
import torch
from test_utils import load_test_image, save_test_result, print_score

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from nodes.hps_evaluation import HPSEvaluationNode

def test_hps_evaluation(image_path=None, focus="综合评价", clip_score=0.0, quality_score=0.0):
    """测试人类感知评分模块"""
    print("\n开始测试 人类感知评分(HPS) 模块...")
    
    # 加载测试图像
    image = load_test_image(image_path)
    
    # 创建模块实例
    hps_node = HPSEvaluationNode()
    
    # 运行评估
    score, annotated_image = hps_node.evaluate_hps(image, focus, clip_score, quality_score)
    
    # 打印结果
    print_score("人类感知评分(HPS)", score)
    print(f"评估重点: {focus}")
    print(f"CLIP分数: {clip_score}")
    print(f"质量分数: {quality_score}")
    
    # 保存结果图像
    focus_str = focus.replace(' ', '_')
    output_path = save_test_result(annotated_image, f"hps_{focus_str}_result.png")
    
    print(f"人类感知评分测试完成。得分: {score:.2f}")
    return score, output_path

if __name__ == "__main__":
    # 如果命令行提供了图像路径和评估重点，则使用它们
    import argparse
    parser = argparse.ArgumentParser(description='测试人类感知评分(HPS)模块')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--focus', type=str, choices=['综合评价', '人像质量', '服装细节', '艺术感'], help='评估重点', default="综合评价")
    parser.add_argument('--clip-score', type=float, help='CLIP分数', default=0.0)
    parser.add_argument('--quality-score', type=float, help='质量分数', default=0.0)
    args = parser.parse_args()
    
    test_hps_evaluation(args.image, args.focus, args.clip_score, args.quality_score)
