"""
综合评分系统测试脚本
测试多项指标综合评分功能
"""

import os
import sys
import torch
from test_utils import load_test_image, save_test_result, print_score

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from nodes.combined_evaluation import CombinedEvaluationNode

def test_combined_evaluation(image_path=None, prompt="一个穿着漂亮衣服的模特", 
                            clip_score=75.0, pose_score=80.0, quality_score=85.0,
                            fashion_score=70.0, human_score=90.0, structure_score=65.0):
    """测试综合评分系统"""
    print("\n开始测试 综合评分系统 模块...")
    
    # 加载测试图像
    image = load_test_image(image_path)
    
    # 创建模块实例
    combined_node = CombinedEvaluationNode()
    
    # 运行评估
    combined_score, visualization = combined_node.evaluate_combined(
        image, prompt, clip_score, pose_score, quality_score,
        fashion_score, human_score, structure_score
    )
    
    # 打印结果
    print_score("综合评分", combined_score)
    print(f"使用的提示词: '{prompt}'")
    print(f"各项评分:")
    print(f"  图文一致性: {clip_score:.2f}")
    print(f"  姿态准确性: {pose_score:.2f}")
    print(f"  图像质量: {quality_score:.2f}")
    print(f"  服装分析: {fashion_score:.2f}")
    print(f"  人体完整性: {human_score:.2f}")
    print(f"  结构评分: {structure_score:.2f}")
    
    # 保存结果图像
    output_path = save_test_result(visualization, "combined_evaluation_result.png")
    
    print(f"综合评分系统测试完成。得分: {combined_score:.2f}")
    return combined_score, output_path

if __name__ == "__main__":
    # 如果命令行提供了参数，则使用它们
    import argparse
    parser = argparse.ArgumentParser(description='测试综合评分系统')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--prompt', type=str, help='文本提示词', default="一个穿着漂亮衣服的模特")
    parser.add_argument('--clip-score', type=float, help='CLIP分数', default=75.0)
    parser.add_argument('--pose-score', type=float, help='姿态准确性分数', default=80.0)
    parser.add_argument('--quality-score', type=float, help='图像质量分数', default=85.0)
    parser.add_argument('--fashion-score', type=float, help='服装分析分数', default=70.0)
    parser.add_argument('--human-score', type=float, help='人体完整性分数', default=90.0)
    parser.add_argument('--structure-score', type=float, help='结构评分', default=65.0)
    args = parser.parse_args()
    
    test_combined_evaluation(
        args.image, args.prompt, args.clip_score, args.pose_score, 
        args.quality_score, args.fashion_score, args.human_score, args.structure_score
    )
