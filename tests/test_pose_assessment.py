"""
姿态准确性模块测试脚本
测试人体姿态评估功能
"""

import os
import sys
import torch
from test_utils import load_test_image, save_test_result, print_score

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from nodes.pose_assessment import PoseAssessmentNode

def test_pose_assessment(image_path=None, method="MediaPipe"):
    """测试姿态准确性评估模块"""
    print("\n开始测试 姿态准确性评估 模块...")
    
    # 加载测试图像
    image = load_test_image(image_path)
    
    # 创建模块实例
    pose_node = PoseAssessmentNode()
    
    # 运行评估
    score, annotated_image = pose_node.evaluate_pose(image, method)
    
    # 打印结果
    print_score("姿态准确性", score)
    print(f"使用的方法: {method}")
    
    # 保存结果图像
    output_path = save_test_result(annotated_image, f"pose_assessment_{method.lower()}_result.png")
    
    print(f"姿态准确性测试完成。得分: {score:.2f}")
    return score, output_path

if __name__ == "__main__":
    # 如果命令行提供了图像路径和方法，则使用它们
    import argparse
    parser = argparse.ArgumentParser(description='测试姿态准确性评估模块')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--method', type=str, choices=['MediaPipe', 'OpenPose'], help='姿态评估方法', default="MediaPipe")
    args = parser.parse_args()
    
    test_pose_assessment(args.image, args.method)
