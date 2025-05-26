"""
CLIPScore模块测试脚本
测试图文一致性评分功能
"""

import os
import sys
import torch
from test_utils import load_test_image, save_test_result, print_score

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from nodes.clip_score import CLIPScoreNode

def test_clip_score(image_path=None, prompt="一个穿着漂亮衣服的模特"):
    """测试CLIPScore模块"""
    print("\n开始测试 CLIPScore 模块...")
    
    # 加载测试图像
    image = load_test_image(image_path)
    
    # 创建模块实例
    clip_node = CLIPScoreNode()
    
    # 运行评估
    score, annotated_image = clip_node.evaluate(image, prompt)
    
    # 打印结果
    print_score("CLIPScore", score)
    print(f"使用的提示词: '{prompt}'")
    
    # 保存结果图像
    output_path = save_test_result(annotated_image, "clip_score_result.png")
    
    print(f"CLIPScore 测试完成。得分: {score:.2f}")
    return score, output_path

if __name__ == "__main__":
    # 如果命令行提供了图像路径和提示词，则使用它们
    import argparse
    parser = argparse.ArgumentParser(description='测试CLIPScore模块')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--prompt', type=str, help='文本提示词', default="一个穿着漂亮衣服的模特")
    args = parser.parse_args()
    
    test_clip_score(args.image, args.prompt)
