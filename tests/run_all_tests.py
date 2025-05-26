"""
整合测试脚本
依次测试所有评估模块
"""

import os
import sys
import time
import argparse

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import load_test_image
from test_clip_score import test_clip_score
from test_pose_assessment import test_pose_assessment
from test_image_quality import test_image_quality
from test_fashion_analysis import test_fashion_analysis
from test_human_detection import test_human_detection
from test_texture_structure import test_texture_structure
from test_hps_evaluation import test_hps_evaluation
from test_combined_evaluation import test_combined_evaluation

def run_all_tests(image_path=None, prompt="一个穿着漂亮衣服的模特"):
    """
    依次运行所有评估模块的测试
    """
    print("\n" + "="*70)
    print("开始运行 Comfyui-Evaluation 插件的所有模块测试")
    print("="*70)
    
    start_time = time.time()
    results = {}
    
    # 测试CLIPScore
    print("\n[1/8] 测试 CLIPScore 模块")
    clip_score, _ = test_clip_score(image_path, prompt)
    results["CLIPScore"] = clip_score
    
    # 测试姿态准确性
    print("\n[2/8] 测试 姿态准确性 模块")
    pose_score, _ = test_pose_assessment(image_path)
    results["姿态准确性"] = pose_score
    
    # 测试图像质量
    print("\n[3/8] 测试 图像质量评估 模块")
    quality_score, _ = test_image_quality(image_path)
    results["图像质量"] = quality_score
    
    # 测试服装分析
    print("\n[4/8] 测试 服装结构分析 模块")
    fashion_score, _ = test_fashion_analysis(image_path)
    results["服装分析"] = fashion_score
    
    # 测试人体检测
    print("\n[5/8] 测试 人体检测评估 模块")
    human_score, _ = test_human_detection(image_path)
    results["人体检测"] = human_score
    
    # 测试图像结构
    print("\n[6/8] 测试 图像结构评估 模块")
    structure_score, _ = test_texture_structure(image_path)
    results["图像结构"] = structure_score
    
    # 测试HPS评分
    print("\n[7/8] 测试 人类感知评分 模块")
    hps_score, _ = test_hps_evaluation(image_path, "综合评价", clip_score, quality_score)
    results["HPS评分"] = hps_score
    
    # 测试综合评分系统
    print("\n[8/8] 测试 综合评分系统 模块")
    combined_score, _ = test_combined_evaluation(
        image_path, prompt, clip_score, pose_score, quality_score, 
        fashion_score, human_score, structure_score
    )
    results["综合评分"] = combined_score
    
    # 打印所有结果
    print("\n" + "="*70)
    print("所有测试完成！总用时: {:.2f}秒".format(time.time() - start_time))
    print("="*70)
    print("\n评分结果汇总:")
    print("-"*40)
    for name, score in results.items():
        print(f"{name}: {score:.2f}")
    print("-"*40)
    print(f"测试图像: {image_path if image_path else '随机生成图像'}")
    print(f"提示词: '{prompt}'")
    print("\n结果图像保存在: tests/outputs/ 目录下")
    
    return results

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行所有评估模块测试')
    parser.add_argument('--image', type=str, help='测试图像的路径', default=None)
    parser.add_argument('--prompt', type=str, help='文本提示词', default="一个穿着漂亮衣服的模特")
    args = parser.parse_args()
    
    # 运行所有测试
    run_all_tests(args.image, args.prompt)
