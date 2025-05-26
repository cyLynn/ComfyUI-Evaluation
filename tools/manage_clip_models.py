"""
CLIP模型管理工具
用于下载和管理CLIP模型到ComfyUI的models目录
"""

import os
import sys
import argparse

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入CLIP模块
from nodes.clip_score import CLIPScoreNode

def download_models():
    """下载预定义的CLIP模型到ComfyUI models目录"""
    # 预定义一组常用的CLIP模型
    models = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14"
    ]
    
    for model_name in models:
        print(f"\n下载模型: {model_name}")
        CLIPScoreNode.download_clip_model(model_name)
    
    print("\n所有模型下载完成！")

def list_models():
    """列出当前已下载的CLIP模型"""
    models = CLIPScoreNode.list_local_clip_models()
    
    print("\nCLIP模型列表:")
    print("-" * 40)
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    print("-" * 40)
    
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP模型管理工具')
    parser.add_argument('action', choices=['download', 'list'], help='操作类型: download(下载模型) 或 list(列出已下载模型)')
    parser.add_argument('--model', type=str, help='要下载的模型名称（仅在单独下载时使用）')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        if args.model:
            # 下载指定模型
            CLIPScoreNode.download_clip_model(args.model)
        else:
            # 下载所有预定义模型
            download_models()
    elif args.action == 'list':
        list_models()
