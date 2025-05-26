"""
OpenPose模型管理工具
用于下载和管理OpenPose模型到ComfyUI的models目录
"""

import os
import sys
import argparse
import urllib.request
import shutil
import json
import folder_paths

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模型URLs和文件信息
OPENPOSE_MODELS = {
    "body_25": {
        "description": "BODY_25模型 - 25个关键点的人体姿势模型",
        "files": {
            "pose_iter_584000.caffemodel": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
            "pose_deploy.prototxt": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt"
        }
    },
    "coco": {
        "description": "COCO模型 - 18个关键点的人体姿势模型",
        "files": {
            "pose_iter_440000.caffemodel": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
            "pose_deploy_linevec.prototxt": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
        }
    },
    "hand": {
        "description": "手部姿势模型",
        "files": {
            "pose_iter_102000.caffemodel": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel",
            "pose_deploy.prototxt": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt"
        }
    }
}

def ensure_directory(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def download_file(url, destination):
    """从URL下载文件到指定目标"""
    print(f"正在下载: {url}")
    try:
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"下载完成: {destination}")
        return True
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return False

def download_model(model_name):
    """下载指定的OpenPose模型"""
    if model_name not in OPENPOSE_MODELS:
        print(f"错误: 未知的模型 '{model_name}'")
        return False
    
    # 确定模型目录
    openpose_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
    ensure_directory(openpose_dir)
    
    # 下载模型文件
    model_info = OPENPOSE_MODELS[model_name]
    success = True
    for file_name, url in model_info["files"].items():
        file_path = os.path.join(openpose_dir, file_name)
        if not os.path.exists(file_path):
            success = success and download_file(url, file_path)
        else:
            print(f"文件已存在: {file_path}")
    
    # 创建模型信息文件
    info_path = os.path.join(openpose_dir, "model_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            "name": model_name,
            "description": model_info["description"],
            "files": list(model_info["files"].keys())
        }, f, ensure_ascii=False, indent=2)
    
    return success

def download_all_models():
    """下载所有预定义的OpenPose模型"""
    print("\n开始下载所有OpenPose模型...")
    
    # 确保OpenPose目录存在
    openpose_dir = os.path.join(folder_paths.models_dir, "openpose")
    ensure_directory(openpose_dir)
    
    # 下载每个模型
    success = True
    for model_name in OPENPOSE_MODELS:
        print(f"\n下载模型: {model_name}")
        success = success and download_model(model_name)
    
    if success:
        print("\n所有模型下载完成！")
    else:
        print("\n部分模型下载失败，请检查错误信息。")
    
    return success

def list_models():
    """列出当前已下载的OpenPose模型"""
    openpose_dir = os.path.join(folder_paths.models_dir, "openpose")
    if not os.path.exists(openpose_dir):
        print("OpenPose模型目录不存在")
        return []
    
    models = []
    for model_name in os.listdir(openpose_dir):
        model_path = os.path.join(openpose_dir, model_name)
        if os.path.isdir(model_path):
            info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    models.append({
                        "name": info.get("name", model_name),
                        "description": info.get("description", ""),
                        "path": model_path
                    })
                except:
                    models.append({
                        "name": model_name,
                        "description": "无法读取模型信息",
                        "path": model_path
                    })
            else:
                models.append({
                    "name": model_name,
                    "description": "未找到模型信息文件",
                    "path": model_path
                })
    
    if not models:
        print("\n未找到已下载的OpenPose模型。")
        print("使用 'python tools/manage_openpose_models.py download' 下载模型。")
    else:
        print("\nOpenPose模型列表:")
        print("-" * 60)
        for i, model in enumerate(models):
            print(f"{i+1}. {model['name']} - {model['description']}")
        print("-" * 60)
    
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenPose模型管理工具')
    parser.add_argument('action', choices=['download', 'list'], help='操作类型: download(下载模型) 或 list(列出已下载模型)')
    parser.add_argument('--model', type=str, help='要下载的模型名称（仅在单独下载时使用）')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        if args.model:
            # 下载指定模型
            download_model(args.model)
        else:
            # 下载所有预定义模型
            download_all_models()
    elif args.action == 'list':
        list_models()
