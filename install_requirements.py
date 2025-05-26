"""
安装脚本，用于自动安装依赖项
"""

import os
import sys
import subprocess
import importlib

# 需要安装的核心依赖项
CORE_REQUIREMENTS = [
    "torch",
    "numpy",
    "opencv-python",
    "Pillow",
    "scikit-image",
]

# 按功能模块划分的依赖项
MODULE_REQUIREMENTS = {
    "clip_score": ["git+https://github.com/openai/CLIP.git"],
    "pose_assessment": ["mediapipe"],
    "image_quality": ["scikit-image"],
    "fashion_analysis": [], # 实际使用时可能需要 detectron2
    "human_detection": ["mediapipe"],
    "texture_structure": ["scikit-image", "scipy"],
    "hps_evaluation": ["scikit-image", "scipy"],
}

def check_module(module_name):
    """检查模块是否已经安装"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_requirements():
    """安装所有依赖项"""
    print("检查核心依赖项...")
    for req in CORE_REQUIREMENTS:
        module_name = req.split("==")[0].strip()
        if not check_module(module_name.replace("-", "_")):
            print(f"安装 {req}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    
    print("\n检查功能模块依赖项...")
    for module, reqs in MODULE_REQUIREMENTS.items():
        print(f"处理 {module} 模块的依赖...")
        for req in reqs:
            try:
                module_name = req.split("==")[0].strip()
                if "git+" in module_name:
                    # 对于git依赖，只检查最后一部分
                    module_name = module_name.split("/")[-1].split(".")[0]
                
                if not check_module(module_name.replace("-", "_")):
                    print(f"安装 {req}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            except Exception as e:
                print(f"安装 {req} 时出错: {str(e)}")
                print("此依赖可能需要手动安装，或者某些功能可能不可用。")

if __name__ == "__main__":
    print("开始安装 Comfyui-Evaluation 插件依赖...")
    install_requirements()
    print("\n安装完成！")
    print("注意：某些高级功能（如DeepFashion2、Detectron2等）可能需要额外的手动安装步骤。")
    print("请查看README文档获取更多信息。")
