"""
Comfyui-Evaluation: AI生成图片评估插件
提供多种评估节点，用于全面评价AI生成的图像质量、真实性和符合度
"""

import os
import sys
import folder_paths

# 确保依赖库可用
def check_and_install_requirements():
    try:
        import torch
        import cv2
        import clip
        import numpy as np
    except ImportError:
        print("正在安装必要依赖...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "opencv-python", "git+https://github.com/openai/CLIP.git", "numpy", "scikit-image"])

# 注册文件夹路径
comfy_path = os.path.dirname(folder_paths.__file__)
module_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_path)

# 导入各个模块
from .nodes.clip_score import CLIPScoreNode
from .nodes.pose_assessment import PoseAssessmentNode
from .nodes.image_quality import ImageQualityNode
from .nodes.fashion_analysis import FashionAnalysisNode
from .nodes.human_detection import HumanDetectionNode
from .nodes.texture_structure import TextureStructureNode
from .nodes.combined_evaluation import CombinedEvaluationNode
from .nodes.hps_evaluation import HPSEvaluationNode

# 节点映射
NODE_CLASS_MAPPINGS = {
    "CLIPScoreNode": CLIPScoreNode,
    "PoseAssessmentNode": PoseAssessmentNode,
    "ImageQualityNode": ImageQualityNode, 
    "FashionAnalysisNode": FashionAnalysisNode,
    "HumanDetectionNode": HumanDetectionNode,
    "TextureStructureNode": TextureStructureNode,
    "CombinedEvaluationNode": CombinedEvaluationNode,
    "HPSEvaluationNode": HPSEvaluationNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPScoreNode": "图文一致性评分",
    "PoseAssessmentNode": "姿态准确性评估",
    "ImageQualityNode": "图像质量评估",
    "FashionAnalysisNode": "服装结构分析",
    "HumanDetectionNode": "人体检测评估",
    "TextureStructureNode": "图像结构评估",
    "CombinedEvaluationNode": "综合评分系统",
    "HPSEvaluationNode": "人类感知评分(HPS)"
}

print("Comfyui-Evaluation插件已加载")