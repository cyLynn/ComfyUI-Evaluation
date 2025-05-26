"""
姿态准确性评估模块
使用OpenPose或MediaPipe进行人体姿态评估
"""

import torch
import numpy as np
import cv2
import os
import json
import folder_paths
import urllib.request
import shutil
import threading
import time
from pathlib import Path

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

# 全局变量，用于跟踪模型下载状态
downloading_models = {}
download_progress = {}

def ensure_directory(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def download_file(url, destination):
    """从URL下载文件到指定目标"""
    try:
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            file_size = int(response.headers.get('Content-Length', 0))
            block_size = 8192
            downloaded = 0
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # 更新下载进度
                if file_size > 0:
                    model_name = os.path.basename(os.path.dirname(destination))
                    file_name = os.path.basename(destination)
                    key = f"{model_name}/{file_name}"
                    download_progress[key] = (downloaded / file_size) * 100
        
        return True
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return False

def download_model_async(model_name):
    """异步下载OpenPose模型"""
    if model_name not in OPENPOSE_MODELS:
        print(f"错误: 未知的模型 '{model_name}'")
        downloading_models[model_name] = False
        return False
    
    # 设置下载状态
    downloading_models[model_name] = True
    print(f"开始下载OpenPose模型: {model_name}")
    
    # 确定模型目录
    openpose_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
    ensure_directory(openpose_dir)
    
    # 下载模型文件
    model_info = OPENPOSE_MODELS[model_name]
    success = True
    for file_name, url in model_info["files"].items():
        file_path = os.path.join(openpose_dir, file_name)
        key = f"{model_name}/{file_name}"
        download_progress[key] = 0
        
        if not os.path.exists(file_path):
            success = success and download_file(url, file_path)
        else:
            download_progress[key] = 100
            print(f"文件已存在: {file_path}")
    
    # 创建模型信息文件
    info_path = os.path.join(openpose_dir, "model_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            "name": model_name,
            "description": model_info["description"],
            "files": list(model_info["files"].keys())
        }, f, ensure_ascii=False, indent=2)
    
    # 更新下载状态
    downloading_models[model_name] = False
    print(f"OpenPose模型下载完成: {model_name}")
    
    return success

def download_model(model_name):
    """启动异步下载线程"""
    if model_name in downloading_models and downloading_models[model_name]:
        print(f"模型 {model_name} 已经在下载中...")
        return
    
    # 创建新线程进行下载
    thread = threading.Thread(target=download_model_async, args=(model_name,))
    thread.daemon = True
    thread.start()

def is_model_ready(model_name):
    """检查模型是否已准备就绪"""
    # 检查是否在下载中
    if model_name in downloading_models and downloading_models[model_name]:
        return False
    
    # 检查模型文件是否存在
    model_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
    if not os.path.exists(model_dir):
        return False
    
    # 若模型不在OPENPOSE_MODELS中，无法验证文件
    if model_name not in OPENPOSE_MODELS:
        # 简单检查是否有.caffemodel和.prototxt文件
        has_caffemodel = False
        has_prototxt = False
        for file in os.listdir(model_dir):
            if file.endswith('.caffemodel'):
                has_caffemodel = True
            elif file.endswith('.prototxt'):
                has_prototxt = True
        
        return has_caffemodel and has_prototxt
    
    # 检查所有必要文件
    for file_name in OPENPOSE_MODELS[model_name]["files"]:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
    
    return True

def get_model_download_progress(model_name):
    """获取模型下载进度"""
    if model_name not in OPENPOSE_MODELS:
        return 0
    
    # 计算总进度
    total_progress = 0
    file_count = 0
    
    for file_name in OPENPOSE_MODELS[model_name]["files"]:
        key = f"{model_name}/{file_name}"
        if key in download_progress:
            total_progress += download_progress[key]
            file_count += 1
    
    if file_count == 0:
        return 0
    
    return total_progress / file_count

class PoseAssessmentNode:
    """评估人体姿态的自然度和肢体比例合理性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取本地OpenPose模型列表，包括可下载的模型
        openpose_models = cls.list_local_openpose_models()
        if not openpose_models:
            # 添加默认可下载模型
            for model in OPENPOSE_MODELS:
                openpose_models.append(f"{model} (点击下载)")
        
        default_model = next((m for m in openpose_models if "body_25" in m and "下载" not in m), openpose_models[0])
        
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["OpenPose", "MediaPipe"], {"default": "MediaPipe"}),
                "openpose_model": (openpose_models, {"default": default_model}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("pose_score", "pose_visualization")
    FUNCTION = "evaluate_pose"
    CATEGORY = "Comfyui-Evaluation/Pose"
    
    @classmethod
    def list_local_openpose_models(cls):
        """列出ComfyUI models/openpose目录下的所有模型和正在下载的模型"""
        models = []
        available_models = []
        openpose_dir = os.path.join(folder_paths.models_dir, "openpose")
        
        # 检查现有模型
        if os.path.exists(openpose_dir):
            for model_name in os.listdir(openpose_dir):
                model_path = os.path.join(openpose_dir, model_name)
                if os.path.isdir(model_path):
                    # 检查是否包含必要的模型文件
                    has_caffe_model = False
                    has_prototxt = False
                    for file in os.listdir(model_path):
                        if file.endswith('.caffemodel'):
                            has_caffe_model = True
                        elif file.endswith('.prototxt'):
                            has_prototxt = True
                    
                    if has_caffe_model and has_prototxt:
                        models.append(model_name)
                        available_models.append(model_name)
                    elif model_name in downloading_models and downloading_models[model_name]:
                        # 如果模型正在下载中，显示"downloading"
                        download_label = f"{model_name} (正在下载: {get_model_download_progress(model_name):.1f}%)"
                        models.append(download_label)
                        # 在下载完成后能够自动使用
                        available_models.append(model_name)
        
        # 添加未下载但可用的预定义模型
        for model_name in OPENPOSE_MODELS:
            if model_name not in available_models:
                models.append(f"{model_name} (点击下载)")
        
        return models
    
    def evaluate_pose(self, image, method="MediaPipe", openpose_model="body_25"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        if method == "MediaPipe":
            score, visualization = self._mediapipe_assessment(image_np)
        else:  # OpenPose
            # 处理模型名称，移除可能的下载标签
            clean_model_name = openpose_model
            if "(" in openpose_model:
                clean_model_name = openpose_model.split("(")[0].strip()
            
            # 自动下载模型如果需要
            if "下载" in openpose_model and clean_model_name in OPENPOSE_MODELS:
                print(f"用户选择了下载 {clean_model_name} 模型")
                download_model(clean_model_name)
                # 创建下载提示图像
                visualization = image_np.copy()
                cv2.putText(
                    visualization,
                    f"正在下载OpenPose模型: {clean_model_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    visualization,
                    "请等待下载完成后重试",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                # 转换回tensor格式
                vis_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
                return (0.0, vis_tensor.unsqueeze(0))
            
            score, visualization = self._openpose_assessment(image_np, clean_model_name)
        
        # 将结果转换回tensor
        vis_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
        
        return (float(score), vis_tensor.unsqueeze(0))
    
    def _mediapipe_assessment(self, image):
        """使用MediaPipe进行姿态评估"""
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测姿态
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5
            ) as pose:
                results = pose.process(image_rgb)
                
                # 创建可视化图像
                visualization = image.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        visualization,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    # 计算姿态自然度分数
                    landmarks = results.pose_landmarks.landmark
                    
                    # 简单的姿态评分标准: 检查关键点可见性和位置合理性
                    visible_points = sum(1 for landmark in landmarks if landmark.visibility > 0.5)
                    visibility_score = visible_points / len(landmarks)
                    
                    # 检查肢体比例
                    proportion_score = self._check_body_proportions(landmarks)
                    
                    # 综合评分
                    final_score = (visibility_score * 0.4 + proportion_score * 0.6) * 100
                else:
                    final_score = 0.0
                
                # 添加评分信息
                cv2.putText(
                    visualization,
                    f"Pose Score: {final_score:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                return final_score, visualization
                
        except Exception as e:
            print(f"MediaPipe姿态评估出错: {str(e)}")
            return 0.0, image
    
    def _openpose_assessment(self, image, model_name="body_25"):
        """使用OpenPose进行姿态评估"""
        try:
            # 检查模型是否已准备就绪
            if not is_model_ready(model_name):
                # 如果模型不存在或不完整，开始下载
                if model_name not in downloading_models or not downloading_models[model_name]:
                    print(f"OpenPose模型 {model_name} 未找到或不完整，开始自动下载...")
                    download_model(model_name)
                
                # 获取当前下载进度
                progress = get_model_download_progress(model_name)
                
                # 创建带有下载进度的可视化图像
                visualization = image.copy()
                cv2.putText(
                    visualization,
                    f"正在下载OpenPose模型: {model_name} ({progress:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    visualization,
                    "请等待下载完成后重试",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                return 0.0, visualization
            
            # 获取模型路径
            model_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
            
            # 查找模型文件
            prototxt_path = None
            model_path = None
            for file in os.listdir(model_dir):
                if file.endswith('.prototxt'):
                    prototxt_path = os.path.join(model_dir, file)
                elif file.endswith('.caffemodel'):
                    model_path = os.path.join(model_dir, file)
            
            if not prototxt_path or not model_path:
                # 如果找不到文件但目录存在，可能文件损坏，尝试重新下载
                print(f"OpenPose模型文件不完整，尝试重新下载 {model_name} 模型...")
                download_model(model_name)
                
                # 创建错误信息图像
                visualization = image.copy()
                cv2.putText(
                    visualization,
                    f"OpenPose模型文件不完整，正在重新下载: {model_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                return 0.0, visualization
            
            # 加载模型
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            # 设置后端和目标
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # 根据不同模型设置关键点数量和对应关系
            if model_name == "body_25":
                num_points = 25
                pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
                              [0,14],[0,15],[14,16],[15,17],[2,17],[5,16],[8,11]]
                # 设置关键点颜色
                point_colors = [(0,255,255), (0,0,255), (255,0,0), (255,0,0), (255,0,0), (0,255,0), (0,255,0), 
                               (0,255,0), (255,255,0), (255,255,0), (255,255,0), (0,255,255), (0,255,255), 
                               (0,255,255), (255,0,255), (255,0,255), (255,0,255), (255,128,0), (255,128,0), 
                               (255,128,0), (0,128,255), (0,128,255), (0,128,255), (128,0,255), (128,0,255)]
            elif model_name == "coco":
                num_points = 18
                pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
                             [0,14],[0,15],[14,16],[15,17]]
                # 设置关键点颜色
                point_colors = [(0,255,255), (0,0,255), (255,0,0), (255,0,0), (255,0,0), (0,255,0), (0,255,0), 
                               (0,255,0), (255,255,0), (255,255,0), (255,255,0), (0,255,255), (0,255,255), 
                               (0,255,255), (255,0,255), (255,0,255), (255,0,255), (255,128,0)]
            else:  # 默认使用body_25
                num_points = 25
                pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
                              [0,14],[0,15],[14,16],[15,17],[2,17],[5,16],[8,11]]
                point_colors = [(0,255,255)] * 25  # 默认颜色
            
            # 准备输入图像
            height, width = image.shape[:2]
            input_size = 368  # OpenPose标准输入大小
            input_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (input_size, input_size), 
                                              (0, 0, 0), swapRB=False, crop=False)
            
            # 设置网络输入并进行前向传播
            net.setInput(input_blob)
            output = net.forward()
            
            # 提取关键点
            points = []
            for i in range(num_points):
                # 获取热力图
                prob_map = output[0, i, :, :]
                prob_map = cv2.resize(prob_map, (width, height))
                
                # 找到关键点位置
                _, prob, _, point = cv2.minMaxLoc(prob_map)
                
                # 检查置信度阈值
                if prob > 0.1:
                    points.append((point[0], point[1], prob))
                else:
                    points.append(None)
            
            # 创建可视化图像
            visualization = image.copy()
            
            # 绘制关键点和连线
            for i, point in enumerate(points):
                if point:
                    cv2.circle(visualization, (point[0], point[1]), 5, point_colors[i], -1)
            
            for pair in pose_pairs:
                if points[pair[0]] and points[pair[1]]:
                    pt1 = points[pair[0]]
                    pt2 = points[pair[1]]
                    cv2.line(visualization, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 2)
            
            # 计算姿态评分
            valid_points = sum(1 for p in points if p is not None)
            visibility_score = valid_points / num_points if num_points > 0 else 0
            
            # 检查肢体比例
            proportion_score = self._check_openpose_proportions(points, model_name)
            
            # 计算最终分数
            final_score = (visibility_score * 0.4 + proportion_score * 0.6) * 100
            
            # 添加评分信息
            cv2.putText(
                visualization,
                f"Pose Score: {final_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return final_score, visualization
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"OpenPose姿态评估出错: {str(e)}")
            
            # 在发生错误时，显示错误信息
            visualization = image.copy()
            cv2.putText(
                visualization,
                f"OpenPose Error: {str(e)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.putText(
                visualization,
                "Please download models with tools/manage_openpose_models.py",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            return 0.0, visualization
    
    def _check_body_proportions(self, landmarks):
        """检查人体比例的合理性"""
        # 简化版的人体比例检查
        # 人体各部位的标准比例关系
        try:
            # 获取关键点坐标
            # 头部到脚底的距离(身高)
            if landmarks[0].visibility > 0.5 and landmarks[27].visibility > 0.5:
                height = abs(landmarks[0].y - landmarks[27].y)
                
                # 躯干长度
                if landmarks[11].visibility > 0.5 and landmarks[23].visibility > 0.5:
                    torso = abs(landmarks[11].y - landmarks[23].y)
                    
                    # 腿长
                    if landmarks[23].visibility > 0.5 and landmarks[27].visibility > 0.5:
                        leg = abs(landmarks[23].y - landmarks[27].y)
                        
                        # 手臂长度
                        if landmarks[11].visibility > 0.5 and landmarks[15].visibility > 0.5:
                            arm = abs(landmarks[11].y - landmarks[15].y)
                            
                            # 计算比例分数
                            # 理想情况下: 腿长约为身高的0.4-0.5，躯干长约为身高的0.3-0.4
                            leg_ratio = leg / height
                            torso_ratio = torso / height
                            arm_ratio = arm / height
                            
                            leg_score = 1.0 - min(abs(leg_ratio - 0.45) * 5, 1.0)
                            torso_score = 1.0 - min(abs(torso_ratio - 0.35) * 5, 1.0)
                            arm_score = 1.0 - min(abs(arm_ratio - 0.3) * 5, 1.0)
                            
                            return (leg_score + torso_score + arm_score) / 3
            
            return 0.5  # 默认中等分数
        except:
            return 0.5  # 出错时返回中等分数
    
    def _check_openpose_proportions(self, points, model_name):
        """检查OpenPose检测出的人体比例"""
        try:
            if model_name == "body_25":
                # 检查关键点是否有效
                valid_points = [p for p in points if p is not None]
                if len(valid_points) < 10:  # 至少需要10个有效点
                    return 0.5
                
                # 计算身高
                neck = points[1]  # 脖子
                mid_hip = points[8]  # 髋部中点
                
                if neck and mid_hip:
                    # 身高
                    body_height = ((neck[0] - mid_hip[0]) ** 2 + (neck[1] - mid_hip[1]) ** 2) ** 0.5
                    
                    scores = []
                    
                    # 检查腿长
                    right_hip = points[9]
                    right_knee = points[10]
                    right_ankle = points[11]
                    
                    if right_hip and right_knee and right_ankle:
                        upper_leg = ((right_hip[0] - right_knee[0]) ** 2 + (right_hip[1] - right_knee[1]) ** 2) ** 0.5
                        lower_leg = ((right_knee[0] - right_ankle[0]) ** 2 + (right_knee[1] - right_ankle[1]) ** 2) ** 0.5
                        
                        leg_ratio = (upper_leg + lower_leg) / body_height
                        leg_score = 1.0 - min(abs(leg_ratio - 0.5) * 4, 1.0)
                        scores.append(leg_score)
                    
                    # 检查手臂长度
                    right_shoulder = points[2]
                    right_elbow = points[3]
                    right_wrist = points[4]
                    
                    if right_shoulder and right_elbow and right_wrist:
                        upper_arm = ((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2) ** 0.5
                        forearm = ((right_elbow[0] - right_wrist[0]) ** 2 + (right_elbow[1] - right_wrist[1]) ** 2) ** 0.5
                        
                        arm_ratio = (upper_arm + forearm) / body_height
                        arm_score = 1.0 - min(abs(arm_ratio - 0.4) * 4, 1.0)
                        scores.append(arm_score)
                    
                    # 计算最终分数
                    if scores:
                        return sum(scores) / len(scores)
            
            elif model_name == "coco":
                # COCO模型的关键点定义不同
                # 省略实现，可根据需要添加
                pass
                
            return 0.5  # 默认中等分数
        except:
            return 0.5  # 出错时返回中等分数
