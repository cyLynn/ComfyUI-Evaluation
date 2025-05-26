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
import os
import folder_paths
import json

class PoseAssessmentNode:
    """评估人体姿态的自然度和肢体比例合理性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取本地OpenPose模型列表
        openpose_models = cls.list_local_openpose_models()
        if not openpose_models:
            openpose_models = ["body_25"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["OpenPose", "MediaPipe"], {"default": "MediaPipe"}),
                "openpose_model": (openpose_models, {"default": openpose_models[0]}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("pose_score", "pose_visualization")
    FUNCTION = "evaluate_pose"
    CATEGORY = "Comfyui-Evaluation/Pose"
    
    @classmethod
    def list_local_openpose_models(cls):
        """列出ComfyUI models/openpose目录下的所有模型"""
        models = []
        openpose_dir = os.path.join(folder_paths.models_dir, "openpose")
        
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
            score, visualization = self._openpose_assessment(image_np, openpose_model)
        
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
            # 获取模型路径
            model_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
            if not os.path.exists(model_dir):
                raise ValueError(f"OpenPose模型 {model_name} 未找到。请先下载模型。")
            
            # 查找模型文件
            prototxt_path = None
            model_path = None
            for file in os.listdir(model_dir):
                if file.endswith('.prototxt'):
                    prototxt_path = os.path.join(model_dir, file)
                elif file.endswith('.caffemodel'):
                    model_path = os.path.join(model_dir, file)
            
            if not prototxt_path or not model_path:
                raise ValueError(f"OpenPose模型文件不完整。请重新下载 {model_name} 模型。")
            
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
