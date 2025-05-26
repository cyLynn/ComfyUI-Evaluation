"""
姿态准确性评估模块
使用OpenPose或MediaPipe进行人体姿态评估
"""

import torch
import numpy as np
import cv2

class PoseAssessmentNode:
    """评估人体姿态的自然度和肢体比例合理性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["OpenPose", "MediaPipe"], {"default": "MediaPipe"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("pose_score", "pose_visualization")
    FUNCTION = "evaluate_pose"
    CATEGORY = "Comfyui-Evaluation/Pose"
    
    def evaluate_pose(self, image, method="MediaPipe"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        if method == "MediaPipe":
            score, visualization = self._mediapipe_assessment(image_np)
        else:  # OpenPose
            score, visualization = self._openpose_assessment(image_np)
        
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
    
    def _openpose_assessment(self, image):
        """使用OpenPose进行姿态评估"""
        try:
            # 这里应该调用OpenPose模型
            # 由于OpenPose需要额外安装，这里提供一个简化的实现
            # 实际使用时应当替换为真实的OpenPose调用
            
            # 模拟OpenPose输出
            visualization = image.copy()
            cv2.putText(
                visualization,
                "OpenPose not implemented yet",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
            
            return 50.0, visualization
            
        except Exception as e:
            print(f"OpenPose姿态评估出错: {str(e)}")
            return 0.0, image
    
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
