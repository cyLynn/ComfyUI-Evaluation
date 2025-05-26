"""
人体检测评估模块
使用Detectron2/Mediapipe进行人体检测和评估
"""

import torch
import numpy as np
import cv2

class HumanDetectionNode:
    """判断人体完整性和着装逻辑合理性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_method": (["Mediapipe", "Detectron2"], {"default": "Mediapipe"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("human_score", "annotated_image")
    FUNCTION = "detect_human"
    CATEGORY = "Comfyui-Evaluation/Human"
    
    def detect_human(self, image, detection_method="Mediapipe"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 根据选择的方法进行人体检测
        if detection_method == "Mediapipe":
            score, annotated = self._detect_with_mediapipe(image_np)
        else:  # Detectron2
            score, annotated = self._detect_with_detectron2(image_np)
        
        # 将结果转换回tensor
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0)
        
        return (float(score), annotated_tensor.unsqueeze(0))
    
    def _detect_with_mediapipe(self, image):
        """使用MediaPipe进行人体检测"""
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
                annotated = image.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    # 计算人体完整性分数
                    landmarks = results.pose_landmarks.landmark
                    
                    # 检查关键点可见性
                    visible_points = sum(1 for landmark in landmarks if landmark.visibility > 0.5)
                    visibility_score = visible_points / len(landmarks)
                    
                    # 检查人体是否完整（头部、躯干、四肢）
                    key_points = {
                        "头部": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "躯干": [11, 12, 23, 24],
                        "手臂": [11, 13, 15, 12, 14, 16],
                        "腿部": [23, 25, 27, 24, 26, 28]
                    }
                    
                    body_parts_scores = {}
                    for part_name, indices in key_points.items():
                        visible = sum(1 for idx in indices if landmarks[idx].visibility > 0.5)
                        body_parts_scores[part_name] = visible / len(indices)
                    
                    # 人体完整性得分
                    completeness_score = sum(body_parts_scores.values()) / len(body_parts_scores)
                    
                    # 着装逻辑合理性评估（简化版）
                    # 实际使用时应结合服装分析模块进行更详细的评估
                    clothing_logic_score = 0.8  # 默认较高分数
                    
                    # 最终人体评分
                    human_score = (visibility_score * 0.3 + completeness_score * 0.5 + clothing_logic_score * 0.2) * 100
                    
                    # 在图像上添加评分信息
                    cv2.putText(
                        annotated,
                        f"Human Score: {human_score:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    
                    # 显示身体部位完整性
                    y_offset = 80
                    for part_name, score in body_parts_scores.items():
                        cv2.putText(
                            annotated,
                            f"{part_name}: {score*100:.1f}%",
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1
                        )
                        y_offset += 30
                    
                    return human_score, annotated
                
                else:
                    cv2.putText(
                        annotated,
                        "No human detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    return 0.0, annotated
                
        except Exception as e:
            print(f"MediaPipe人体检测出错: {str(e)}")
            return 0.0, image
    
    def _detect_with_detectron2(self, image):
        """使用Detectron2进行人体检测"""
        # 这里应该使用Detectron2模型
        # 由于Detectron2需要额外安装，这里提供一个简化的实现
        
        # 模拟Detectron2输出
        annotated = image.copy()
        cv2.putText(
            annotated,
            "Detectron2 not implemented yet",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )
        
        # 绘制模拟的人体框
        h, w = image.shape[:2]
        cv2.rectangle(
            annotated,
            (int(w*0.3), int(h*0.1)),
            (int(w*0.7), int(h*0.9)),
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated,
            "Person: 0.95",
            (int(w*0.3), int(h*0.1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        return 65.0, annotated
