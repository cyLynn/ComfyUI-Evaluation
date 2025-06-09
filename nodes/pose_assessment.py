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
from pathlib import Path
from urllib.parse import urlparse

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
            },
            "optional": {
                "use_proxy": (["全局设置", "使用代理", "不使用代理"], {"default": "全局设置"})
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
                    has_caffe_model = any(f.endswith('.caffemodel') for f in os.listdir(model_path))
                    has_prototxt = any(f.endswith('.prototxt') for f in os.listdir(model_path))
                    if has_caffe_model and has_prototxt:
                        models.append(model_name)
        # 只显示本地存在的模型
        return models

    def _check_model_files(self, model_dir):
        """检查指定目录下是否有有效的caffemodel和prototxt文件"""
        has_caffe_model = False
        has_prototxt = False
        for file in os.listdir(model_dir):
            if file.endswith('.caffemodel') and os.path.getsize(os.path.join(model_dir, file)) > 0:
                has_caffe_model = True
            if file.endswith('.prototxt') and os.path.getsize(os.path.join(model_dir, file)) > 0:
                has_prototxt = True
        return has_caffe_model and has_prototxt

    def evaluate_pose(self, image, method="MediaPipe", openpose_model="body_25", use_proxy="全局设置"):
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
            
            # 检查模型是否存在，不自动下载
            model_dir = os.path.join(folder_paths.models_dir, "openpose", clean_model_name)
            model_exists = os.path.exists(model_dir) and self._check_model_files(model_dir)
            
            if not model_exists:
                # 创建模型不存在的提示图像
                visualization = image_np.copy()
                
                # 添加半透明蒙版
                overlay = visualization.copy()
                cv2.rectangle(overlay, (0, 0), (visualization.shape[1], 180), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, visualization, 0.4, 0, visualization)
                
                # 显示模型缺失信息
                cv2.putText(
                    visualization,
                    f"OpenPose模型未找到: {clean_model_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                cv2.putText(
                    visualization,
                    "请参考README手动下载并放置模型文件",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                cv2.putText(
                    visualization,
                    f"模型路径: {model_dir}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
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
    
    def _get_openpose_pairs(self, model_name):
        """返回每种模型的骨骼连线规则，参考ComfyUI-OpenPose仓库"""
        if model_name == "body_25":
            # 官方body_25骨骼连线
            return [
                [0,1],[1,2],[2,3],[3,4],    # 鼻-脖-右肩-右肘-右腕
                [1,5],[5,6],[6,7],          # 脖-左肩-左肘-左腕
                [1,8],[8,9],[9,10],         # 脖-髋-右膝-右踝
                [1,11],[11,12],[12,13],     # 脖-左髋-左膝-左踝
                [0,15],[15,17],             # 鼻-右眼-右耳
                [0,14],[14,16],             # 鼻-左眼-左耳
                [2,17],[5,16],              # 右肩-右耳，左肩-左耳
                [8,11]                      # 髋-左髋
            ]
        elif model_name == "coco":
            # 官方coco骨骼连线
            return [
                [0,1],[1,2],[2,3],[3,4],
                [1,5],[5,6],[6,7],
                [1,8],[8,9],[9,10],
                [1,11],[11,12],[12,13],
                [0,14],[0,15],[14,16],[15,17]
            ]
        elif model_name == "hand":
            # 官方hand骨骼连线
            return [
                [0,1],[1,2],[2,3],[3,4],
                [0,5],[5,6],[6,7],[7,8],
                [0,9],[9,10],[10,11],[11,12],
                [0,13],[13,14],[14,15],[15,16],
                [0,17],[17,18],[18,19],[19,20]
            ]
        else:
            return []

    def _openpose_assessment(self, image, model_name="body_25", export_pose=False, single_bone=None):
        """
        OpenPose推理，支持：
        - 彩色骨骼点可视化
        - 导出原始pose点数据（export_pose=True）
        - 只画单根骨骼（single_bone=(i,j)）
        """
        try:
            import cv2
            model_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
            prototxt_path = None
            model_path = None
            for file in os.listdir(model_dir):
                if file.endswith('.prototxt'):
                    prototxt_path = os.path.join(model_dir, file)
                elif file.endswith('.caffemodel'):
                    model_path = os.path.join(model_dir, file)
            if not prototxt_path or not model_path:
                raise FileNotFoundError("OpenPose模型文件不完整，请参考README手动下载！")

            # 配置参数和颜色
            if model_name == "body_25":
                num_points = 25
                point_colors = [
                    (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0), (0,255,0),
                    (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255), (0,0,255), (85,0,255),
                    (170,0,255), (255,0,255), (255,0,170), (255,0,85), (128,128,128), (128,0,0), (0,128,0),
                    (0,0,128), (128,128,0), (0,128,128), (128,0,128)
                ]
            elif model_name == "coco":
                num_points = 18
                point_colors = [
                    (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0), (0,255,0),
                    (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255), (0,0,255), (85,0,255),
                    (170,0,255), (255,0,255), (255,0,170), (255,0,85)
                ]
            elif model_name == "hand":
                num_points = 22
                point_colors = [
                    (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0), (0,255,0),
                    (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255), (0,0,255), (85,0,255),
                    (170,0,255), (255,0,255), (255,0,170), (255,0,85), (128,128,128), (128,0,0), (0,128,0), (0,0,128)
                ]
            else:
                raise ValueError(f"未知OpenPose模型: {model_name}")

            pose_pairs = self._get_openpose_pairs(model_name)

            # 推理
            height, width = image.shape[:2]
            input_size = 368
            inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (input_size, input_size), (0, 0, 0), swapRB=False, crop=False)
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            net.setInput(inp)
            out = net.forward()

            points = []
            for i in range(num_points):
                prob_map = out[0, i, :, :]
                prob_map = cv2.resize(prob_map, (width, height))
                _, prob, _, point = cv2.minMaxLoc(prob_map)
                if prob > 0.1:
                    points.append((int(point[0]), int(point[1]), float(prob)))
                else:
                    points.append(None)

            # 导出pose数据
            if export_pose:
                return points

            # 可视化
            visualization = image.copy()
            for i, p in enumerate(points):
                if p:
                    color = point_colors[i % len(point_colors)]
                    cv2.circle(visualization, (p[0], p[1]), 6, color, -1, lineType=cv2.LINE_AA)
            # 连线
            if single_bone is not None:
                # 只画一根骨骼
                i, j = single_bone
                if points[i] and points[j]:
                    cv2.line(visualization, (points[i][0], points[i][1]), (points[j][0], points[j][1]), (0,255,0), 3, lineType=cv2.LINE_AA)
            else:
                for pair in pose_pairs:
                    if points[pair[0]] and points[pair[1]]:
                        cv2.line(visualization, (points[pair[0]][0], points[pair[0]][1]), (points[pair[1]][0], points[pair[1]][1]), (0,255,0), 2, lineType=cv2.LINE_AA)

            # 简单分数
            valid_points = sum(1 for p in points if p is not None)
            score = valid_points / num_points * 100
            cv2.putText(visualization, f"Pose Score: {score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return score, visualization
        except Exception as e:
            import traceback
            traceback.print_exc()
            visualization = image.copy()
            cv2.putText(visualization, f"OpenPose Error: {str(e)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(visualization, "请参考README手动下载模型", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
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
        """检查OpenPose检测出的人体比例，body_25下：全身有下肢就全身评分，否则只算半身"""
        try:
            if model_name == "body_25":
                valid_points = [p for p in points if p is not None]
                if len(valid_points) < 7:  # 至少需要7个有效点
                    return 0.5
                # 判断下肢点
                lower_body_ids = [8,9,10,11,12,13,14]  # 髋、膝、踝
                upper_body_ids = [1,2,3,4,5,6,7]      # 脖、肩、肘、腕
                lower_valid = all(points[i] is not None for i in lower_body_ids)
                upper_valid = all(points[i] is not None for i in upper_body_ids)
                if lower_valid and upper_valid:
                    # 全身评分
                    neck = points[1]
                    mid_hip = points[8]
                    body_height = ((neck[0] - mid_hip[0]) ** 2 + (neck[1] - mid_hip[1]) ** 2) ** 0.5
                    # 腿长
                    right_hip = points[9]
                    right_knee = points[10]
                    right_ankle = points[11]
                    left_hip = points[12]
                    left_knee = points[13]
                    left_ankle = points[14]
                    right_leg = (((right_hip[0] - right_knee[0]) ** 2 + (right_hip[1] - right_knee[1]) ** 2) ** 0.5 +
                                 ((right_knee[0] - right_ankle[0]) ** 2 + (right_knee[1] - right_ankle[1]) ** 2) ** 0.5)
                    left_leg = (((left_hip[0] - left_knee[0]) ** 2 + (left_hip[1] - left_knee[1]) ** 2) ** 0.5 +
                                ((left_knee[0] - left_ankle[0]) ** 2 + (left_knee[1] - left_ankle[1]) ** 2) ** 0.5)
                    leg_ratio = (right_leg + left_leg) / (2 * body_height) if body_height > 0 else 0
                    # 手臂
                    right_shoulder = points[2]
                    right_elbow = points[3]
                    right_wrist = points[4]
                    left_shoulder = points[5]
                    left_elbow = points[6]
                    left_wrist = points[7]
                    right_arm = (((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2) ** 0.5 +
                                 ((right_elbow[0] - right_wrist[0]) ** 2 + (right_elbow[1] - right_wrist[1]) ** 2) ** 0.5)
                    left_arm = (((left_shoulder[0] - left_elbow[0]) ** 2 + (left_shoulder[1] - left_elbow[1]) ** 2) ** 0.5 +
                                ((left_elbow[0] - left_wrist[0]) ** 2 + (left_elbow[1] - left_wrist[1]) ** 2) ** 0.5)
                    arm_ratio = (right_arm + left_arm) / (2 * body_height) if body_height > 0 else 0
                    # 躯干
                    torso = ((neck[0] - mid_hip[0]) ** 2 + (neck[1] - mid_hip[1]) ** 2) ** 0.5
                    torso_ratio = torso / body_height if body_height > 0 else 0
                    leg_score = 1.0 - min(abs(leg_ratio - 0.9) * 2, 1.0)
                    arm_score = 1.0 - min(abs(arm_ratio - 0.7) * 2, 1.0)
                    torso_score = 1.0 - min(abs(torso_ratio - 0.35) * 4, 1.0)
                    return (leg_score + arm_score + torso_score) / 3
                elif upper_valid:
                    # 只评价上半身
                    neck = points[1]
                    right_shoulder = points[2]
                    right_elbow = points[3]
                    right_wrist = points[4]
                    left_shoulder = points[5]
                    left_elbow = points[6]
                    left_wrist = points[7]
                    right_arm = ((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2) ** 0.5 + \
                                ((right_elbow[0] - right_wrist[0]) ** 2 + (right_elbow[1] - right_wrist[1]) ** 2) ** 0.5
                    left_arm = ((left_shoulder[0] - left_elbow[0]) ** 2 + (left_shoulder[1] - left_elbow[1]) ** 2) ** 0.5 + \
                               ((left_elbow[0] - left_wrist[0]) ** 2 + (left_elbow[1] - left_wrist[1]) ** 2) ** 0.5
                    torso = ((neck[0] - ((right_shoulder[0] + left_shoulder[0]) / 2)) ** 2 + (neck[1] - ((right_shoulder[1] + left_shoulder[1]) / 2)) ** 2) ** 0.5
                    arm_ratio = (right_arm + left_arm) / (2 * torso) if torso > 0 else 0
                    arm_score = 1.0 - min(abs(arm_ratio - 1.8) * 2, 1.0)
                    return arm_score
                else:
                    return 0.5
            return 0.5
        except:
            return 0.5

class OpenPoseBody25ProportionNode:
    """
    ComfyUI节点：输入OpenPose body_25格式关键点JSON，输出人体比例合理性分数。
    - 输入：body_25格式的JSON（dict或str），可只含上半身关键点
    - 输出：分数（float），0~100，越高表示比例越合理
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body25_json": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("proportion_score",)
    FUNCTION = "evaluate_body25_proportion"
    CATEGORY = "Comfyui-Evaluation/Pose"

    def evaluate_body25_proportion(self, body25_json):
        import json
        # 解析输入
        if isinstance(body25_json, str):
            try:
                data = json.loads(body25_json)
            except Exception as e:
                print(f"JSON解析失败: {e}")
                return (0.0,)
        else:
            data = body25_json
        # 取关键点
        try:
            people = data["people"]
            if not people or "pose_keypoints_2d" not in people[0]:
                return (0.0,)
            keypoints = people[0]["pose_keypoints_2d"]
            # body_25: 25个点
            if len(keypoints) < 25:
                return (0.0,)
            # 只取前25个点
            keypoints = keypoints[:25]
            # 转为(x, y, conf)列表
            pts = [(float(x), float(y), float(c)) for x, y, c in keypoints]
            # 判断上半身/全身
            lower_body_ids = [8,9,10,11,12,13,14]  # 髋、膝、踝
            upper_body_ids = [1,2,3,4,5,6,7]      # 脖、肩、肘、腕
            lower_valid = all(pts[i][2] > 0.1 for i in lower_body_ids)
            upper_valid = all(pts[i][2] > 0.1 for i in upper_body_ids)
            # 评分
            if lower_valid and upper_valid:
                # 全身比例
                neck = pts[1]
                mid_hip = pts[8]
                body_height = abs(neck[1] - mid_hip[1])
                # 腿长
                right_leg = abs(pts[9][1] - pts[10][1]) + abs(pts[10][1] - pts[11][1])
                left_leg = abs(pts[12][1] - pts[13][1]) + abs(pts[13][1] - pts[14][1])
                leg_ratio = (right_leg + left_leg) / (2 * body_height) if body_height > 0 else 0
                # 手臂
                right_arm = abs(pts[2][1] - pts[3][1]) + abs(pts[3][1] - pts[4][1])
                left_arm = abs(pts[5][1] - pts[6][1]) + abs(pts[6][1] - pts[7][1])
                arm_ratio = (right_arm + left_arm) / (2 * body_height) if body_height > 0 else 0
                # 躯干
                torso = abs(neck[1] - mid_hip[1])
                torso_ratio = torso / body_height if body_height > 0 else 0
                # 理想比例
                leg_score = 1.0 - min(abs(leg_ratio - 0.9) * 2, 1.0)
                arm_score = 1.0 - min(abs(arm_ratio - 0.7) * 2, 1.0)
                torso_score = 1.0 - min(abs(torso_ratio - 0.35) * 4, 1.0)
                score = (leg_score + arm_score + torso_score) / 3 * 100
                return (float(score),)
            elif upper_valid:
                # 只评价上半身
                neck = pts[1]
                right_shoulder = pts[2]
                right_elbow = pts[3]
                right_wrist = pts[4]
                left_shoulder = pts[5]
                left_elbow = pts[6]
                left_wrist = pts[7]
                # 右臂
                right_arm = abs(right_shoulder[1] - right_elbow[1]) + abs(right_elbow[1] - right_wrist[1])
                # 左臂
                left_arm = abs(left_shoulder[1] - left_elbow[1]) + abs(left_elbow[1] - left_wrist[1])
                # 躯干
                torso = abs(neck[1] - (right_shoulder[1] + left_shoulder[1]) / 2)
                arm_ratio = (right_arm + left_arm) / (2 * torso) if torso > 0 else 0
                arm_score = 1.0 - min(abs(arm_ratio - 1.8) * 2, 1.0)
                score = arm_score * 100
                return (float(score),)
            else:
                return (0.0,)
        except Exception as e:
            print(f"body25比例评分异常: {e}")
            return (0.0,)
