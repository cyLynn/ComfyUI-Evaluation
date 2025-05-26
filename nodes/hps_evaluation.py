"""
人类感知评分(HPS)模块
模拟人类对AI生成图像的主观评价
"""

import torch
import numpy as np
import cv2
import random

class HPSEvaluationNode:
    """使用人类感知评分(Human Perceptual Score)评估图像质量"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "evaluation_focus": (["综合评价", "人像质量", "服装细节", "艺术感"], {"default": "综合评价"}),
            },
            "optional": {
                "clip_score": ("FLOAT", {"default": 0.0}),
                "quality_score": ("FLOAT", {"default": 0.0}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("hps_score", "annotated_image")
    FUNCTION = "evaluate_hps"
    CATEGORY = "Comfyui-Evaluation/HPS"
    
    def evaluate_hps(self, image, evaluation_focus="综合评价", clip_score=0.0, quality_score=0.0):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 根据评估重点选择不同的评估策略
        if evaluation_focus == "人像质量":
            hps_score, annotated = self._evaluate_portrait(image_np, clip_score, quality_score)
        elif evaluation_focus == "服装细节":
            hps_score, annotated = self._evaluate_clothing(image_np, clip_score, quality_score)
        elif evaluation_focus == "艺术感":
            hps_score, annotated = self._evaluate_artistic(image_np, clip_score, quality_score)
        else:  # 综合评价
            hps_score, annotated = self._evaluate_comprehensive(image_np, clip_score, quality_score)
        
        # 将结果转换回tensor
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0)
        
        return (float(hps_score), annotated_tensor.unsqueeze(0))
    
    def _evaluate_portrait(self, image, clip_score=0.0, quality_score=0.0):
        """评估人像质量"""
        # 分析图像特征
        features = self._extract_image_features(image)
        
        # 计算人像相关特征权重
        weights = {
            "清晰度": 0.3,
            "自然度": 0.25,
            "细节丰富": 0.2,
            "光影效果": 0.15,
            "色彩和谐": 0.1
        }
        
        # 使用质量分数和CLIP分数调整特征评分
        if quality_score > 0:
            features["清晰度"] = (features["清晰度"] + quality_score / 100.0) / 2.0
        
        # 计算加权HPS分数
        hps_score = sum(score * weights[feature] for feature, score in features.items() if feature in weights)
        hps_score = min(1.0, max(0.0, hps_score)) * 100.0
        
        # 创建注释图像
        annotated = self._create_hps_visualization(image, features, weights, hps_score, "人像质量")
        
        return hps_score, annotated
    
    def _evaluate_clothing(self, image, clip_score=0.0, quality_score=0.0):
        """评估服装细节"""
        # 分析图像特征
        features = self._extract_image_features(image)
        
        # 计算服装相关特征权重
        weights = {
            "细节丰富": 0.35,
            "质地表现": 0.25,
            "结构合理": 0.2,
            "清晰度": 0.1,
            "色彩和谐": 0.1
        }
        
        # 使用CLIP分数调整结构合理性评分
        if clip_score > 0:
            features["结构合理"] = (features["结构合理"] + clip_score / 100.0) / 2.0
        
        # 计算加权HPS分数
        hps_score = sum(score * weights[feature] for feature, score in features.items() if feature in weights)
        hps_score = min(1.0, max(0.0, hps_score)) * 100.0
        
        # 创建注释图像
        annotated = self._create_hps_visualization(image, features, weights, hps_score, "服装细节")
        
        return hps_score, annotated
    
    def _evaluate_artistic(self, image, clip_score=0.0, quality_score=0.0):
        """评估艺术感"""
        # 分析图像特征
        features = self._extract_image_features(image)
        
        # 计算艺术相关特征权重
        weights = {
            "创意表达": 0.3,
            "光影效果": 0.25,
            "色彩和谐": 0.2,
            "构图平衡": 0.15,
            "情绪传达": 0.1
        }
        
        # 使用CLIP分数调整情绪传达评分
        if clip_score > 0:
            features["情绪传达"] = (features["情绪传达"] + clip_score / 100.0) / 2.0
        
        # 计算加权HPS分数
        hps_score = sum(score * weights[feature] for feature, score in features.items() if feature in weights)
        hps_score = min(1.0, max(0.0, hps_score)) * 100.0
        
        # 创建注释图像
        annotated = self._create_hps_visualization(image, features, weights, hps_score, "艺术感")
        
        return hps_score, annotated
    
    def _evaluate_comprehensive(self, image, clip_score=0.0, quality_score=0.0):
        """综合评价"""
        # 分析图像特征
        features = self._extract_image_features(image)
        
        # 综合评价的特征权重
        weights = {
            "清晰度": 0.15,
            "细节丰富": 0.15,
            "色彩和谐": 0.15,
            "质地表现": 0.1,
            "光影效果": 0.1,
            "构图平衡": 0.1,
            "自然度": 0.1,
            "创意表达": 0.05,
            "情绪传达": 0.05,
            "结构合理": 0.05
        }
        
        # 使用外部得分调整评分
        if quality_score > 0:
            features["清晰度"] = (features["清晰度"] + quality_score / 100.0) / 2.0
        
        if clip_score > 0:
            features["情绪传达"] = (features["情绪传达"] + clip_score / 100.0) / 2.0
            features["结构合理"] = (features["结构合理"] + clip_score / 100.0) / 2.0
        
        # 计算加权HPS分数
        hps_score = sum(score * weights[feature] for feature, score in features.items() if feature in weights)
        hps_score = min(1.0, max(0.0, hps_score)) * 100.0
        
        # 创建注释图像
        annotated = self._create_hps_visualization(image, features, weights, hps_score, "综合评价")
        
        return hps_score, annotated
    
    def _extract_image_features(self, image):
        """提取图像特征并进行评分"""
        # 实际应用中应使用更复杂的模型
        # 这里使用简化的图像分析方法
        
        # 转为灰度图和HSV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 特征提取
        features = {}
        
        # 1. 清晰度：使用拉普拉斯算子的方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features["清晰度"] = min(1.0, np.var(laplacian) / 500.0)
        
        # 2. 细节丰富：边缘密度
        edges = cv2.Canny(gray, 100, 200)
        features["细节丰富"] = min(1.0, np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 5.0)
        
        # 3. 色彩和谐：颜色直方图分布
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_normalized = hist / np.sum(hist)
        # 计算熵作为色彩多样性指标
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
        features["色彩和谐"] = min(1.0, entropy / 5.0)
        
        # 4. 质地表现：局部方差
        texture_complexity = self._compute_local_std(gray, 8)
        features["质地表现"] = min(1.0, np.mean(texture_complexity) / 30.0)
        
        # 5. 光影效果：亮度对比度
        brightness = np.mean(hsv[:, :, 2])
        brightness_contrast = np.std(hsv[:, :, 2])
        features["光影效果"] = min(1.0, brightness_contrast / 50.0)
        
        # 6. 构图平衡：重心分布
        y_coords, x_coords = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        mass = np.sum(gray)
        if mass > 0:
            center_y = np.sum(gray * y_coords) / mass
            center_x = np.sum(gray * x_coords) / mass
            # 计算中心点偏离度
            center_offset = np.sqrt(((center_y - gray.shape[0]/2) / (gray.shape[0]/2))**2 + 
                                    ((center_x - gray.shape[1]/2) / (gray.shape[1]/2))**2)
            features["构图平衡"] = min(1.0, max(0.0, 1.0 - center_offset))
        else:
            features["构图平衡"] = 0.5
        
        # 7-10. 主观特征：使用随机值模拟
        # 实际应用中应使用专门的模型
        features["自然度"] = self._get_biased_random(0.6, 0.2)
        features["创意表达"] = self._get_biased_random(0.7, 0.2)
        features["情绪传达"] = self._get_biased_random(0.65, 0.2)
        features["结构合理"] = self._get_biased_random(0.75, 0.15)
        
        return features
    
    def _compute_local_std(self, img, patch_size):
        """计算局部标准差作为纹理复杂度度量"""
        from scipy.ndimage import uniform_filter
        
        # 均值滤波
        mean = uniform_filter(img.astype(float), size=patch_size)
        # 平方均值滤波
        mean_sq = uniform_filter(img.astype(float)**2, size=patch_size)
        # 方差 = 平方均值 - 均值的平方
        var = mean_sq - mean**2
        # 标准差
        std = np.sqrt(np.maximum(var, 0))
        
        return std
    
    def _get_biased_random(self, mean, std_dev):
        """生成有偏随机数，用于模拟主观评分"""
        return min(1.0, max(0.0, np.random.normal(mean, std_dev)))
    
    def _create_hps_visualization(self, image, features, weights, hps_score, focus_type):
        """创建HPS评分可视化"""
        h, w = image.shape[:2]
        padding = 30
        chart_height = 300
        chart_width = 400
        
        # 创建可视化画布
        vis_height = max(h, chart_height + 2 * padding)
        vis_width = w + chart_width + padding * 2
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # 放置原图
        visualization[0:h, 0:w] = image
        
        # 绘制雷达图背景
        chart_area = visualization[
            padding:padding+chart_height, 
            w+padding:w+padding+chart_width
        ]
        
        # 添加标题
        cv2.putText(
            visualization,
            f"HPS {focus_type}评分: {hps_score:.1f}",
            (w + padding, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # 绘制各项特征评分条形图
        y_offset = 60
        bar_height = 20
        bar_width = 300
        gap = 25
        
        # 按权重排序特征
        sorted_features = sorted(
            [(f, s, weights.get(f, 0.0)) for f, s in features.items() if f in weights],
            key=lambda x: x[2],
            reverse=True
        )
        
        # 绘制条形图
        for feature_name, score, weight in sorted_features:
            # 特征名称
            cv2.putText(
                visualization,
                f"{feature_name} ({weight*100:.0f}%)",
                (w + padding, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            
            # 条形背景
            cv2.rectangle(
                visualization,
                (w + padding, y_offset + 5),
                (w + padding + bar_width, y_offset + bar_height),
                (50, 50, 50),
                -1
            )
            
            # 条形进度
            score_width = int(bar_width * score)
            
            # 根据分数设置颜色
            if score >= 0.8:
                color = (0, 255, 0)  # 绿色
            elif score >= 0.6:
                color = (0, 255, 255)  # 黄色
            elif score >= 0.4:
                color = (0, 165, 255)  # 橙色
            else:
                color = (0, 0, 255)  # 红色
                
            cv2.rectangle(
                visualization,
                (w + padding, y_offset + 5),
                (w + padding + score_width, y_offset + bar_height),
                color,
                -1
            )
            
            # 显示具体分数
            cv2.putText(
                visualization,
                f"{score*100:.1f}",
                (w + padding + bar_width + 10, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            y_offset += gap
        
        return visualization
