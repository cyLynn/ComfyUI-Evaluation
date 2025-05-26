"""
综合评分系统
集成各个模块的评分，提供整体评价
"""

import torch
import numpy as np
import cv2

class CombinedEvaluationNode:
    """集成各模块分数，提供整体评价和可视化评分结果"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "clip_score": ("FLOAT", {"default": 0.0}),
                "pose_score": ("FLOAT", {"default": 0.0}),
                "quality_score": ("FLOAT", {"default": 0.0}),
                "fashion_score": ("FLOAT", {"default": 0.0}),
                "human_score": ("FLOAT", {"default": 0.0}),
                "structure_score": ("FLOAT", {"default": 0.0}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("combined_score", "visualization")
    FUNCTION = "evaluate_combined"
    CATEGORY = "Comfyui-Evaluation/Combined"
    
    def evaluate_combined(self, image, prompt, 
                          clip_score=0.0, pose_score=0.0, quality_score=0.0, 
                          fashion_score=0.0, human_score=0.0, structure_score=0.0):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 计算缺失的评分
        # 如果某个分数为0，可能是未连接对应节点，可以在这里通过调用其他模块来补充
        scores = {
            "图文一致性": clip_score if clip_score > 0 else self._compute_clip_score(image_np, prompt),
            "姿态准确性": pose_score if pose_score > 0 else 0.0,
            "图像质量": quality_score if quality_score > 0 else self._compute_quality_score(image_np),
            "服装分析": fashion_score if fashion_score > 0 else 0.0,
            "人体完整性": human_score if human_score > 0 else 0.0,
            "结构评分": structure_score if structure_score > 0 else self._compute_structure_score(image_np),
        }
        
        # 各项评分的权重
        weights = {
            "图文一致性": 0.25,
            "姿态准确性": 0.15,
            "图像质量": 0.2,
            "服装分析": 0.15,
            "人体完整性": 0.15,
            "结构评分": 0.1,
        }
        
        # 计算加权综合评分
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        if valid_scores:
            # 调整权重，使总权重为1
            total_weight = sum(weights[k] for k in valid_scores.keys())
            adjusted_weights = {k: weights[k]/total_weight for k in valid_scores.keys()}
            
            # 计算加权评分
            combined_score = sum(v * adjusted_weights[k] for k, v in valid_scores.items())
        else:
            # 没有有效评分时返回50分
            combined_score = 50.0
        
        # 生成评分可视化
        visualization = self._create_score_visualization(image_np, scores, combined_score)
        
        # 将结果转换回tensor
        vis_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
        
        return (float(combined_score), vis_tensor.unsqueeze(0))
    
    def _compute_clip_score(self, image, prompt):
        """简化版的CLIP评分计算"""
        # 简单返回一个默认分数，实际应调用CLIPScoreNode
        return 0.0
    
    def _compute_quality_score(self, image):
        """简化版的图像质量评分"""
        # 使用图像清晰度作为简单评分指标
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return min(100.0, max(0.0, np.var(laplacian) * 0.01))
    
    def _compute_structure_score(self, image):
        """简化版的图像结构评分"""
        # 使用边缘密度作为简单结构评分指标
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return min(100.0, max(0.0, edge_density * 500))
    
    def _create_score_visualization(self, image, scores, combined_score):
        """创建评分可视化图像"""
        # 图像尺寸
        h, w = image.shape[:2]
        padding = 50
        chart_height = 300
        
        # 创建底图
        if h > w:
            vis_height = h
            vis_width = w + chart_height + padding * 2
            image_pos = (0, 0)
            chart_pos = (w + padding, padding)
            chart_size = (chart_height, h - padding * 2)
        else:
            vis_width = w
            vis_height = h + chart_height + padding * 2
            image_pos = (0, 0)
            chart_pos = (padding, h + padding)
            chart_size = (w - padding * 2, chart_height)
        
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # 放置原图
        visualization[image_pos[1]:image_pos[1]+h, image_pos[0]:image_pos[0]+w] = image
        
        # 绘制评分图表
        chart_area = visualization[
            chart_pos[1]:chart_pos[1]+chart_size[1], 
            chart_pos[0]:chart_pos[0]+chart_size[0]
        ]
        
        # 绘制雷达图/条形图
        if chart_size[0] > chart_size[1]:  # 横向图表，用条形图
            self._draw_bar_chart(chart_area, scores)
        else:  # 纵向图表，用雷达图
            self._draw_radar_chart(chart_area, scores)
        
        # 添加综合评分
        cv2.putText(
            visualization,
            f"综合评分: {combined_score:.1f}",
            (padding, padding - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )
        
        return visualization
    
    def _draw_bar_chart(self, chart_area, scores):
        """在给定区域绘制条形图"""
        h, w = chart_area.shape[:2]
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        
        if not valid_scores:
            cv2.putText(
                chart_area,
                "No scores available",
                (10, h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
            return
        
        num_scores = len(valid_scores)
        bar_height = h // (num_scores + 1)
        bar_width = w - 150
        
        # 颜色映射
        colors = [
            (255, 0, 0),    # 红
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 蓝
            (255, 255, 0),  # 黄
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 青
        ]
        
        for i, (name, score) in enumerate(valid_scores.items()):
            y = (i + 1) * bar_height
            
            # 绘制标签
            cv2.putText(
                chart_area,
                f"{name}: {score:.1f}",
                (5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # 绘制条形
            length = int(bar_width * score / 100)
            color = colors[i % len(colors)]
            cv2.rectangle(
                chart_area,
                (0, y),
                (length, y + bar_height // 2),
                color,
                -1
            )
            
            # 绘制边框
            cv2.rectangle(
                chart_area,
                (0, y),
                (bar_width, y + bar_height // 2),
                (255, 255, 255),
                1
            )
    
    def _draw_radar_chart(self, chart_area, scores):
        """在给定区域绘制雷达图"""
        h, w = chart_area.shape[:2]
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        
        if not valid_scores:
            cv2.putText(
                chart_area,
                "No scores available",
                (w//2 - 50, h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
            return
        
        num_scores = len(valid_scores)
        if num_scores < 3:
            # 雷达图至少需要3个点，不够时改用条形图
            self._draw_bar_chart(chart_area, scores)
            return
        
        # 中心点和半径
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y) - 20
        
        # 绘制同心圆
        for r in range(10, 110, 20):
            cv2.circle(
                chart_area,
                (center_x, center_y),
                radius * r // 100,
                (100, 100, 100),
                1
            )
        
        # 绘制角度线和标签
        angle_step = 2 * np.pi / num_scores
        names = list(valid_scores.keys())
        scores_list = list(valid_scores.values())
        
        points = []
        for i in range(num_scores):
            angle = i * angle_step - np.pi / 2  # 从顶部开始
            
            # 角度线
            end_x = int(center_x + radius * np.cos(angle))
            end_y = int(center_y + radius * np.sin(angle))
            cv2.line(chart_area, (center_x, center_y), (end_x, end_y), (100, 100, 100), 1)
            
            # 标签
            label_x = int(center_x + (radius + 20) * np.cos(angle))
            label_y = int(center_y + (radius + 20) * np.sin(angle))
            
            # 调整标签位置，避免超出边界
            text_size = cv2.getTextSize(names[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            if label_x < center_x:
                label_x -= text_size[0]
            if label_y < center_y:
                label_y -= text_size[1]
                
            cv2.putText(
                chart_area,
                names[i],
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # 计算评分点位置
            score_radius = radius * scores_list[i] / 100
            point_x = int(center_x + score_radius * np.cos(angle))
            point_y = int(center_y + score_radius * np.sin(angle))
            points.append((point_x, point_y))
        
        # 连接评分点形成多边形
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(chart_area, [points], True, (0, 255, 0), 2)
        
        # 填充多边形
        overlay = chart_area.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0, 64))
        cv2.addWeighted(overlay, 0.4, chart_area, 0.6, 0, chart_area)
        
        # 突出显示各个评分点
        for point in points:
            cv2.circle(chart_area, tuple(point[0]), 4, (255, 255, 255), -1)
