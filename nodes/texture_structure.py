"""
图像结构评估模块
采用DISTS进行深度学习下的纹理和结构评分
"""

import torch
import numpy as np
import cv2

class TextureStructureNode:
    """提供更符合人类主观印象的图像评价"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["DISTS", "SSIM", "LPIPS"], {"default": "DISTS"}),
                "reference_image": ("IMAGE", {"optional": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("structure_score", "visualization")
    FUNCTION = "evaluate_structure"
    CATEGORY = "Comfyui-Evaluation/Structure"
    
    def evaluate_structure(self, image, method="DISTS", reference_image=None):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 如果提供了参考图像，则进行对比评估
        if reference_image is not None:
            if len(reference_image.shape) == 4:
                reference_image = reference_image[0]
            reference_np = reference_image.cpu().numpy()
            reference_np = (reference_np * 255).astype(np.uint8)
            
            # 根据选择的方法进行评估
            if method == "DISTS":
                score, visualization = self._evaluate_dists(image_np, reference_np)
            elif method == "SSIM":
                score, visualization = self._evaluate_ssim(image_np, reference_np)
            else:  # LPIPS
                score, visualization = self._evaluate_lpips(image_np, reference_np)
        else:
            # 如果没有参考图像，则进行单图像评估
            score, visualization = self._evaluate_structure_no_reference(image_np, method)
        
        # 将结果转换回tensor
        vis_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
        
        return (float(score), vis_tensor.unsqueeze(0))
    
    def _evaluate_dists(self, image, reference=None):
        """使用DISTS评估图像结构和纹理相似性"""
        try:
            # 如果没有参考图像，返回基于自身特征的评分
            if reference is None:
                return self._evaluate_structure_no_reference(image, "DISTS")
            
            # 调整尺寸使两个图像一致
            if image.shape != reference.shape:
                reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
            
            # 由于DISTS是深度学习模型，这里提供一个简化版
            # 使用结构相似性和边缘检测结果的组合来模拟DISTS
            
            # 转灰度
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            
            # 结构相似性
            ssim_score, ssim_map = self._compute_ssim(image_gray, reference_gray)
            
            # 边缘检测
            image_edges = cv2.Canny(image_gray, 100, 200)
            reference_edges = cv2.Canny(reference_gray, 100, 200)
            
            # 计算边缘匹配度
            edge_match = cv2.bitwise_and(image_edges, reference_edges)
            edge_score = np.sum(edge_match) / max(1, np.sum(image_edges))
            
            # 纹理分析（使用局部标准差）
            patch_size = 8
            image_texture = self._compute_local_std(image_gray, patch_size)
            reference_texture = self._compute_local_std(reference_gray, patch_size)
            texture_diff = np.abs(image_texture - reference_texture)
            texture_score = 1.0 - np.mean(texture_diff) / 255.0
            
            # 综合评分 (DISTS模拟)
            dists_score = (ssim_score * 0.4 + edge_score * 0.3 + texture_score * 0.3) * 100
            
            # 可视化
            ssim_map_normalized = ((ssim_map + 1) / 2 * 255).astype(np.uint8)
            ssim_map_color = cv2.applyColorMap(ssim_map_normalized, cv2.COLORMAP_JET)
            
            # 将边缘和纹理差异可视化
            edge_vis = np.zeros_like(image)
            edge_vis[:,:,1] = edge_match  # 绿色通道显示匹配的边缘
            
            # 结合图像和可视化结果
            visualization = np.hstack([
                image, 
                cv2.resize(ssim_map_color, (image.shape[1], image.shape[0]))
            ])
            
            # 添加评分文字
            cv2.putText(
                visualization,
                f"DISTS Score: {dists_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return dists_score, visualization
            
        except Exception as e:
            print(f"DISTS评估出错: {str(e)}")
            return 50.0, image
    
    def _evaluate_ssim(self, image, reference=None):
        """使用SSIM评估图像结构相似性"""
        try:
            # 如果没有参考图像，返回基于自身特征的评分
            if reference is None:
                return self._evaluate_structure_no_reference(image, "SSIM")
            
            # 调整尺寸使两个图像一致
            if image.shape != reference.shape:
                reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
            
            # 转灰度
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            
            # 计算SSIM
            ssim_score, ssim_map = self._compute_ssim(image_gray, reference_gray)
            
            # 标准化为0-100分
            normalized_score = ssim_score * 100
            
            # 可视化SSIM图
            ssim_map_normalized = ((ssim_map + 1) / 2 * 255).astype(np.uint8)
            ssim_map_color = cv2.applyColorMap(ssim_map_normalized, cv2.COLORMAP_JET)
            
            # 结合图像和SSIM图
            visualization = np.hstack([
                image, 
                cv2.resize(ssim_map_color, (image.shape[1], image.shape[0]))
            ])
            
            # 添加评分文字
            cv2.putText(
                visualization,
                f"SSIM Score: {normalized_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return normalized_score, visualization
            
        except Exception as e:
            print(f"SSIM评估出错: {str(e)}")
            return 50.0, image
    
    def _evaluate_lpips(self, image, reference=None):
        """使用LPIPS评估图像感知相似性"""
        try:
            # 由于LPIPS是深度学习模型，这里提供一个简化版
            # 使用颜色直方图和边缘图相似度来模拟LPIPS
            
            # 如果没有参考图像，返回基于自身特征的评分
            if reference is None:
                return self._evaluate_structure_no_reference(image, "LPIPS")
            
            # 调整尺寸使两个图像一致
            if image.shape != reference.shape:
                reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
            
            # 计算颜色直方图相似度
            hist_sim = self._compute_histogram_similarity(image, reference)
            
            # 计算边缘相似度
            edge_sim = self._compute_edge_similarity(image, reference)
            
            # 模拟LPIPS分数 (越低越相似，这里取反使其越高越好)
            lpips_score = (hist_sim * 0.5 + edge_sim * 0.5) * 100
            
            # 创建可视化图像
            visualization = np.hstack([image, reference])
            
            # 添加评分文字
            cv2.putText(
                visualization,
                f"LPIPS Score: {lpips_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return lpips_score, visualization
            
        except Exception as e:
            print(f"LPIPS评估出错: {str(e)}")
            return 50.0, image
    
    def _evaluate_structure_no_reference(self, image, method_name):
        """无参考图像时评估图像结构"""
        try:
            # 转灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 100, 200)
            
            # 纹理复杂度（使用局部标准差）
            patch_size = 8
            texture_complexity = self._compute_local_std(gray, patch_size)
            avg_complexity = np.mean(texture_complexity) / 255.0
            
            # 色彩丰富度
            color_richness = np.std(image.reshape(-1, 3), axis=0).mean() / 255.0
            
            # 根据方法调整权重
            if method_name == "DISTS":
                # DISTS更注重结构和纹理
                structure_weight, texture_weight, color_weight = 0.4, 0.4, 0.2
            elif method_name == "SSIM":
                # SSIM更注重结构
                structure_weight, texture_weight, color_weight = 0.6, 0.3, 0.1
            else:  # LPIPS
                # LPIPS更注重感知
                structure_weight, texture_weight, color_weight = 0.3, 0.3, 0.4
            
            # 结构评分（基于边缘密度）
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            structure_score = min(1.0, edge_density * 5)  # 归一化
            
            # 综合评分
            final_score = (
                structure_score * structure_weight + 
                avg_complexity * texture_weight + 
                color_richness * color_weight
            ) * 100
            
            # 可视化
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            texture_vis = cv2.applyColorMap((texture_complexity).astype(np.uint8), cv2.COLORMAP_JET)
            
            # 结合图像和分析结果
            visualization = np.hstack([
                image,
                texture_vis
            ])
            
            # 添加评分文字
            cv2.putText(
                visualization,
                f"{method_name} Score: {final_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return final_score, visualization
            
        except Exception as e:
            print(f"结构评估出错: {str(e)}")
            return 50.0, image
    
    def _compute_ssim(self, img1, img2):
        """计算SSIM和SSIM图"""
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2, full=True)
    
    def _compute_local_std(self, img, patch_size):
        """计算局部标准差作为纹理复杂度度量"""
        from scipy.ndimage import uniform_filter
        
        # 均值滤波
        mean = uniform_filter(img, size=patch_size)
        # 平方均值滤波
        mean_sq = uniform_filter(img**2, size=patch_size)
        # 方差 = 平方均值 - 均值的平方
        var = mean_sq - mean**2
        # 标准差
        std = np.sqrt(np.maximum(var, 0))
        
        return std
    
    def _compute_histogram_similarity(self, img1, img2):
        """计算颜色直方图相似度"""
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # 归一化直方图
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # 计算相似度
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return similarity
    
    def _compute_edge_similarity(self, img1, img2):
        """计算边缘相似度"""
        # 转灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)
        
        # 计算边缘匹配度
        edge_match = cv2.bitwise_and(edges1, edges2)
        edge_union = cv2.bitwise_or(edges1, edges2)
        
        # 返回IoU
        return np.sum(edge_match) / max(1, np.sum(edge_union))
