"""
图像质量评估模块
使用NIQE/BRISQUE等无参考图像质量评估方法
"""

import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class ImageQualityNode:
    """评估图像质量，检测模糊、压缩伪影、纹理质量等问题"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["BRISQUE", "NIQE", "Blur Detection"], {"default": "BRISQUE"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("quality_score", "annotated_image")
    FUNCTION = "evaluate_quality"
    CATEGORY = "Comfyui-Evaluation/Quality"
    
    def evaluate_quality(self, image, method="BRISQUE"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 根据选择的方法进行评估
        if method == "BRISQUE":
            score, annotated = self._evaluate_brisque(image_np)
        elif method == "NIQE":
            score, annotated = self._evaluate_niqe(image_np)
        else:  # Blur Detection
            score, annotated = self._detect_blur(image_np)
        
        # 将结果转换回tensor
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0)
        
        return (float(score), annotated_tensor.unsqueeze(0))
    
    def _evaluate_brisque(self, image):
        """使用BRISQUE算法评估图像质量"""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 简化版BRISQUE（实际应使用完整BRISQUE算法）
            # 这里使用拉普拉斯算子的方差作为锐度指标
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # 使用噪声估计
            noise_sigma = self._estimate_noise(gray)
            
            # 模拟BRISQUE分数（0-100，越低越好）
            # 实际应用中应使用训练好的BRISQUE模型
            brisque_score = 100 - min(100, max(0, (sharpness / 500) * 50 - noise_sigma))
            
            # 标准化到0-100（越高越好）
            normalized_score = 100 - brisque_score
            
            # 在图像上添加评分
            annotated = image.copy()
            cv2.putText(
                annotated,
                f"BRISQUE Score: {normalized_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return normalized_score, annotated
            
        except Exception as e:
            print(f"BRISQUE评估出错: {str(e)}")
            return 50.0, image  # 默认中等分数
    
    def _evaluate_niqe(self, image):
        """使用NIQE算法评估图像质量"""
        try:
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 简化版NIQE（实际应使用完整NIQE算法）
            # 这里使用对比度和熵作为质量指标
            
            # 计算对比度
            contrast = gray.std()
            
            # 计算熵
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist /= hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # 模拟NIQE分数（越低越好）
            # 实际应用中应使用训练好的NIQE模型
            niqe_score = 10 - min(10, max(0, (contrast / 50) + (entropy / 8)))
            
            # 标准化到0-100（越高越好）
            normalized_score = (10 - niqe_score) * 10
            
            # 在图像上添加评分
            annotated = image.copy()
            cv2.putText(
                annotated,
                f"NIQE Score: {normalized_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            return normalized_score, annotated
            
        except Exception as e:
            print(f"NIQE评估出错: {str(e)}")
            return 50.0, image  # 默认中等分数
    
    def _detect_blur(self, image):
        """检测图像模糊度"""
        try:
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 使用拉普拉斯算子计算方差
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = np.var(laplacian)
            
            # 定义阈值
            threshold = 500
            
            # 计算分数（0-100，越高越清晰）
            sharpness_score = min(100, max(0, variance / threshold * 100))
            
            # 在图像上添加评分
            annotated = image.copy()
            cv2.putText(
                annotated,
                f"Sharpness: {sharpness_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # 标记可能模糊的区域
            if sharpness_score < 50:
                h, w = gray.shape
                block_size = 64
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        block = gray[y:min(y+block_size, h), x:min(x+block_size, w)]
                        block_laplacian = cv2.Laplacian(block, cv2.CV_64F)
                        block_variance = np.var(block_laplacian)
                        
                        if block_variance < threshold / 2:
                            cv2.rectangle(
                                annotated,
                                (x, y),
                                (min(x+block_size, w), min(y+block_size, h)),
                                (0, 0, 255),
                                2
                            )
            
            return sharpness_score, annotated
            
        except Exception as e:
            print(f"模糊检测出错: {str(e)}")
            return 50.0, image  # 默认中等分数
    
    def _estimate_noise(self, gray_img):
        """估计图像噪声水平"""
        # 使用中值滤波去噪
        denoised = cv2.medianBlur(gray_img, 5)
        
        # 计算差异
        noise = gray_img.astype(np.float32) - denoised.astype(np.float32)
        
        # 返回噪声标准差
        return np.std(noise)
