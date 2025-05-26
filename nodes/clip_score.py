"""
图文一致性评估模块
使用CLIP模型计算图像与文本的匹配度
"""

import torch
import numpy as np
import clip
from PIL import Image

class CLIPScoreNode:
    """使用CLIP模型评估图像和文本的匹配程度"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True}),
                "clip_model": (["ViT-B/32", "ViT-B/16", "ViT-L/14"], {"default": "ViT-B/32"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("clip_score", "annotated_image")
    FUNCTION = "evaluate"
    CATEGORY = "Comfyui-Evaluation/Text-Image"
    
    def evaluate(self, image, text, clip_model="ViT-B/32"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为PIL图像
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        
        # 加载CLIP模型
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(clip_model, device=device)
            
            # 预处理图像和文本
            image_input = preprocess(image_pil).unsqueeze(0).to(device)
            text_input = clip.tokenize([text]).to(device)
            
            # 计算特征
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)
                
                # 归一化特征
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # 计算相似度
                similarity = (100.0 * image_features @ text_features.T).item()
                
            # 在图像上添加分数标注
            import cv2
            annotated_img = image_np.copy()
            cv2.putText(
                annotated_img,
                f"CLIP Score: {similarity:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # 将结果转换回tensor
            annotated_tensor = torch.from_numpy(annotated_img.astype(np.float32) / 255.0)
            
            return (float(similarity), annotated_tensor.unsqueeze(0))
            
        except Exception as e:
            print(f"CLIP评分出错: {str(e)}")
            return (0.0, image.unsqueeze(0) if len(image.shape) == 3 else image)
