"""
图文一致性评估模块
使用Transformers CLIP模型计算图像与文本的匹配度
支持从ComfyUI models目录读取本地模型
"""

import torch
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import folder_paths

class CLIPScoreNode:
    """使用CLIP模型评估图像和文本的匹配程度"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取本地CLIP模型列表
        clip_models_path = os.path.join(folder_paths.models_dir, "clip")
        os.makedirs(clip_models_path, exist_ok=True)
        local_models = cls.list_local_clip_models()
        
        # 如果没有本地模型，则提供预训练模型选项
        if not local_models:
            local_models = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True}),
                "clip_model": (local_models, {"default": local_models[0] if local_models else "openai/clip-vit-base-patch32"}),
                "use_local_model": (["是", "否"], {"default": "是"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("clip_score", "annotated_image")
    FUNCTION = "evaluate"
    CATEGORY = "Comfyui-Evaluation/Text-Image"
    
    @classmethod
    def list_local_clip_models(cls):
        """列出ComfyUI models/clip目录下的所有模型"""
        models = []
        clip_dir = os.path.join(folder_paths.models_dir, "clip")
        
        if os.path.exists(clip_dir):
            for model_name in os.listdir(clip_dir):
                model_path = os.path.join(clip_dir, model_name)
                if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                    models.append(model_name)
        
        # 如果没有本地模型，添加一些常用的预训练模型
        if not models:
            models = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"]
            
        return models
        
    @classmethod
    def download_clip_model(cls, model_name):
        """下载CLIP模型到ComfyUI models/clip目录"""
        from transformers import CLIPModel, CLIPProcessor
        
        clip_dir = os.path.join(folder_paths.models_dir, "clip")
        os.makedirs(clip_dir, exist_ok=True)
        
        model_path = os.path.join(clip_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"下载CLIP模型 {model_name} 到 {model_path}")
            
            # 下载模型和处理器
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            
            # 保存到本地
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            
            print(f"CLIP模型 {model_name} 已保存到 {model_path}")
            return True
        else:
            print(f"CLIP模型 {model_name} 已存在于 {model_path}")
            return False
    
    def evaluate(self, image, text, clip_model, use_local_model="是"):
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
            
            # 确定模型路径
            if use_local_model == "是":
                # 尝试从本地加载模型
                model_path = os.path.join(folder_paths.models_dir, "clip", clip_model)
                
                # 如果本地路径不存在，则可能是Hugging Face模型ID
                if not os.path.exists(model_path):
                    model_path = clip_model
            else:
                # 直接使用Hugging Face模型ID
                model_path = clip_model
            
            # 加载模型和处理器
            model = CLIPModel.from_pretrained(model_path).to(device)
            processor = CLIPProcessor.from_pretrained(model_path)
            
            # 预处理图像和文本
            inputs = processor(
                text=[text],
                images=image_pil,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # 计算特征
            with torch.no_grad():
                outputs = model(**inputs)
                # 获取图像和文本特征
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # 归一化特征
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # 计算相似度 (0-100分)
                similarity = (100.0 * (image_features @ text_features.T)).item()
                
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
            import traceback
            traceback.print_exc()
            return (0.0, image.unsqueeze(0) if len(image.shape) == 3 else image)
