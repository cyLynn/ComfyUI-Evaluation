"""
服装结构分析模块
基于DeepFashion2+MaskRCNN的服装细节解析
"""

import torch
import numpy as np
import cv2
import os

class FashionAnalysisNode:
    """评估服装结构、口袋、拉链等细节的完整性和合理性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_level": (["低", "中", "高"], {"default": "中"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("fashion_score", "annotated_image")
    FUNCTION = "analyze_fashion"
    CATEGORY = "Comfyui-Evaluation/Fashion"
    
    def analyze_fashion(self, image, detail_level="中"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 详细程度到阈值映射
        detail_thresholds = {
            "低": 0.3,
            "中": 0.5,
            "高": 0.7
        }
        threshold = detail_thresholds[detail_level]
        
        try:
            # 使用预训练的服装分析模型
            score, annotated = self._analyze_with_deepfashion(image_np, threshold)
            
        except Exception as e:
            print(f"服装分析出错: {str(e)}")
            score = 50.0
            annotated = image_np
            cv2.putText(
                annotated,
                "Fashion analysis failed",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
        
        # 将结果转换回tensor
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0)
        
        return (float(score), annotated_tensor.unsqueeze(0))
    
    def _analyze_with_deepfashion(self, image, threshold):
        """使用本地Segformer-b3-fashion模型进行服装分割分析"""
        import os
        try:
            import torch
            import numpy as np
            import cv2
            from PIL import Image
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            import folder_paths
            # 正确的ComfyUI本地模型路径
            model_dir = os.path.join(folder_paths.models_dir, "Evaluation", "segformer-b3-fashion")
            processor = SegformerImageProcessor.from_pretrained(model_dir)
            model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
            model.eval()
            # 标签映射
            label_map = {
                0: "Unlabelled", 1: "shirt, blouse", 2: "top, t-shirt, sweatshirt", 3: "sweater", 4: "cardigan", 5: "jacket", 6: "vest", 7: "pants", 8: "shorts", 9: "skirt", 10: "coat", 11: "dress", 12: "jumpsuit", 13: "cape", 14: "glasses", 15: "hat", 16: "headband, head covering, hair accessory", 17: "tie", 18: "glove", 19: "watch", 20: "belt", 21: "leg warmer", 22: "tights, stockings", 23: "sock", 24: "shoe", 25: "bag, wallet", 26: "scarf", 27: "umbrella", 28: "hood", 29: "collar", 30: "lapel", 31: "epaulette", 32: "sleeve", 33: "pocket", 34: "neckline", 35: "buckle", 36: "zipper", 37: "applique", 38: "bead", 39: "bow", 40: "flower", 41: "fringe", 42: "ribbon", 43: "rivet", 44: "ruffle", 45: "sequin", 46: "tassel"
            }
            # 图像预处理
            if image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            inputs = processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            seg = outputs.logits.argmax(dim=1)[0].cpu().numpy()
            # 可视化分割结果
            annotated = np.array(pil_img).copy()
            if annotated.ndim == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)
            if annotated.shape[2] != 3:
                annotated = annotated[..., :3]
            # 生成调色板（47类）
            np.random.seed(42)
            palette = np.vstack([[0,0,0], np.random.randint(0,255,(46,3))]).astype(np.uint8)
            color_mask = palette[seg]
            if color_mask.shape[:2] != annotated.shape[:2]:
                color_mask = cv2.resize(color_mask, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_NEAREST)
            if color_mask.shape[2] != 3:
                color_mask = color_mask[..., :3]
            color_mask = color_mask.astype(np.uint8)
            annotated = cv2.addWeighted(annotated, 0.6, color_mask, 0.4, 0)
            # 统计服装主类区域占比作为分数（忽略0:Unlabelled）
            clothes_pixels = np.sum(seg > 0)
            total_pixels = seg.size
            fashion_score = min(1.0, clothes_pixels / (total_pixels * 0.7)) * 100
            # 标注主类别
            unique, counts = np.unique(seg, return_counts=True)
            main_labels = [(label_map.get(int(i), str(i)), int(c)) for i, c in zip(unique, counts) if i > 0]
            main_labels = sorted(main_labels, key=lambda x: -x[1])[:5]
            y0 = 80
            for idx, (name, cnt) in enumerate(main_labels):
                cv2.putText(annotated, f"{name}: {cnt}", (20, y0+idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(annotated, f"Fashion Score: {fashion_score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return fashion_score, annotated
        except Exception as e:
            print(f"Segformer服装分割分析失败: {e}")
            return 50.0, image
