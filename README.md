# Comfyui-Evaluation

ComfyUI图像评估插件：对AI生成图像进行全面的主客观评价

## 项目简介

Comfyui-Evaluation是一个专为ComfyUI设计的插件，提供多种评估节点，用于全面评价AI生成的图像质量、真实性和符合度，特别针对人物模特与服装图像的评估。

## 功能模块

### 图文一致性模块
- 使用CLIPScore进行图文匹配度评分
- 评估生成图像与文本提示的一致性

### 姿态准确性模块
- 基于OpenPose/MediaPipe Pose的关键点检测
- 评估人体姿态的自然度和肢体比例合理性

### 图像质量评估模块
- 采用NIQE/BRISQUE等无参考图像质量评估方法
- 检测模糊、压缩伪影、纹理质量等问题

### 服装结构分析模块
- 基于DeepFashion2+MaskRCNN的服装细节解析
- 评估服装结构、口袋、拉链等细节的完整性和合理性

### 人体检测评估模块
- 使用Detectron2/Mediapipe进行人体检测
- 判断人体完整性和着装逻辑合理性

### 图像结构评估模块
- 采用DISTS进行深度学习下的纹理和结构评分
- 提供更符合人类主观印象的图像评价

### 人类感知评分(HPS)模块
- 模拟人类对AI生成图像的主观评价
- 支持多种评估维度：综合评价、人像质量、服装细节、艺术感
- 可视化关键特征评分，提供直观反馈

### 综合评分系统
- 集成各模块分数，提供整体评价
- 可视化评分结果和改进建议

## 安装指南

```bash
git clone https://github.com/yourusername/Comfyui-Evaluation.git
cd ComfyUI/custom_nodes/
ln -s /path/to/Comfyui-Evaluation ./
```                     

## 使用方法

将评估节点添加到ComfyUI工作流程中，连接图像输入，获取多维度评分结果。

## 依赖项

- Python 3.8+
- PyTorch
- OpenCV
- CLIP
- MediaPipe
- Detectron2
- 其他依赖将在安装过程中自动处理

## 许可证

[MIT License](LICENSE) 
