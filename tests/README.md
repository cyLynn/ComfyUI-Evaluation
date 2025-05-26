# Comfyui-Evaluation 测试指南

本目录包含对 Comfyui-Evaluation 插件中各个评估模块的测试脚本。这些脚本可以单独运行，也可以通过 `run_all_tests.py` 一次性运行所有测试。

## 使用方法

### 1. 测试所有模块

要一次性测试所有评估模块，可以运行：

```bash
cd /path/to/Comfyui-Evaluation
python tests/run_all_tests.py --image /path/to/your/test/image.jpg --prompt "一个穿着漂亮衣服的模特"
```

### 2. 测试单个模块

每个模块都有对应的测试脚本，可以单独运行：

#### CLIPScore 模块测试

```bash
python tests/test_clip_score.py --image /path/to/your/test/image.jpg --prompt "一个穿着漂亮衣服的模特"
```

#### 姿态准确性模块测试

```bash
python tests/test_pose_assessment.py --image /path/to/your/test/image.jpg --method MediaPipe
```

#### 图像质量评估模块测试

```bash
python tests/test_image_quality.py --image /path/to/your/test/image.jpg --method BRISQUE
```

#### 服装结构分析模块测试

```bash
python tests/test_fashion_analysis.py --image /path/to/your/test/image.jpg --detail 中
```

#### 人体检测评估模块测试

```bash
python tests/test_human_detection.py --image /path/to/your/test/image.jpg --method Mediapipe
```

#### 图像结构评估模块测试

```bash
python tests/test_texture_structure.py --image /path/to/your/test/image.jpg --method DISTS
# 如果有参考图像
python tests/test_texture_structure.py --image /path/to/your/test/image.jpg --reference /path/to/reference.jpg --method SSIM
```

#### 人类感知评分模块测试

```bash
python tests/test_hps_evaluation.py --image /path/to/your/test/image.jpg --focus 综合评价
```

#### 综合评分系统测试

```bash
python tests/test_combined_evaluation.py --image /path/to/your/test/image.jpg --prompt "一个穿着漂亮衣服的模特"
```

## 参数说明

每个测试脚本都接受不同的参数，可以通过 `--help` 选项查看详细用法：

```bash
python tests/test_clip_score.py --help
```

## 注意事项

1. 如果不提供图像路径，测试脚本会创建一个随机噪声图像用于测试
2. 测试结果会保存在 `tests/outputs/` 目录下
3. 某些模块（如MediaPipe、CLIP）可能需要先安装相应的依赖项
4. 首次运行可能需要下载模型权重，这可能需要一些时间

## 依赖关系

运行这些测试需要安装项目所需的依赖项。你可以通过运行项目根目录下的安装脚本来完成：

```bash
python install_requirements.py
```
