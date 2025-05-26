# CLIP模型管理工具使用说明

这个工具用于下载和管理CLIP模型到ComfyUI的models目录，以便在Comfyui-Evaluation插件中使用本地CLIP模型。

## 功能

1. 下载预定义的常用CLIP模型
2. 列出已下载的CLIP模型
3. 下载指定的CLIP模型

## 使用方法

### 下载预定义的CLIP模型

```bash
python tools/manage_clip_models.py download
```

这将下载以下常用的CLIP模型到ComfyUI的models/clip目录：
- openai/clip-vit-base-patch32
- openai/clip-vit-base-patch16
- openai/clip-vit-large-patch14

### 列出已下载的CLIP模型

```bash
python tools/manage_clip_models.py list
```

这将显示已下载到ComfyUI models/clip目录的所有模型。

### 下载指定的CLIP模型

```bash
python tools/manage_clip_models.py download --model MODEL_NAME
```

例如：
```bash
python tools/manage_clip_models.py download --model openai/clip-vit-large-patch14-336
```

## 注意事项

1. 首次下载模型时，会从Hugging Face下载模型文件，这可能需要一些时间，取决于您的网络速度。
2. 模型会被保存到ComfyUI的models/clip目录中，这样在插件中就可以直接使用这些本地模型。
3. 如果您想使用自定义的CLIP模型，只需将其放入ComfyUI的models/clip目录中，插件会自动检测并使用它们。
