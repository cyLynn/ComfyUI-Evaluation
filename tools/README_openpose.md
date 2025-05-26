# OpenPose模型管理工具使用说明

这个工具用于下载、检查和管理OpenPose模型到ComfyUI的models目录，以便在Comfyui-Evaluation插件的姿态评估中使用。

## 功能

1. 下载预定义的OpenPose模型
2. 列出已下载的OpenPose模型
3. 检查模型文件完整性
4. 配置和测试代理设置

## 使用方法

### 下载OpenPose模型

下载所有预定义的OpenPose模型：

```bash
python tools/manage_openpose_models.py download
```

下载特定模型：

```bash
python tools/manage_openpose_models.py download --model body_25
```

可用模型：
- `body_25`: BODY_25模型 - 25个关键点的人体姿势模型
- `coco`: COCO模型 - 18个关键点的人体姿势模型
- `hand`: 手部姿势模型
- `all`: 下载所有模型

使用代理下载：

```bash
python tools/manage_openpose_models.py download --proxy yes
```

代理选项：
- `yes`: 强制使用代理
- `no`: 强制不使用代理
- `auto`: 使用配置文件中的设置(默认)

### 列出已下载的OpenPose模型

```bash
python tools/manage_openpose_models.py list
```

这将显示已下载到ComfyUI models/openpose目录的所有模型。

### 检查模型文件完整性

检查特定模型文件是否完整：

```bash
python tools/manage_openpose_models.py check --model body_25
```

检查所有模型：

```bash
python tools/manage_openpose_models.py check
```

### 代理设置

测试代理连接：

```bash
python tools/manage_openpose_models.py proxy --test
```

配置代理设置：

```bash
python tools/manage_openpose_models.py proxy --config
```

## 代理配置

代理配置文件保存在`config/proxy.json`，格式如下：

```json
{
  "enabled": false,
  "http": "http://127.0.0.1:7890",
  "https": "http://127.0.0.1:7890",
  "socks": "socks5://127.0.0.1:7890",
  "no_proxy": "localhost,127.0.0.1"
}
```

- `enabled`: 是否启用代理
- `http`: HTTP代理地址
- `https`: HTTPS代理地址
- `socks`: SOCKS代理地址
- `no_proxy`: 不使用代理的地址

## 注意事项

1. 初次使用OpenPose模型进行姿态评估时，如果模型不存在，将自动尝试下载
2. 自动下载可能受网络环境限制，如遇下载问题可使用此工具手动下载
3. OpenPose模型文件较大，下载可能需要一定时间
4. 在网络受限环境下，建议配置代理以提高下载成功率

## OpenPose模型手动下载说明

**本插件不再自动下载OpenPose模型。请用户手动下载以下模型文件，并放置到指定目录：**

### 1. 下载地址

- BODY_25模型：
  - [pose_iter_584000.caffemodel (BODY_25)](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel)
  - [pose_deploy.prototxt (BODY_25)](https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt)

- COCO模型：
  - [pose_iter_440000.caffemodel (COCO)](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel)
  - [pose_deploy_linevec.prototxt (COCO)](https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt)

- 手部模型：
  - [pose_iter_102000.caffemodel (Hand)](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel)
  - [pose_deploy.prototxt (Hand)](https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt)

### 2. 放置目录

请将上述文件分别放入如下目录（如无请手动创建文件夹）：

```
ComfyUI根目录/models/openpose/body_25/
    pose_iter_584000.caffemodel
    pose_deploy.prototxt
ComfyUI根目录/models/openpose/coco/
    pose_iter_440000.caffemodel
    pose_deploy_linevec.prototxt
ComfyUI根目录/models/openpose/hand/
    pose_iter_102000.caffemodel
    pose_deploy.prototxt
```

- 例如：`/your_comfyui_path/models/openpose/body_25/pose_iter_584000.caffemodel`

### 3. 注意事项

- 文件名必须与上面完全一致。
- 下载后请确保文件大小正常（caffemodel通常几十MB）。
- 没有模型文件时，节点会在界面上提示缺失。
- 推荐使用科学上网工具下载。

---

原有的自动下载、代理配置等功能已废弃。请务必手动下载并放置模型文件，否则OpenPose节点无法正常工作。
