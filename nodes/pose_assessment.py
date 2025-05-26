"""
姿态准确性评估模块
使用OpenPose或MediaPipe进行人体姿态评估
"""

import torch
import numpy as np
import cv2
import os
import json
import folder_paths
import urllib.request
import shutil
import threading
import time
import socket
from pathlib import Path
from urllib.parse import urlparse

# 代理设置
PROXY_CONFIG = {
    "enabled": False,
    "http": "",
    "https": "",
    "socks": "",
    "no_proxy": "localhost,127.0.0.1"
}

# 模型URLs和文件信息
OPENPOSE_MODELS = {
    "body_25": {
        "description": "BODY_25模型 - 25个关键点的人体姿势模型",
        "files": {
            "pose_iter_584000.caffemodel": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
            "pose_deploy.prototxt": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt"
        }
    },
    "coco": {
        "description": "COCO模型 - 18个关键点的人体姿势模型",
        "files": {
            "pose_iter_440000.caffemodel": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
            "pose_deploy_linevec.prototxt": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
        }
    },
    "hand": {
        "description": "手部姿势模型",
        "files": {
            "pose_iter_102000.caffemodel": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel",
            "pose_deploy.prototxt": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt"
        }
    }
}

# 全局变量，用于跟踪模型下载状态
downloading_models = {}
download_progress = {}

def ensure_directory(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def download_file(url, destination, use_proxy=None, max_retries=3, timeout=30):
    """从URL下载文件到指定目标
    
    Args:
        url: 下载的URL
        destination: 目标文件路径
        use_proxy: 是否使用代理，None时使用全局配置，True/False强制开启/关闭
        max_retries: 最大重试次数
        timeout: 连接超时时间(秒)
    """
    retry_count = 0
    temp_file = f"{destination}.download"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                print(f"第 {retry_count + 1} 次尝试下载: {url}")
            else:
                print(f"开始下载: {url}")
                print(f"保存到: {destination}")
            
            # 处理代理设置
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
            
            # 确定是否使用代理
            use_proxy_flag = PROXY_CONFIG["enabled"] if use_proxy is None else use_proxy
            
            if use_proxy_flag:
                print(f"使用代理下载: {url}")
                # 根据URL协议选择合适的代理
                parsed_url = urlparse(url)
                if parsed_url.scheme == 'https' and PROXY_CONFIG['https']:
                    proxy = urllib.request.ProxyHandler({
                        'https': PROXY_CONFIG['https']
                    })
                    print(f"使用HTTPS代理: {PROXY_CONFIG['https']}")
                elif parsed_url.scheme == 'http' and PROXY_CONFIG['http']:
                    proxy = urllib.request.ProxyHandler({
                        'http': PROXY_CONFIG['http']
                    })
                    print(f"使用HTTP代理: {PROXY_CONFIG['http']}")
                else:
                    # 默认使用http代理
                    proxy = urllib.request.ProxyHandler({
                        'http': PROXY_CONFIG['http'],
                        'https': PROXY_CONFIG['https']
                    }) if PROXY_CONFIG['http'] or PROXY_CONFIG['https'] else None
                    print(f"使用默认代理设置")
                    
                if proxy:
                    opener = urllib.request.build_opener(proxy)
                    urllib.request.install_opener(opener)
            else:
                print(f"不使用代理下载: {url}")
            
            # 开始下载
            try:
                # 设置超时时间
                with urllib.request.urlopen(request, timeout=timeout) as response, open(temp_file, 'wb') as out_file:
                    file_size = int(response.headers.get('Content-Length', 0))
                    block_size = 8192
                    downloaded = 0
                    start_time = time.time()
                    last_update_time = start_time
                    
                    while True:
                        try:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            
                            downloaded += len(buffer)
                            out_file.write(buffer)
                            
                            # 更新下载进度
                            if file_size > 0:
                                model_name = os.path.basename(os.path.dirname(destination))
                                file_name = os.path.basename(destination)
                                key = f"{model_name}/{file_name}"
                                download_progress[key] = (downloaded / file_size) * 100
                                
                                # 显示下载速度和进度
                                current_time = time.time()
                                if current_time - last_update_time > 0.5:  # 每0.5秒更新一次
                                    elapsed = current_time - start_time
                                    speed = downloaded / elapsed if elapsed > 0 else 0
                                    print(f"\r下载进度: {download_progress[key]:.1f}% - {downloaded/(1024*1024):.1f}/{file_size/(1024*1024):.1f}MB - {speed/(1024*1024):.2f}MB/s", end="")
                                    last_update_time = current_time
                        except socket.timeout:
                            print("\n读取超时，重试...")
                            continue
                    
                    print("\n下载完成!")
            except urllib.error.URLError as e:
                if isinstance(e.reason, socket.timeout):
                    print(f"连接超时: {e}")
                    retry_count += 1
                    continue
                raise
            
            # 验证下载完成后再重命名文件
            if os.path.exists(temp_file):
                if os.path.getsize(temp_file) > 0:
                    # 如果文件大小已知，验证下载是否完整
                    if file_size > 0 and os.path.getsize(temp_file) < file_size:
                        print(f"警告: 下载不完整 ({os.path.getsize(temp_file)}/{file_size} 字节)")
                        retry_count += 1
                        continue
                    
                    # 如果目标文件已存在先删除它
                    if os.path.exists(destination):
                        os.remove(destination)
                    os.rename(temp_file, destination)
                    print(f"下载完成并保存到: {destination}")
                    
                    # 更新进度为100%
                    model_name = os.path.basename(os.path.dirname(destination))
                    file_name = os.path.basename(destination)
                    key = f"{model_name}/{file_name}"
                    download_progress[key] = 100
                    
                    return True
                else:
                    # 如果临时文件大小为0，说明下载失败
                    os.remove(temp_file)
                    print(f"下载失败: {destination} 文件大小为0")
                    retry_count += 1
                    continue
            else:
                print(f"下载失败: 临时文件不存在")
                retry_count += 1
                continue
                
        except Exception as e:
            print(f"下载出错: {str(e)}")
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            retry_count += 1
            
            # 如果网络错误，可能需要切换代理
            if "Connection" in str(e) or "Timeout" in str(e) or "timeout" in str(e).lower():
                print("检测到网络连接问题，可能需要调整代理设置")
    
    # 更新下载进度为失败状态(-1)
    model_name = os.path.basename(os.path.dirname(destination))
    file_name = os.path.basename(destination)
    key = f"{model_name}/{file_name}"
    download_progress[key] = -1
    
    print(f"下载失败: 已达到最大重试次数 ({max_retries})")
    return False

def download_model_async(model_name, use_proxy=None):
    """异步下载OpenPose模型
    
    Args:
        model_name: 模型名称
        use_proxy: 是否使用代理，None表示使用全局设置
    """
    if model_name not in OPENPOSE_MODELS:
        print(f"错误: 未知的模型 '{model_name}'")
        downloading_models[model_name] = False
        return False
    
    # 设置下载状态
    downloading_models[model_name] = True
    print(f"开始下载OpenPose模型: {model_name}")
    
    # 确定模型目录
    openpose_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
    ensure_directory(openpose_dir)
    
    # 下载模型文件
    model_info = OPENPOSE_MODELS[model_name]
    success = True
    all_files_exist = True
    
    # 下载每个模型文件
    for file_name, url in model_info["files"].items():
        file_path = os.path.join(openpose_dir, file_name)
        key = f"{model_name}/{file_name}"
        download_progress[key] = 0
        
        if not os.path.exists(file_path):
            all_files_exist = False
            file_success = download_file(url, file_path, use_proxy=use_proxy)
            success = success and file_success
            
            # 如果下载失败，更新进度为-1表示错误
            if not file_success:
                download_progress[key] = -1
        else:
            # 检查文件大小是否正常
            if os.path.getsize(file_path) > 0:
                download_progress[key] = 100
                print(f"文件已存在: {file_path}")
            else:
                # 如果文件存在但大小为0，尝试重新下载
                print(f"文件大小为0，重新下载: {file_path}")
                all_files_exist = False
                os.remove(file_path)
                file_success = download_file(url, file_path, use_proxy=use_proxy)
                success = success and file_success
                if not file_success:
                    download_progress[key] = -1
    
    # 只有在所有文件下载成功时才创建模型信息文件
    if success:
        # 创建模型信息文件
        info_path = os.path.join(openpose_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": model_name,
                "description": model_info["description"],
                "files": list(model_info["files"].keys()),
                "status": "complete"
            }, f, ensure_ascii=False, indent=2)
        print(f"OpenPose模型下载完成: {model_name}")
    else:
        # 创建失败状态的信息文件
        info_path = os.path.join(openpose_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": model_name,
                "description": model_info["description"],
                "files": list(model_info["files"].keys()),
                "status": "incomplete",
                "error": "下载失败，请尝试重新下载"
            }, f, ensure_ascii=False, indent=2)
        print(f"OpenPose模型 {model_name} 下载失败，部分文件可能不完整")
    
    # 更新下载状态
    downloading_models[model_name] = False
    
    return success

def download_model(model_name, use_proxy=None):
    """启动异步下载线程
    
    Args:
        model_name: 要下载的模型名称
        use_proxy: 是否使用代理，None表示使用全局配置
    """
    if model_name in downloading_models and downloading_models[model_name]:
        print(f"模型 {model_name} 已经在下载中...")
        return
    
    # 创建新线程进行下载
    thread = threading.Thread(target=download_model_async, args=(model_name, use_proxy))
    thread.daemon = True
    thread.start()

def is_model_ready(model_name):
    """检查模型是否已准备就绪"""
    # 检查是否在下载中
    if model_name in downloading_models and downloading_models[model_name]:
        return False
    
    # 检查模型文件是否存在
    model_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
    if not os.path.exists(model_dir):
        return False
    
    # 检查模型信息文件
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                # 检查模型状态
                if info.get("status") == "incomplete":
                    return False
        except:
            # 如果无法读取信息文件，则进行常规检查
            pass
    
    # 若模型不在OPENPOSE_MODELS中，无法验证文件
    if model_name not in OPENPOSE_MODELS:
        # 简单检查是否有.caffemodel和.prototxt文件
        has_caffemodel = False
        has_prototxt = False
        for file in os.listdir(model_dir):
            if file.endswith('.caffemodel'):
                has_caffemodel = True
                # 检查文件大小
                if os.path.getsize(os.path.join(model_dir, file)) == 0:
                    return False
            elif file.endswith('.prototxt'):
                has_prototxt = True
                # 检查文件大小
                if os.path.getsize(os.path.join(model_dir, file)) == 0:
                    return False
        
        return has_caffemodel and has_prototxt
    
    # 检查所有必要文件
    for file_name in OPENPOSE_MODELS[model_name]["files"]:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            return False
        # 检查文件大小
        if os.path.getsize(file_path) == 0:
            return False
        
        # 对于caffemodel文件，检查文件大小是否合理
        if file_name.endswith('.caffemodel'):
            # caffemodel文件通常较大，至少应该有几MB
            min_size = 5 * 1024 * 1024  # 5MB
            if os.path.getsize(file_path) < min_size:
                print(f"警告: {file_path} 文件大小异常 ({os.path.getsize(file_path) / 1024 / 1024:.2f}MB < 5MB)")
                return False
    
    return True

def get_model_download_progress(model_name):
    """获取模型下载进度，如果有错误返回负值"""
    if model_name not in OPENPOSE_MODELS:
        return 0
    
    # 计算总进度
    total_progress = 0
    file_count = 0
    has_error = False
    
    for file_name in OPENPOSE_MODELS[model_name]["files"]:
        key = f"{model_name}/{file_name}"
        if key in download_progress:
            # 检查是否有错误（值为-1）
            if download_progress[key] == -1:
                has_error = True
            else:
                total_progress += download_progress[key]
                file_count += 1
    
    # 如果有任何文件下载失败，返回-1表示错误
    if has_error:
        return -1
    
    if file_count == 0:
        return 0
    
    return total_progress / file_count

# 加载代理设置
def load_proxy_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "proxy.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for key, value in config.items():
                    if key in PROXY_CONFIG:
                        PROXY_CONFIG[key] = value
            print(f"已加载代理配置: {'启用' if PROXY_CONFIG['enabled'] else '禁用'}")
            if PROXY_CONFIG['enabled']:
                if PROXY_CONFIG['http']:
                    print(f"HTTP代理: {PROXY_CONFIG['http']}")
                if PROXY_CONFIG['https']:
                    print(f"HTTPS代理: {PROXY_CONFIG['https']}")
                if PROXY_CONFIG['socks']:
                    print(f"SOCKS代理: {PROXY_CONFIG['socks']}")
        except Exception as e:
            print(f"加载代理配置失败: {str(e)}")

# 创建默认代理配置（如果不存在）
def create_default_proxy_config():
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    config_path = os.path.join(config_dir, "proxy.json")
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "enabled": False,
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
                "socks": "socks5://127.0.0.1:7890",
                "no_proxy": "localhost,127.0.0.1"
            }, f, ensure_ascii=False, indent=2)
        print(f"已创建默认代理配置: {config_path}")

# 初始化代理设置
create_default_proxy_config()
load_proxy_config()

class PoseAssessmentNode:
    """评估人体姿态的自然度和肢体比例合理性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取本地OpenPose模型列表，包括可下载的模型
        openpose_models = cls.list_local_openpose_models()
        if not openpose_models:
            # 添加默认可下载模型
            for model in OPENPOSE_MODELS:
                openpose_models.append(f"{model} (点击下载)")
        
        default_model = next((m for m in openpose_models if "body_25" in m and "下载" not in m), openpose_models[0])
        
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["OpenPose", "MediaPipe"], {"default": "MediaPipe"}),
                "openpose_model": (openpose_models, {"default": default_model}),
            },
            "optional": {
                "use_proxy": (["全局设置", "使用代理", "不使用代理"], {"default": "全局设置"})
            }
        }
    
    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("pose_score", "pose_visualization")
    FUNCTION = "evaluate_pose"
    CATEGORY = "Comfyui-Evaluation/Pose"
    
    @classmethod
    def list_local_openpose_models(cls):
        """列出ComfyUI models/openpose目录下的所有模型"""
        models = []
        openpose_dir = os.path.join(folder_paths.models_dir, "openpose")
        if os.path.exists(openpose_dir):
            for model_name in os.listdir(openpose_dir):
                model_path = os.path.join(openpose_dir, model_name)
                if os.path.isdir(model_path):
                    has_caffe_model = any(f.endswith('.caffemodel') for f in os.listdir(model_path))
                    has_prototxt = any(f.endswith('.prototxt') for f in os.listdir(model_path))
                    if has_caffe_model and has_prototxt:
                        models.append(model_name)
        # 只显示本地存在的模型
        return models

    def _check_model_files(self, model_dir):
        """检查指定目录下是否有有效的caffemodel和prototxt文件"""
        has_caffe_model = False
        has_prototxt = False
        for file in os.listdir(model_dir):
            if file.endswith('.caffemodel') and os.path.getsize(os.path.join(model_dir, file)) > 0:
                has_caffe_model = True
            if file.endswith('.prototxt') and os.path.getsize(os.path.join(model_dir, file)) > 0:
                has_prototxt = True
        return has_caffe_model and has_prototxt

    def evaluate_pose(self, image, method="MediaPipe", openpose_model="body_25", use_proxy="全局设置"):
        # 确保只处理单张图像
        if len(image.shape) == 4:
            image = image[0]
        
        # 将图像从torch tensor转换为cv2可用格式
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        if method == "MediaPipe":
            score, visualization = self._mediapipe_assessment(image_np)
        else:  # OpenPose
            # 处理模型名称，移除可能的下载标签
            clean_model_name = openpose_model
            if "(" in openpose_model:
                clean_model_name = openpose_model.split("(")[0].strip()
            
            # 检查模型是否存在，不自动下载
            model_dir = os.path.join(folder_paths.models_dir, "openpose", clean_model_name)
            model_exists = os.path.exists(model_dir) and self._check_model_files(model_dir)
            
            if not model_exists:
                # 创建模型不存在的提示图像
                visualization = image_np.copy()
                
                # 添加半透明蒙版
                overlay = visualization.copy()
                cv2.rectangle(overlay, (0, 0), (visualization.shape[1], 180), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, visualization, 0.4, 0, visualization)
                
                # 显示模型缺失信息
                cv2.putText(
                    visualization,
                    f"OpenPose模型未找到: {clean_model_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                cv2.putText(
                    visualization,
                    "请参考README手动下载并放置模型文件",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                cv2.putText(
                    visualization,
                    f"模型路径: {model_dir}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # 转换回tensor格式
                vis_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
                return (0.0, vis_tensor.unsqueeze(0))
            
            score, visualization = self._openpose_assessment(image_np, clean_model_name)
        
        # 将结果转换回tensor
        vis_tensor = torch.from_numpy(visualization.astype(np.float32) / 255.0)
        
        return (float(score), vis_tensor.unsqueeze(0))
    
    def _mediapipe_assessment(self, image):
        """使用MediaPipe进行姿态评估"""
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测姿态
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5
            ) as pose:
                results = pose.process(image_rgb)
                
                # 创建可视化图像
                visualization = image.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        visualization,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                    # 计算姿态自然度分数
                    landmarks = results.pose_landmarks.landmark
                    
                    # 简单的姿态评分标准: 检查关键点可见性和位置合理性
                    visible_points = sum(1 for landmark in landmarks if landmark.visibility > 0.5)
                    visibility_score = visible_points / len(landmarks)
                    
                    # 检查肢体比例
                    proportion_score = self._check_body_proportions(landmarks)
                    
                    # 综合评分
                    final_score = (visibility_score * 0.4 + proportion_score * 0.6) * 100
                else:
                    final_score = 0.0
                
                # 添加评分信息
                cv2.putText(
                    visualization,
                    f"Pose Score: {final_score:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                return final_score, visualization
                
        except Exception as e:
            print(f"MediaPipe姿态评估出错: {str(e)}")
            return 0.0, image
    
    def _openpose_assessment(self, image, model_name="body_25"):
        """参考ComfyUI-OpenPose仓库，兼容body_25/coco/hand三种模型推理"""
        try:
            import cv2
            model_dir = os.path.join(folder_paths.models_dir, "openpose", model_name)
            prototxt_path = None
            model_path = None
            for file in os.listdir(model_dir):
                if file.endswith('.prototxt'):
                    prototxt_path = os.path.join(model_dir, file)
                elif file.endswith('.caffemodel'):
                    model_path = os.path.join(model_dir, file)
            if not prototxt_path or not model_path:
                raise FileNotFoundError("OpenPose模型文件不完整，请参考README手动下载！")

            # 配置参数和颜色
            if model_name == "body_25":
                num_points = 25
                pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17],[2,17],[5,16],[8,11]]
                point_colors = [
                    (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0), (0,255,0),
                    (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255), (0,0,255), (85,0,255),
                    (170,0,255), (255,0,255), (255,0,170), (255,0,85), (128,128,128), (128,0,0), (0,128,0),
                    (0,0,128), (128,128,0), (0,128,128), (128,0,128)
                ]
            elif model_name == "coco":
                num_points = 18
                pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
                point_colors = [
                    (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0), (0,255,0),
                    (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255), (0,0,255), (85,0,255),
                    (170,0,255), (255,0,255), (255,0,170), (255,0,85)
                ]
            elif model_name == "hand":
                num_points = 22
                pose_pairs = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
                point_colors = [
                    (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0), (0,255,0),
                    (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255), (0,0,255), (85,0,255),
                    (170,0,255), (255,0,255), (255,0,170), (255,0,85), (128,128,128), (128,0,0), (0,128,0), (0,0,128)
                ]
            else:
                raise ValueError(f"未知OpenPose模型: {model_name}")

            # 推理
            height, width = image.shape[:2]
            input_size = 368
            inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (input_size, input_size), (0, 0, 0), swapRB=False, crop=False)
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            net.setInput(inp)
            out = net.forward()

            points = []
            for i in range(num_points):
                prob_map = out[0, i, :, :]
                prob_map = cv2.resize(prob_map, (width, height))
                _, prob, _, point = cv2.minMaxLoc(prob_map)
                if prob > 0.1:
                    points.append((int(point[0]), int(point[1]), prob))
                else:
                    points.append(None)

            # 彩色可视化
            visualization = image.copy()
            for i, p in enumerate(points):
                if p:
                    color = point_colors[i % len(point_colors)]
                    cv2.circle(visualization, (p[0], p[1]), 6, color, -1, lineType=cv2.LINE_AA)
            for pair in pose_pairs:
                if points[pair[0]] and points[pair[1]]:
                    cv2.line(visualization, (points[pair[0]][0], points[pair[0]][1]), (points[pair[1]][0], points[pair[1]][1]), (0,255,0), 2, lineType=cv2.LINE_AA)

            # 简单分数
            valid_points = sum(1 for p in points if p is not None)
            score = valid_points / num_points * 100
            cv2.putText(visualization, f"Pose Score: {score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return score, visualization
        except Exception as e:
            import traceback
            traceback.print_exc()
            visualization = image.copy()
            cv2.putText(visualization, f"OpenPose Error: {str(e)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(visualization, "请参考README手动下载模型", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return 0.0, visualization
    
    def _check_body_proportions(self, landmarks):
        """检查人体比例的合理性"""
        # 简化版的人体比例检查
        # 人体各部位的标准比例关系
        try:
            # 获取关键点坐标
            # 头部到脚底的距离(身高)
            if landmarks[0].visibility > 0.5 and landmarks[27].visibility > 0.5:
                height = abs(landmarks[0].y - landmarks[27].y)
                
                # 躯干长度
                if landmarks[11].visibility > 0.5 and landmarks[23].visibility > 0.5:
                    torso = abs(landmarks[11].y - landmarks[23].y)
                    
                    # 腿长
                    if landmarks[23].visibility > 0.5 and landmarks[27].visibility > 0.5:
                        leg = abs(landmarks[23].y - landmarks[27].y)
                        
                        # 手臂长度
                        if landmarks[11].visibility > 0.5 and landmarks[15].visibility > 0.5:
                            arm = abs(landmarks[11].y - landmarks[15].y)
                            
                            # 计算比例分数
                            # 理想情况下: 腿长约为身高的0.4-0.5，躯干长约为身高的0.3-0.4
                            leg_ratio = leg / height
                            torso_ratio = torso / height
                            arm_ratio = arm / height
                            
                            leg_score = 1.0 - min(abs(leg_ratio - 0.45) * 5, 1.0)
                            torso_score = 1.0 - min(abs(torso_ratio - 0.35) * 5, 1.0)
                            arm_score = 1.0 - min(abs(arm_ratio - 0.3) * 5, 1.0)
                            
                            return (leg_score + torso_score + arm_score) / 3
            
            return 0.5  # 默认中等分数
        except:
            return 0.5  # 出错时返回中等分数
    
    def _check_openpose_proportions(self, points, model_name):
        """检查OpenPose检测出的人体比例"""
        try:
            if model_name == "body_25":
                # 检查关键点是否有效
                valid_points = [p for p in points if p is not None]
                if len(valid_points) < 10:  # 至少需要10个有效点
                    return 0.5
                
                # 计算身高
                neck = points[1]  # 脖子
                mid_hip = points[8]  # 髋部中点
                
                if neck and mid_hip:
                    # 身高
                    body_height = ((neck[0] - mid_hip[0]) ** 2 + (neck[1] - mid_hip[1]) ** 2) ** 0.5
                    
                    scores = []
                    
                    # 检查腿长
                    right_hip = points[9]
                    right_knee = points[10]
                    right_ankle = points[11]
                    
                    if right_hip and right_knee and right_ankle:
                        upper_leg = ((right_hip[0] - right_knee[0]) ** 2 + (right_hip[1] - right_knee[1]) ** 2) ** 0.5
                        lower_leg = ((right_knee[0] - right_ankle[0]) ** 2 + (right_knee[1] - right_ankle[1]) ** 2) ** 0.5
                        
                        leg_ratio = (upper_leg + lower_leg) / body_height
                        leg_score = 1.0 - min(abs(leg_ratio - 0.5) * 4, 1.0)
                        scores.append(leg_score)
                    
                    # 检查手臂长度
                    right_shoulder = points[2]
                    right_elbow = points[3]
                    right_wrist = points[4]
                    
                    if right_shoulder and right_elbow and right_wrist:
                        upper_arm = ((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2) ** 0.5
                        forearm = ((right_elbow[0] - right_wrist[0]) ** 2 + (right_elbow[1] - right_wrist[1]) ** 2) ** 0.5
                        
                        arm_ratio = (upper_arm + forearm) / body_height
                        arm_score = 1.0 - min(abs(arm_ratio - 0.4) * 4, 1.0)
                        scores.append(arm_score)
                    
                    # 计算最终分数
                    if scores:
                        return sum(scores) / len(scores)
            
            elif model_name == "coco":
                # COCO模型的关键点定义不同
                # 省略实现，可根据需要添加
                pass
                
            return 0.5  # 默认中等分数
        except:
            return 0.5  # 出错时返回中等分数
