"""
OpenPose模型管理工具
用于下载和管理OpenPose模型到ComfyUI的models目录
"""

import os
import sys
import argparse
import urllib.request
import shutil
import json
import time
import socket
from urllib.parse import urlparse
from pathlib import Path

# 尝试导入ComfyUI的folder_paths
try:
    import folder_paths
    models_dir = folder_paths.models_dir
except ImportError:
    print("警告：无法导入ComfyUI的folder_paths，将使用当前目录")
    # 使用当前目录下的models文件夹
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入代理配置
try:
    from nodes.pose_assessment import PROXY_CONFIG, load_proxy_config
    # 加载现有的代理配置
    load_proxy_config()
    print(f"已加载代理配置: {'启用' if PROXY_CONFIG['enabled'] else '禁用'}")
except ImportError:
    print("警告：无法导入代理配置，将使用默认设置")
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
        timeout: 连接和读取超时时间(秒)
    """
    print(f"开始下载: {url}")
    print(f"保存到: {destination}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    temp_file = f"{destination}.download"
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                print(f"第 {retry_count} 次重试下载...")
            
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
                with urllib.request.urlopen(request, timeout=timeout) as response, open(temp_file, 'wb') as out_file:
                    # 获取文件大小信息
                    file_size = int(response.headers.get('Content-Length', 0))
                    if file_size == 0:
                        print(f"警告: 无法获取文件大小信息")
                    
                    block_size = 8192
                    downloaded = 0
                    
                    # 显示进度条
                    start_time = time.time()
                    last_update_time = start_time
                    
                    while True:
                        try:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            
                            downloaded += len(buffer)
                            out_file.write(buffer)
                            
                            # 更新下载进度显示
                            if file_size > 0:
                                # 每0.5秒更新一次进度
                                current_time = time.time()
                                if current_time - last_update_time > 0.5:
                                    elapsed_time = current_time - start_time
                                    speed = downloaded / elapsed_time if elapsed_time > 0 else 0
                                    percent = (downloaded / file_size) * 100
                                    
                                    sys.stdout.write(f"\r下载进度: {percent:.1f}% - {downloaded/(1024*1024):.1f}/{file_size/(1024*1024):.1f}MB - {speed/(1024*1024):.2f}MB/s")
                                    sys.stdout.flush()
                                    last_update_time = current_time
                        except socket.timeout:
                            print("\n读取超时，正在重试...")
                            continue
            except urllib.error.URLError as e:
                if isinstance(e.reason, socket.timeout):
                    print(f"连接超时: {e}")
                    retry_count += 1
                    continue
                else:
                    raise
            
            sys.stdout.write("\n")
            
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
                    print(f"下载完成: {destination}")
                    return True
                else:
                    # 如果临时文件大小为0，说明下载失败
                    os.remove(temp_file)
                    print(f"下载失败: {destination} 文件大小为0")
                    retry_count += 1
            else:
                print(f"下载失败: 临时文件不存在")
                retry_count += 1
                
        except Exception as e:
            print(f"下载出错: {str(e)}")
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            retry_count += 1
    
    print(f"下载失败: 已达到最大重试次数 ({max_retries})")
    return False

def download_model(model_name, use_proxy=None, models_dir_override=None):
    """下载指定的OpenPose模型
    
    Args:
        model_name: 要下载的模型名称
        use_proxy: 是否使用代理，None表示使用全局配置
        models_dir_override: 可选的模型目录覆盖，如果不指定则使用ComfyUI的默认目录
    """
    if model_name not in OPENPOSE_MODELS:
        print(f"错误: 未知的模型 '{model_name}'")
        return False
    
    # 确定模型目录
    if models_dir_override:
        openpose_dir = os.path.join(models_dir_override, "openpose", model_name)
    else:
        openpose_dir = os.path.join(models_dir, "openpose", model_name)
    ensure_directory(openpose_dir)
    
    print(f"开始下载OpenPose模型: {model_name}")
    print(f"模型描述: {OPENPOSE_MODELS[model_name]['description']}")
    print(f"保存目录: {openpose_dir}")
    
    # 下载模型文件
    model_info = OPENPOSE_MODELS[model_name]
    success = True
    for file_name, url in model_info["files"].items():
        file_path = os.path.join(openpose_dir, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
                print(f"文件大小为0，重新下载: {file_path}")
                try:
                    os.remove(file_path)
                except:
                    pass
                    
            print(f"下载文件 {file_name}...")
            file_success = download_file(url, file_path, use_proxy=use_proxy)
            success = success and file_success
            
            if not file_success:
                print(f"警告: 文件 {file_name} 下载失败!")
        else:
            print(f"文件已存在并且非空: {file_path}")
    
    # 创建模型信息文件
    if success:
        info_path = os.path.join(openpose_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": model_name,
                "description": model_info["description"],
                "files": list(model_info["files"].keys()),
                "status": "complete",
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)
        print(f"OpenPose模型 {model_name} 下载完成!")
    else:
        # 创建失败状态的信息文件
        info_path = os.path.join(openpose_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": model_name,
                "description": model_info["description"],
                "files": list(model_info["files"].keys()),
                "status": "incomplete",
                "error": "下载失败，请尝试重新下载",
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)
        print(f"OpenPose模型 {model_name} 下载不完整，部分文件可能缺失!")
    
    return success

def download_all_models(use_proxy=None, models_dir_override=None):
    """下载所有预定义的OpenPose模型
    
    Args:
        use_proxy: 是否使用代理，None表示使用全局配置
        models_dir_override: 可选的模型目录覆盖
    """
    print("\n开始下载所有OpenPose模型...")
    
    # 确保OpenPose目录存在
    openpose_dir = os.path.join(models_dir if not models_dir_override else models_dir_override, "openpose")
    ensure_directory(openpose_dir)
    
    # 下载每个模型
    success = True
    for model_name in OPENPOSE_MODELS:
        print(f"\n下载模型: {model_name}")
        success = success and download_model(model_name, use_proxy=use_proxy, models_dir_override=models_dir_override)
    
    if success:
        print("\n所有模型下载完成！")
    else:
        print("\n部分模型下载失败，请检查错误信息。")
    
    return success

def create_default_proxy_config():
    """创建默认代理配置文件"""
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
    return config_path

def update_proxy_config():
    """交互式更新代理配置"""
    global PROXY_CONFIG
    
    print("\n当前代理配置:")
    print(f"启用代理: {'是' if PROXY_CONFIG['enabled'] else '否'}")
    print(f"HTTP代理: {PROXY_CONFIG['http']}")
    print(f"HTTPS代理: {PROXY_CONFIG['https']}")
    print(f"SOCKS代理: {PROXY_CONFIG['socks']}")
    print(f"无代理: {PROXY_CONFIG['no_proxy']}")
    
    # 询问是否要更新
    response = input("\n是否更新代理设置？(y/n): ")
    if response.lower() != 'y':
        return
    
    # 更新设置
    enabled = input(f"启用代理？(y/n) [{'y' if PROXY_CONFIG['enabled'] else 'n'}]: ")
    if enabled.strip():
        PROXY_CONFIG['enabled'] = enabled.lower() == 'y'
    
    http_proxy = input(f"HTTP代理 [{PROXY_CONFIG['http']}]: ")
    if http_proxy.strip():
        PROXY_CONFIG['http'] = http_proxy
    
    https_proxy = input(f"HTTPS代理 [{PROXY_CONFIG['https']}]: ")
    if https_proxy.strip():
        PROXY_CONFIG['https'] = https_proxy
    
    socks_proxy = input(f"SOCKS代理 [{PROXY_CONFIG['socks']}]: ")
    if socks_proxy.strip():
        PROXY_CONFIG['socks'] = socks_proxy
    
    no_proxy = input(f"无代理 [{PROXY_CONFIG['no_proxy']}]: ")
    if no_proxy.strip():
        PROXY_CONFIG['no_proxy'] = no_proxy
    
    # 保存更新后的配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "proxy.json")
    config_dir = os.path.dirname(config_path)
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(PROXY_CONFIG, f, ensure_ascii=False, indent=2)
    
    print(f"代理配置已更新并保存到: {config_path}")
    print("新的代理配置:")
    print(f"启用代理: {'是' if PROXY_CONFIG['enabled'] else '否'}")
    print(f"HTTP代理: {PROXY_CONFIG['http']}")
    print(f"HTTPS代理: {PROXY_CONFIG['https']}")
    print(f"SOCKS代理: {PROXY_CONFIG['socks']}")
    print(f"无代理: {PROXY_CONFIG['no_proxy']}")

def list_models(models_dir_override=None):
    """列出当前已下载的OpenPose模型"""
    openpose_dir = os.path.join(models_dir if not models_dir_override else models_dir_override, "openpose")
    if not os.path.exists(openpose_dir):
        print("OpenPose模型目录不存在")
        return []
    
    models = []
    for model_name in os.listdir(openpose_dir):
        model_path = os.path.join(openpose_dir, model_name)
        if os.path.isdir(model_path):
            info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    models.append({
                        "name": info.get("name", model_name),
                        "description": info.get("description", ""),
                        "status": info.get("status", "unknown"),
                        "download_date": info.get("download_date", "未知"),
                        "path": model_path
                    })
                except:
                    models.append({
                        "name": model_name,
                        "description": "无法读取模型信息",
                        "status": "unknown",
                        "path": model_path
                    })
            else:
                models.append({
                    "name": model_name,
                    "description": "未找到模型信息文件",
                    "status": "unknown",
                    "path": model_path
                })
    
    if not models:
        print("\n未找到已下载的OpenPose模型。")
        print("使用 'python tools/manage_openpose_models.py download' 下载模型。")
    else:
        print("\nOpenPose模型列表:")
        print("-" * 80)
        for i, model in enumerate(models):
            status = "完成" if model.get('status') == "complete" else "不完整" if model.get('status') == "incomplete" else "未知"
            print(f"{i+1}. {model['name']} - {model['description']}")
            print(f"   状态: {status} | 下载时间: {model.get('download_date', '未知')}")
            print(f"   路径: {model['path']}")
        print("-" * 80)
    
    return models

def check_model(model_name, models_dir_override=None):
    """检查模型文件是否已下载并完整
    
    Args:
        model_name: 要检查的模型名称
        models_dir_override: 可选的模型目录覆盖
    """
    if model_name not in OPENPOSE_MODELS:
        print(f"错误: 未知的模型 '{model_name}'")
        return False
    
    # 确定模型目录
    if models_dir_override:
        model_dir = os.path.join(models_dir_override, "openpose", model_name)
    else:
        model_dir = os.path.join(models_dir, "openpose", model_name)
    
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return False
    
    # 检查所有必要文件
    model_info = OPENPOSE_MODELS[model_name]
    all_files_exist = True
    
    for file_name in model_info["files"]:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            all_files_exist = False
        else:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"文件大小为0: {file_path}")
                all_files_exist = False
            else:
                # 对于caffemodel文件，检查文件大小是否合理
                if file_name.endswith('.caffemodel'):
                    # caffemodel文件通常较大，至少应该有几MB
                    min_size = 5 * 1024 * 1024  # 5MB
                    if file_size < min_size:
                        print(f"警告: {file_path} 文件大小异常 ({file_size / (1024*1024):.2f}MB < 5MB)")
                        all_files_exist = False
                    else:
                        print(f"文件正常: {file_path} ({file_size / (1024*1024):.2f} MB)")
                else:
                    print(f"文件正常: {file_path} ({file_size / 1024:.2f} KB)")
    
    # 检查模型信息文件
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                print(f"模型信息: {info}")
                if info.get("status") == "incomplete":
                    all_files_exist = False
        except Exception as e:
            print(f"读取模型信息文件失败: {str(e)}")
            all_files_exist = False
    
    if all_files_exist:
        print(f"模型 '{model_name}' 检查通过，所有文件都存在且有效")
    else:
        print(f"模型 '{model_name}' 检查失败，部分文件不存在或无效")
    
    return all_files_exist

def test_proxy_connection():
    """测试代理连接是否工作"""
    print("\n测试代理连接...")
    
    # 测试URLs
    test_urls = [
        "https://www.google.com",
        "https://github.com",
        "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel"
    ]
    
    # 保存当前的代理设置
    current_proxy = PROXY_CONFIG.copy()
    
    results = []
    
    # 测试1: 不使用代理
    print("\n1. 不使用代理测试:")
    PROXY_CONFIG["enabled"] = False
    for url in test_urls:
        try:
            print(f"  测试URL: {url}")
            req = urllib.request.Request(url)
            urllib.request.urlopen(req, timeout=5)
            print(f"  ✓ 成功")
            results.append(("不使用代理", url, True))
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            results.append(("不使用代理", url, False))
    
    # 测试2: 使用代理
    if current_proxy["http"] or current_proxy["https"]:
        print("\n2. 使用代理测试:")
        PROXY_CONFIG["enabled"] = True
        for url in test_urls:
            try:
                print(f"  测试URL: {url}")
                req = urllib.request.Request(url)
                
                # 根据URL协议选择合适的代理
                parsed_url = urlparse(url)
                if parsed_url.scheme == 'https' and PROXY_CONFIG['https']:
                    proxy = urllib.request.ProxyHandler({'https': PROXY_CONFIG['https']})
                    print(f"  使用HTTPS代理: {PROXY_CONFIG['https']}")
                elif parsed_url.scheme == 'http' and PROXY_CONFIG['http']:
                    proxy = urllib.request.ProxyHandler({'http': PROXY_CONFIG['http']})
                    print(f"  使用HTTP代理: {PROXY_CONFIG['http']}")
                else:
                    proxy = urllib.request.ProxyHandler({
                        'http': PROXY_CONFIG['http'],
                        'https': PROXY_CONFIG['https']
                    })
                    print(f"  使用默认代理")
                
                opener = urllib.request.build_opener(proxy)
                urllib.request.install_opener(opener)
                
                urllib.request.urlopen(req, timeout=5)
                print(f"  ✓ 成功")
                results.append(("使用代理", url, True))
            except Exception as e:
                print(f"  ✗ 失败: {str(e)}")
                results.append(("使用代理", url, False))
    
    # 恢复原始代理设置
    PROXY_CONFIG.update(current_proxy)
    
    # 输出总结
    print("\n测试结果总结:")
    print("-" * 80)
    success_no_proxy = sum(1 for r in results if r[0] == "不使用代理" and r[2])
    success_with_proxy = sum(1 for r in results if r[0] == "使用代理" and r[2])
    
    print(f"不使用代理: {success_no_proxy}/{len(test_urls)} 成功")
    if current_proxy["http"] or current_proxy["https"]:
        print(f"使用代理: {success_with_proxy}/{len(test_urls)} 成功")
    
    # 建议
    print("\n建议:")
    if success_no_proxy > success_with_proxy:
        print("- 直接连接似乎工作得更好，建议禁用代理")
        return False
    elif success_with_proxy > success_no_proxy:
        print("- 使用代理似乎工作得更好，建议启用代理")
        return True
    else:
        if success_no_proxy == len(test_urls):
            print("- 直接连接和代理都可以工作，您可以根据需要选择")
        else:
            print("- 直接连接和代理都有一些问题，可能需要检查您的网络设置")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenPose模型管理工具')
    subparsers = parser.add_subparsers(dest='action', help='操作类型')
    
    # 下载命令
    download_parser = subparsers.add_parser('download', help='下载模型')
    download_parser.add_argument('--model', type=str, help='要下载的模型名称，不指定则下载所有模型')
    download_parser.add_argument('--proxy', choices=['yes', 'no', 'auto'], default='auto', 
                               help='是否使用代理下载 (默认: auto 使用配置文件设置)')
    download_parser.add_argument('--dir', help='保存模型的目录，默认使用ComfyUI models目录')
    
    # 列表命令
    list_parser = subparsers.add_parser('list', help='列出已下载模型')
    list_parser.add_argument('--dir', help='模型所在的目录，默认使用ComfyUI models目录')
    
    # 检查命令
    check_parser = subparsers.add_parser('check', help='检查模型文件是否完整')
    check_parser.add_argument('--model', type=str, help='要检查的模型名称，不指定则检查所有模型')
    check_parser.add_argument('--dir', help='模型所在的目录，默认使用ComfyUI models目录')
    
    # 代理命令
    proxy_parser = subparsers.add_parser('proxy', help='代理设置')
    proxy_parser.add_argument('--test', action='store_true', help='测试代理连接')
    proxy_parser.add_argument('--config', action='store_true', help='配置代理设置')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        use_proxy = None
        if args.proxy == 'yes':
            use_proxy = True
        elif args.proxy == 'no':
            use_proxy = False
            
        if args.model:
            if args.model.lower() == 'all':
                # 下载所有预定义模型
                download_all_models(use_proxy=use_proxy, models_dir_override=args.dir)
            else:
                # 下载指定模型
                if args.model not in OPENPOSE_MODELS:
                    print(f"错误: 未知的模型 '{args.model}'")
                    print("可用的模型有:")
                    for model in OPENPOSE_MODELS:
                        print(f"- {model}: {OPENPOSE_MODELS[model]['description']}")
                    sys.exit(1)
                download_model(args.model, use_proxy=use_proxy, models_dir_override=args.dir)
        else:
            # 下载所有预定义模型
            download_all_models(use_proxy=use_proxy, models_dir_override=args.dir)
            
    elif args.action == 'list':
        list_models(models_dir_override=args.dir)
        
    elif args.action == 'check':
        if args.model:
            if args.model.lower() == 'all':
                # 检查所有模型
                all_ok = True
                for model_name in OPENPOSE_MODELS:
                    print(f"\n检查模型: {model_name}")
                    model_ok = check_model(model_name, models_dir_override=args.dir)
                    all_ok = all_ok and model_ok
                
                if all_ok:
                    print("\n所有模型检查通过")
                    sys.exit(0)
                else:
                    print("\n部分模型检查失败")
                    sys.exit(1)
            else:
                # 检查指定模型
                if args.model not in OPENPOSE_MODELS:
                    print(f"错误: 未知的模型 '{args.model}'")
                    print("可用的模型有:")
                    for model in OPENPOSE_MODELS:
                        print(f"- {model}: {OPENPOSE_MODELS[model]['description']}")
                    sys.exit(1)
                success = check_model(args.model, models_dir_override=args.dir)
                sys.exit(0 if success else 1)
        else:
            # 检查所有模型
            all_ok = True
            for model_name in OPENPOSE_MODELS:
                print(f"\n检查模型: {model_name}")
                model_ok = check_model(model_name, models_dir_override=args.dir)
                all_ok = all_ok and model_ok
            
            if all_ok:
                print("\n所有模型检查通过")
                sys.exit(0)
            else:
                print("\n部分模型检查失败")
                sys.exit(1)
                
    elif args.action == 'proxy':
        if args.test:
            test_proxy_connection()
        elif args.config:
            update_proxy_config()
        else:
            proxy_parser.print_help()
    else:
        parser.print_help()
