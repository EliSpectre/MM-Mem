"""
使用 vLLM 部署的 OpenAI 兼容 API 进行 HD-EPIC 数据集的视频问答
并行化版本：使用 asyncio 异步并发请求加速处理

支持功能：
1. 读取给定文件夹下的所有JSON文件
2. 处理每个JSON文件中的问题
3. 根据问题中的id递归查找对应的视频文件
4. 处理inputs中的多个"video 1"、"video 2"或"image 1"等
5. 如果video和image的id相同，自动去重
"""

import json
import os
import sys
import asyncio
import glob
import base64
from openai import AsyncOpenAI
import time
from typing import Dict, List, Set, Tuple, Optional
import argparse
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- 1. 配置 ---

# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"

# 默认文件路径（可通过命令行参数覆盖）
DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/Benchmark/VideoPre/rgb_224_1_vig"
DEFAULT_JSON_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test"

# 并行化配置
MAX_CONCURRENT_REQUESTS = 2  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
SEMAPHORE = None  # 信号量，控制并发数

# 视频采帧配置
NUM_FRAMES_PER_VIDEO = 128  # 总共均匀采帧数量（多个视频时平均分配）

# 帧预提取配置
NUM_WORKERS = 8  # 帧提取并行工作进程数

# --- 2. 核心功能函数 ---

def load_all_json_files(json_dir: str) -> Dict[str, dict]:
    """
    递归加载给定文件夹下的所有 JSON 文件。
    
    Args:
        json_dir: JSON 文件所在的文件夹路径
        
    Returns:
        dict: {json文件路径: json内容}
    """
    all_data = {}
    
    # 递归查找所有 JSON 文件
    json_pattern = os.path.join(json_dir, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        print(f"警告：在 {json_dir} 中未找到 JSON 文件", file=sys.stderr)
        return all_data
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data[json_path] = data
                print(f"  - 加载: {os.path.basename(json_path)} ({len(data)} 个问题)")
        except json.JSONDecodeError as e:
            print(f"解析 JSON 文件时出错 [{json_path}]: {e}", file=sys.stderr)
        except Exception as e:
            print(f"读取文件时出错 [{json_path}]: {e}", file=sys.stderr)
    
    return all_data


def find_video_path(video_id: str, video_dir: str, video_cache: Dict[str, str]) -> Optional[str]:
    """
    递归查找视频文件路径。
    
    Args:
        video_id: 视频ID，例如 "P01-20240202-110250"
        video_dir: 视频文件夹根路径
        video_cache: 视频路径缓存字典
        
    Returns:
        视频文件完整路径，如果未找到则返回 None
    """
    # 首先检查缓存
    if video_id in video_cache:
        return video_cache[video_id]
    
    # 支持的视频扩展名
    video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MKV', '.mov', '.MOV']
    
    # 递归搜索
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            # 检查文件名（不包含扩展名）是否匹配
            file_name_no_ext = os.path.splitext(f)[0]
            file_ext = os.path.splitext(f)[1]
            
            if file_name_no_ext == video_id and file_ext in video_extensions:
                video_path = os.path.join(root, f)
                video_cache[video_id] = video_path
                return video_path
    
    return None


def build_video_cache(video_dir: str) -> Dict[str, str]:
    """
    预先构建视频ID到路径的缓存映射。
    
    Args:
        video_dir: 视频文件夹根路径
        
    Returns:
        dict: {video_id: video_path}
    """
    video_cache = {}
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
    
    print(f"正在扫描视频目录: {video_dir}")
    
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in video_extensions:
                video_id = os.path.splitext(f)[0]
                video_cache[video_id] = os.path.join(root, f)
    
    print(f"共找到 {len(video_cache)} 个视频文件")
    return video_cache


def extract_video_ids_from_inputs(inputs: dict) -> Tuple[List[str], List[str]]:
    """
    从 inputs 字典中提取所有唯一的视频/图像 ID 和对应的标签。
    
    支持格式：
    - "video 1": {"id": "P01-20240202-110250"}
    - "video 2": {"id": "P01-20240203-152323"}
    - "image 1": {"id": "P01-20240202-110250"}
    
    Args:
        inputs: 问题中的 inputs 字典
        
    Returns:
        tuple: (去重后的视频ID列表, 对应的视频标签列表如["video 1", "video 2"])
    """
    video_ids = []
    video_labels = []
    seen_ids = set()
    
    # 按键名排序，确保 video/image 1, 2, 3... 的顺序
    sorted_keys = sorted(inputs.keys(), key=lambda x: (
        0 if x.startswith("video") else 1,  # video 优先
        int(x.split()[-1]) if x.split()[-1].isdigit() else 0  # 按数字排序
    ))
    
    label_counter = 1
    for key in sorted_keys:
        value = inputs[key]
        # 处理 "video X" 或 "image X" 格式
        if key.startswith("video") or key.startswith("image"):
            if isinstance(value, dict) and "id" in value:
                vid_id = value["id"]
                if vid_id not in seen_ids:
                    video_ids.append(vid_id)
                    # 统一使用 "video X" 格式作为标签
                    video_labels.append(f"video {label_counter}")
                    seen_ids.add(vid_id)
                    label_counter += 1
    
    return video_ids, video_labels


def normalize_answer(answer_str) -> str:
    """从答案字符串中提取唯一的字母选项（例如 "A." -> "A"）。"""
    if isinstance(answer_str, int):
        # 如果是索引，转换为字母
        return chr(ord('A') + answer_str)
    if not isinstance(answer_str, str):
        return ""
    for char in answer_str.strip():
        if char.isalpha():
            return char.upper()
    return ""


def get_correct_answer_letter(correct_idx: int) -> str:
    """将正确答案索引转换为字母。"""
    if 0 <= correct_idx <= 25:
        return chr(ord('A') + correct_idx)
    return ""


def extract_frames_from_video(video_path: str, num_frames: int = 128) -> List[str]:
    """
    从视频中均匀采帧并转换为 base64 编码的图片。
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        
    Returns:
        list: base64 编码的图片列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}", file=sys.stderr)
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"视频帧数无效: {video_path}", file=sys.stderr)
        cap.release()
        return []
    
    # 计算均匀采帧的帧索引
    if total_frames <= num_frames:
        # 如果视频帧数少于目标帧数，取所有帧
        frame_indices = list(range(total_frames))
    else:
        # 均匀采帧
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames_base64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 将帧编码为 JPEG 格式
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # 转换为 base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_base64)
    
    cap.release()
    return frames_base64


async def ask_question_with_frames_async(
    client,
    preextracted_frames: List[List[str]],
    video_labels: List[str],
    question_text: str,
    choices: List[str],
    task_info: dict
):
    """
    异步版本：使用 vLLM API 传入预提取的帧图片进行问答。
    支持多视频图文交互，每个视频的帧前加入 <video X> 标识。
    
    Args:
        client: AsyncOpenAI 客户端
        preextracted_frames (list): 预提取的帧列表，每个元素是一个视频的帧base64列表
        video_labels (list): 视频标签列表，如 ["video 1", "video 2"]
        question_text (str): 问题文本
        choices (list): 选项列表
        task_info (dict): 任务信息，用于结果记录
        
    Returns:
        dict: 包含结果的字典
    """
    global SEMAPHORE
    
    # 构建选项字符串（带字母标号）
    options_with_letters = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        options_with_letters.append(f"{letter}. {choice}")
    options_str = "\n".join(options_with_letters)
    
    prompt = (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
        f"Question: {question_text}\n"
        f"Possible answer choices:\n{options_str}\n"
        "The best answer is:"
    )
    
    # 构建消息内容，使用预提取的帧
    content = []
    
    if not preextracted_frames:
        # 添加问题文本
        content.append({"type": "text", "text": prompt})
    else:
        for video_idx, (frames_base64, video_label) in enumerate(zip(preextracted_frames, video_labels)):
            # 添加当前视频的标识文本
            content.append({
                "type": "text",
                "text": f"<{video_label}>"
            })
            
            # 将每一帧作为图片添加到 content 中
            for frame_base64 in frames_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}"
                    }
                })
        
        # 添加问题文本
        content.append({"type": "text", "text": prompt})
    
    # 使用信号量控制并发数
    async with SEMAPHORE:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video analyzer, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure. Videos are at 1fps, and timestamps are MM:SS. Bounding boxes are in the format (ymin, xmin, ymax, xmax) relative to an image size of 1000x1000."
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=2,
                temperature=0,
            )
            
            model_answer_raw = response.choices[0].message.content.strip()
            model_answer_norm = normalize_answer(model_answer_raw)
            ground_truth_norm = task_info['ground_truth']
            
            is_correct = (model_answer_norm and ground_truth_norm and 
                         model_answer_norm == ground_truth_norm)
            
            return {
                'success': True,
                'question_key': task_info['question_key'],
                'video_ids': task_info['video_ids'],
                'question': question_text,
                'choices': choices,
                'ground_truth': ground_truth_norm,
                'ground_truth_idx': task_info['ground_truth_idx'],
                'model_prediction': model_answer_raw,
                'model_prediction_norm': model_answer_norm,
                'is_correct': is_correct,
                'json_file': task_info['json_file'],
            }
            
        except Exception as e:
            print(f"请求失败 [question={task_info['question_key']}]: {e}", file=sys.stderr)
            return {
                'success': False,
                'question_key': task_info['question_key'],
                'video_ids': task_info['video_ids'],
                'question': question_text,
                'choices': choices,
                'ground_truth': task_info['ground_truth'],
                'ground_truth_idx': task_info['ground_truth_idx'],
                'model_prediction': None,
                'model_prediction_norm': None,
                'is_correct': False,
                'json_file': task_info['json_file'],
                'error': str(e),
            }


async def process_all_tasks(tasks: list, client):
    """
    并行处理所有任务。
    
    Args:
        tasks: 任务列表（已包含预提取的帧）
        client: AsyncOpenAI 客户端
        
    Returns:
        list: 所有结果
    """
    # 创建所有异步任务
    async_tasks = []
    for task in tasks:
        async_task = ask_question_with_frames_async(
            client,
            task['preextracted_frames'],
            task['video_labels'],
            task['question_text'],
            task['choices'],
            task['task_info']
        )
        async_tasks.append(async_task)
    
    # 并行执行所有任务，显示进度
    print(f"\n开始并行处理 {len(async_tasks)} 个任务（最大并发数: {MAX_CONCURRENT_REQUESTS}）...")
    
    results = []
    completed = 0
    total_tasks = len(async_tasks)
    progress_width = len(str(total_tasks))
    
    for coro in asyncio.as_completed(async_tasks):
        result = await coro
        completed += 1
        if result.get('success'):
            is_correct = result.get('is_correct')
            status = "✓" if is_correct else "✗"
            prefix = "\t" if is_correct else ""
            print(f"[{completed:>{progress_width}}/{total_tasks}] {status} "
                  f"Q={result['question_key'][:50]:<50} "
                  f"预测={result['model_prediction_norm']:<2} "
                  f"答案={result['ground_truth']:<2} "
                  f"{prefix}正确={str(is_correct):<5}")
        else:
            print(f"[{completed:>{progress_width}}/{total_tasks}] ⚠ "
                  f"Q={result['question_key'][:50]:<50} 失败")
        results.append(result)
    
    return results


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="HD-EPIC 数据集视频问答测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Async_API_answer_HD_EPIC_questions.py --json-dir ./Dataset --video-dir /path/to/videos
  python Async_API_answer_HD_EPIC_questions.py -j ./Dataset -v /data/HD-EPIC/videos -c 20
        """
    )
    
    parser.add_argument(
        "-j", "--json-dir",
        type=str,
        default=DEFAULT_JSON_DIR,
        help=f"JSON 数据集文件夹路径 (默认: {DEFAULT_JSON_DIR})"
    )
    
    parser.add_argument(
        "-v", "--video-dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"视频文件夹路径 (默认: {DEFAULT_VIDEO_DIR})"
    )
    
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help=f"最大并发请求数 (默认: {MAX_CONCURRENT_REQUESTS})"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出结果文件路径 (默认: 当前目录下的 hd_epic_results.json)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help=f"vLLM API 基础 URL (默认: {BASE_URL})"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"模型名称 (默认: {MODEL_NAME})"
    )
    
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=NUM_FRAMES_PER_VIDEO,
        help=f"总共均匀采帧数量（多个视频时平均分配） (默认: {NUM_FRAMES_PER_VIDEO})"
    )
    
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"帧提取并行工作进程数 (默认: {NUM_WORKERS})"
    )
    
    return parser.parse_args()


# --- 3. 主程序 ---

def extract_frames_for_video(task_data: dict) -> dict:
    """
    为单个视频提取帧（用于多进程并行）。
    
    Args:
        task_data: 包含视频ID、路径和帧数信息的字典
        
    Returns:
        dict: 包含视频ID和提取的帧的字典
    """
    video_id = task_data['video_id']
    video_path = task_data['video_path']
    num_frames = task_data['num_frames']
    
    frames_base64 = extract_frames_from_video(video_path, num_frames)
    
    return {
        'video_id': video_id,
        'frames': frames_base64
    }


async def main_async():
    """
    异步主执行函数：并行处理所有视频问答任务。
    """
    global SEMAPHORE, BASE_URL, MODEL_NAME, MAX_CONCURRENT_REQUESTS, NUM_FRAMES_PER_VIDEO, NUM_WORKERS
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新全局配置
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    MAX_CONCURRENT_REQUESTS = args.concurrency
    NUM_FRAMES_PER_VIDEO = args.num_frames
    NUM_WORKERS = args.num_workers
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    json_dir = args.json_dir
    video_dir = args.video_dir
    output_path = args.output or os.path.join(os.getcwd(), MODEL_NAME, 'hd_epic_results.json')
    
    start_time = time.time()
    
    print("=" * 60)
    print("HD-EPIC 数据集视频问答测试（均匀采帧模式 - 预提取优化版）")
    print("=" * 60)
    print(f"JSON 文件夹: {json_dir}")
    print(f"视频文件夹: {video_dir}")
    print(f"总采帧数: {NUM_FRAMES_PER_VIDEO} (多视频时平均分配)")
    print(f"帧提取并行进程数: {NUM_WORKERS}")
    print(f"API最大并发数: {MAX_CONCURRENT_REQUESTS}")
    print(f"API 地址: {BASE_URL}")
    print(f"模型名称: {MODEL_NAME}")
    print("=" * 60)
    
    # 步骤 1: 检查路径有效性
    if not os.path.isdir(json_dir):
        print(f"错误：JSON 文件夹不存在: {json_dir}", file=sys.stderr)
        return
    
    if not os.path.isdir(video_dir):
        print(f"错误：视频文件夹不存在: {video_dir}", file=sys.stderr)
        return
    
    # 步骤 2: 加载所有 JSON 文件
    print("\n[步骤 1] 加载 JSON 数据集...")
    all_json_data = load_all_json_files(json_dir)
    if not all_json_data:
        print("错误：未能加载任何 JSON 数据", file=sys.stderr)
        return
    
    # 步骤 3: 预先构建视频缓存
    print("\n[步骤 2] 扫描视频文件...")
    video_cache = build_video_cache(video_dir)
    if not video_cache:
        print("警告：未找到任何视频文件", file=sys.stderr)
    
    # 步骤 4: 初始化异步 OpenAI 客户端
    print(f"\n[步骤 3] 连接 vLLM 服务: {BASE_URL}")
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    print("异步客户端初始化成功！")
    
    # 步骤 5: 构建所有任务
    print("\n[步骤 4] 构建问答任务...")
    tasks = []
    skipped_no_video = 0
    
    for json_path, json_content in all_json_data.items():
        json_basename = os.path.basename(json_path)
        
        for question_key, question_data in json_content.items():
            # 提取问题信息
            inputs = question_data.get("inputs", {})
            question_text = question_data.get("question", "")
            choices = question_data.get("choices", [])
            correct_idx = question_data.get("correct_idx", -1)
            
            if not question_text or not choices:
                print(f"  跳过无效问题: {question_key}", file=sys.stderr)
                continue
            
            # 提取所有视频ID和对应标签（自动去重）
            video_ids, video_labels = extract_video_ids_from_inputs(inputs)
            
            if not video_ids:
                print(f"  跳过无视频ID的问题: {question_key}", file=sys.stderr)
                continue
            
            # 查找所有视频路径
            video_paths = []
            valid_labels = []
            missing_videos = []
            for vid_id, vid_label in zip(video_ids, video_labels):
                video_path = video_cache.get(vid_id)
                if video_path:
                    video_paths.append(video_path)
                    valid_labels.append(vid_label)
                else:
                    missing_videos.append(vid_id)
            
            if not video_paths:
                skipped_no_video += 1
                if skipped_no_video <= 5:  # 只显示前5个警告
                    print(f"  跳过（视频未找到）: {question_key}, 缺少: {missing_videos}", file=sys.stderr)
                continue
            
            if missing_videos:
                print(f"  警告: {question_key} 部分视频未找到: {missing_videos}", file=sys.stderr)
            
            # 获取正确答案字母
            ground_truth_letter = get_correct_answer_letter(correct_idx)
            
            tasks.append({
                'video_paths': video_paths,
                'video_labels': valid_labels,
                'question_text': question_text,
                'choices': choices,
                'task_info': {
                    'question_key': question_key,
                    'video_ids': video_ids,
                    'ground_truth': ground_truth_letter,
                    'ground_truth_idx': correct_idx,
                    'json_file': json_basename,
                }
            })
    
    if skipped_no_video > 5:
        print(f"  ... 还有 {skipped_no_video - 5} 个问题因视频未找到而跳过", file=sys.stderr)
    
    if not tasks:
        print("没有找到有效的问答任务。")
        return
    
    print(f"\n共构建 {len(tasks)} 个问答任务（跳过 {skipped_no_video} 个因视频未找到）")
    
    # 步骤 6: 收集所有需要的唯一视频ID
    print("\n[步骤 5] 收集需要提取帧的视频...")
    unique_video_ids = set()
    for task in tasks:
        for vid_id in task['task_info']['video_ids']:
            if vid_id in video_cache:
                unique_video_ids.add(vid_id)
    
    print(f"  共有 {len(unique_video_ids)} 个唯一视频需要提取帧")
    
    # 步骤 7: 预提取所有视频帧（按视频ID去重，多进程并行）
    print(f"\n[步骤 6] 预提取视频帧（{NUM_WORKERS} 个并行进程）...")
    frame_extraction_start = time.time()
    
    # 准备帧提取任务（每个视频只提取一次）
    extraction_tasks = []
    for video_id in unique_video_ids:
        extraction_tasks.append({
            'video_id': video_id,
            'video_path': video_cache[video_id],
            'num_frames': NUM_FRAMES_PER_VIDEO  # 每个视频都提取完整帧数
        })
    
    # 使用多进程并行提取帧
    video_frames_cache = {}  # {video_id: [frame1_base64, frame2_base64, ...]}
    completed_extractions = 0
    total_extractions = len(extraction_tasks)
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(extract_frames_for_video, task): task['video_id'] 
                   for task in extraction_tasks}
        
        for future in as_completed(futures):
            result = future.result()
            video_id = result['video_id']
            video_frames_cache[video_id] = result['frames']
            completed_extractions += 1
            
            # 每10个视频打印一次进度
            if completed_extractions % 10 == 0 or completed_extractions == total_extractions:
                print(f"  帧提取进度: {completed_extractions}/{total_extractions} 个视频 "
                      f"({100*completed_extractions/total_extractions:.1f}%)")
    
    frame_extraction_elapsed = time.time() - frame_extraction_start
    print(f"  帧提取完成！耗时: {frame_extraction_elapsed:.2f} 秒")
    
    # 步骤 8: 为每个任务组装预提取的帧
    print("\n[步骤 7] 为任务分配帧...")
    for task in tasks:
        video_ids = task['task_info']['video_ids']
        num_videos = len(video_ids)
        
        # 计算每个视频应该使用的帧数（平均分配）
        frames_per_video = NUM_FRAMES_PER_VIDEO // num_videos if num_videos > 0 else 0
        remaining_frames = NUM_FRAMES_PER_VIDEO % num_videos if num_videos > 0 else 0
        
        preextracted_frames = []
        for video_idx, vid_id in enumerate(video_ids):
            if vid_id in video_frames_cache:
                all_frames = video_frames_cache[vid_id]
                # 第一个视频获得额外的剩余帧数
                num_frames_for_this_video = frames_per_video + (remaining_frames if video_idx == 0 else 0)
                # 从缓存的帧中均匀采样
                if len(all_frames) <= num_frames_for_this_video:
                    selected_frames = all_frames
                else:
                    indices = np.linspace(0, len(all_frames) - 1, num_frames_for_this_video, dtype=int).tolist()
                    selected_frames = [all_frames[i] for i in indices]
                preextracted_frames.append(selected_frames)
            else:
                preextracted_frames.append([])
        
        task['preextracted_frames'] = preextracted_frames
    
    print(f"  任务帧分配完成！")
    
    # 步骤 9: 并行处理所有API请求
    print("\n[步骤 8] 开始问答测试...")
    all_results = await process_all_tasks(tasks, client)
    
    # 步骤 10: 统计结果
    results_by_json = {}  # 按 JSON 文件分类统计
    total_correct = 0
    total_questions = 0
    successful_results = []
    
    for result in all_results:
        if result.get('success'):
            json_file = result.get('json_file', 'unknown')
            
            if json_file not in results_by_json:
                results_by_json[json_file] = {'correct': 0, 'total': 0}
            
            results_by_json[json_file]['total'] += 1
            total_questions += 1
            
            if result['is_correct']:
                results_by_json[json_file]['correct'] += 1
                total_correct += 1
            
            successful_results.append(result)
    
    # 步骤 8: 打印最终结果
    elapsed_time = time.time() - start_time
    print(f"\n\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    if total_questions > 0:
        overall_accuracy = (total_correct / total_questions) * 100
        print(f"\n总问题数: {total_questions}")
        print(f"正确回答数: {total_correct}")
        print(f"总准确率: {overall_accuracy:.2f}%")
        print(f"平均每题耗时: {elapsed_time/total_questions:.2f} 秒")
        
        print("\n--- 按数据集文件分类准确率 ---")
        for json_file, stats in sorted(results_by_json.items()):
            correct = stats['correct']
            total = stats['total']
            if total > 0:
                accuracy = (correct / total) * 100
                print(f"  - {json_file}: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("\n未处理任何问题，无法计算准确率。")
    
    # 步骤 9: 保存所有结果
    try:
        # 构建完整的结果报告
        output_data = {
            'summary': {
                'total_questions': total_questions,
                'total_correct': total_correct,
                'accuracy': (total_correct / total_questions * 100) if total_questions > 0 else 0,
                'elapsed_time_seconds': elapsed_time,
                'json_dir': json_dir,
                'video_dir': video_dir,
                'model_name': MODEL_NAME,
            },
            'results_by_file': results_by_json,
            'detailed_results': successful_results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！所有结果已保存到: {output_path}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}", file=sys.stderr)


def main():
    """同步入口函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
