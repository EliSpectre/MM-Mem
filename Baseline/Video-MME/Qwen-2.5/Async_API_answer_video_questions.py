"""
使用 vLLM 部署的 OpenAI 兼容 API 进行视频问答
并行化版本：使用 asyncio 异步并发请求加速处理
每个视频均匀采样128帧，分辨率224x224
"""
import json
import os
import sys
import asyncio
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# --- 1. 配置 ---

# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen3-VL-2B-Instruct"
# 文件路径（配置）
VIDEO_DIR = "/data/tempuser2/MMAgent/Video-MME/Video_MME_Videos"
JSON_PATH = "/data/tempuser2/MMAgent/MM-Mem/Baseline/Video-MME/Dataset/video_mme_test.json"

# 并行化配置
MAX_CONCURRENT_REQUESTS = 15  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
SEMAPHORE = None  # 信号量，控制并发数

# 视频帧采样配置
NUM_FRAMES = 128  # 均匀采样帧数
FRAME_SIZE = (224, 224)  # 帧分辨率 (宽, 高)

# --- 2. 核心功能函数 ---

def load_questions(json_path):
    """从 JSON 文件加载问题列表。"""
    if not os.path.exists(json_path):
        print(f"错误：问题文件未找到 at {json_path}", file=sys.stderr)
        return []
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"解析 JSON 文件时出错: {e}", file=sys.stderr)
        return []

def normalize_answer(answer_str):
    """从答案字符串中提取唯一的字母选项（例如 "A." -> "A"）。"""
    if not isinstance(answer_str, str):
        return ""
    for char in answer_str.strip():
        if char.isalpha():
            return char.upper()
    return ""


def extract_frames_from_video(video_path, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
    """
    从视频中均匀采样指定数量的帧，并调整分辨率。
    
    Args:
        video_path (str): 视频文件路径
        num_frames (int): 要采样的帧数
        frame_size (tuple): 目标分辨率 (宽, 高)
        
    Returns:
        list: base64 编码的图片列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"视频帧数无效: {video_path}")
    
    # 计算均匀采样的帧索引
    if total_frames <= num_frames:
        # 如果视频帧数少于目标帧数，取所有帧
        frame_indices = list(range(total_frames))
    else:
        # 均匀采样
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames_base64 = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # BGR 转 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整分辨率
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize(frame_size, Image.LANCZOS)
        
        # 转换为 base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        frames_base64.append(img_base64)
    
    cap.release()
    return frames_base64


def build_image_content_list(frames_base64):
    """
    构建多图片内容列表，用于 API 请求。
    
    Args:
        frames_base64 (list): base64 编码的图片列表
        
    Returns:
        list: 图片内容列表
    """
    content_list = []
    for img_b64 in frames_base64:
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })
    return content_list

async def ask_question_with_video_async(client, video_path, question_text, options, task_info, frames_base64=None):
    """
    异步版本：使用 vLLM API 传入采样帧进行问答。
    
    Args:
        client: AsyncOpenAI 客户端
        video_path (str): 视频文件路径
        question_text (str): 问题文本
        options (list): 选项列表
        task_info (dict): 任务信息，用于结果记录
        frames_base64 (list): 预提取的帧（base64编码），如果为None则现场提取
        
    Returns:
        dict: 包含结果的字典
    """
    global SEMAPHORE
    
    # 构建 Prompt
    options_str = "\n".join(options)
    prompt = (
        f"These are {NUM_FRAMES} frames uniformly sampled from a video.\n"
        "Select the best answer to the following multiple-choice question based on these frames.\n"
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"Question: {question_text} Possible answer choices:\n"
        f"{options_str}\n"
        "The best answer is:"
    )
    
    # 使用信号量控制并发数
    async with SEMAPHORE:
        try:
            # 如果没有预提取帧，则现场提取
            if frames_base64 is None:
                frames_base64 = extract_frames_from_video(video_path)
            
            # 构建图片内容列表
            image_content_list = build_image_content_list(frames_base64)
            
            # 添加文本提示
            content = image_content_list + [{"type": "text", "text": prompt}]
            
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
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
            ground_truth_norm = normalize_answer(task_info['ground_truth'])
            
            is_correct = (model_answer_norm and ground_truth_norm and 
                         model_answer_norm == ground_truth_norm)
            
            return {
                'success': True,
                'video_file': task_info['video_file'],
                'videoID': task_info['video_id'],
                'question_id': task_info['question_id'],
                'duration': task_info['duration'],
                'question': question_text,
                'options': options,
                'ground_truth': task_info['ground_truth'],
                'model_prediction': model_answer_raw,
                'is_correct': is_correct,
            }
            
        except Exception as e:
            print(f"请求失败 [video={task_info['video_id']}]: {e}", file=sys.stderr)
            return {
                'success': False,
                'video_file': task_info['video_file'],
                'videoID': task_info['video_id'],
                'question_id': task_info['question_id'],
                'duration': task_info['duration'],
                'question': question_text,
                'options': options,
                'ground_truth': task_info['ground_truth'],
                'model_prediction': None,
                'is_correct': False,
                'error': str(e),
            }

async def process_all_tasks(tasks, client):
    """
    并行处理所有任务。
    
    Args:
        tasks: 任务列表
        client: AsyncOpenAI 客户端
        
    Returns:
        list: 所有结果
    """
    # 创建所有异步任务
    async_tasks = []
    for task in tasks:
        async_task = ask_question_with_video_async(
            client,
            task['video_path'],
            task['question_text'],
            task['options'],
            task['task_info'],
            task.get('frames_base64')  # 传递预提取的帧
        )
        async_tasks.append(async_task)
    
    # 并行执行所有任务，显示进度
    print(f"\n开始并行处理 {len(async_tasks)} 个任务（最大并发数: {MAX_CONCURRENT_REQUESTS}）...")
    
    results = []
    # 使用 asyncio.as_completed 实时显示进度
    completed = 0
    total_tasks = len(async_tasks)
    # 计算进度数字的宽度，用于对齐
    progress_width = len(str(total_tasks))
    
    for coro in asyncio.as_completed(async_tasks):
        result = await coro
        completed += 1
        if result.get('success'):
            is_correct = result.get('is_correct')
            status = "✓" if is_correct else "✗"
            prefix = "\t" if is_correct else ""
            # 格式化输出，保证对齐
            print(f"[{completed:>{progress_width}}/{total_tasks}] {status} "
                  f"video={result['videoID']:<20} 预测={result['model_prediction']:<4} "
                  f"答案={result['ground_truth']:<4} {prefix}{prefix}正确={str(is_correct):<5}")
        else:
            print(f"[{completed:>{progress_width}}/{total_tasks}] ⚠ "
                  f"video={result['videoID']:<20} 失败")
        results.append(result)
    
    return results

# --- 3. 主程序 ---

async def main_async():
    """
    异步主执行函数：并行处理所有视频问答任务。
    """
    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    start_time = time.time()
    
    # 步骤 1: 加载所有问题并建立 videoID -> questions 的映射
    all_questions = load_questions(JSON_PATH)
    if not all_questions:
        print(f"错误：无法从 {JSON_PATH} 加载问题。", file=sys.stderr)
        return

    vid_map = {}
    for q in all_questions:
        vid = q.get("videoID") or q.get("video_id") or q.get("videoId")
        if vid:
            vid_map.setdefault(vid, []).append(q)

    # 步骤 2: 初始化异步 OpenAI 客户端
    print(f"正在连接 vLLM 服务: {BASE_URL}")
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    print("异步客户端初始化成功！")

    # 步骤 3: 遍历视频目录（递归搜索所有子目录）
    if not os.path.isdir(VIDEO_DIR):
        print(f"错误：视频目录不存在: {VIDEO_DIR}", file=sys.stderr)
        return

    # 递归查找所有 mp4 文件
    mp4_files = []
    for root, dirs, files in os.walk(VIDEO_DIR):
        for f in files:
            if f.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, f))
    
    if not mp4_files:
        print(f"在目录 {VIDEO_DIR} 中未找到 mp4 文件。")
        return
    
    print(f"找到 {len(mp4_files)} 个 mp4 文件。")

    # 步骤 4: 构建所有任务（预提取视频帧）
    tasks = []
    video_frames_cache = {}  # 缓存已提取的视频帧
    
    print(f"\n开始预提取视频帧（每个视频 {NUM_FRAMES} 帧，分辨率 {FRAME_SIZE[0]}x{FRAME_SIZE[1]}）...")
    
    for video_path in sorted(mp4_files):
        mp4_file = os.path.basename(video_path)
        video_id = os.path.splitext(mp4_file)[0]
        
        questions_for_video = vid_map.get(video_id)
        if not questions_for_video:
            continue
        
        # 预提取当前视频的帧（同一视频的多个问题共享）
        if video_id not in video_frames_cache:
            try:
                print(f"  提取帧: {video_id}...")
                frames_base64 = extract_frames_from_video(video_path)
                video_frames_cache[video_id] = frames_base64
                print(f"    成功提取 {len(frames_base64)} 帧")
            except Exception as e:
                print(f"    提取帧失败: {e}", file=sys.stderr)
                video_frames_cache[video_id] = None
        
        frames_base64 = video_frames_cache.get(video_id)
        if frames_base64 is None:
            continue  # 跳过无法提取帧的视频

        for q_data in questions_for_video:
            duration = q_data.get("duration", "unknown").lower()
            question_text = q_data.get("question", "No question text found.")
            options = q_data.get("options", [])
            ground_truth_answer = q_data.get("answer", "")

            tasks.append({
                'video_path': video_path,
                'question_text': question_text,
                'options': options,
                'frames_base64': frames_base64,  # 添加预提取的帧
                'task_info': {
                    'video_file': mp4_file,
                    'video_id': video_id,
                    'question_id': q_data.get('question_id'),
                    'duration': duration,
                    'ground_truth': ground_truth_answer,
                }
            })
    
    print(f"帧提取完成，共处理 {len(video_frames_cache)} 个视频。")

    if not tasks:
        print("没有找到匹配的视频-问题对。")
        return

    print(f"共构建 {len(tasks)} 个问答任务。")

    # 步骤 5: 并行处理所有任务
    all_results = await process_all_tasks(tasks, client)

    # 步骤 6: 统计结果
    correct_by_duration = {'short': 0, 'medium': 0, 'long': 0, 'unknown': 0}
    total_by_duration = {'short': 0, 'medium': 0, 'long': 0, 'unknown': 0}
    
    successful_results = []
    for result in all_results:
        if result.get('success'):
            duration = result['duration']
            total_by_duration[duration] += 1
            if result['is_correct']:
                correct_by_duration[duration] += 1
            successful_results.append(result)

    # 步骤 7: 打印最终结果
    elapsed_time = time.time() - start_time
    print(f"\n\n{'='*20}\n最终结果\n{'='*20}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    total_questions = sum(total_by_duration.values())
    total_correct = sum(correct_by_duration.values())

    if total_questions > 0:
        overall_accuracy = (total_correct / total_questions) * 100
        print(f"总问题数: {total_questions}")
        print(f"正确回答数: {total_correct}")
        print(f"总准确率: {overall_accuracy:.2f}%")
        print(f"平均每题耗时: {elapsed_time/total_questions:.2f} 秒")
        
        print("\n--- 按时长分类准确率 ---")
        for duration, total in total_by_duration.items():
            if total > 0:
                correct = correct_by_duration[duration]
                accuracy = (correct / total) * 100
                print(f"  - {duration.capitalize()}: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("\n未处理任何问题，无法计算准确率。")

    # 步骤 8: 保存所有结果
    out_path = os.path.join(os.getcwd(), MODEL_NAME, 'video_answers_vllm_api_parallel.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(successful_results, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！所有结果已保存到: {out_path}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}", file=sys.stderr)


def main():
    """同步入口函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
