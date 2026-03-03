"""
VStream-QA MovieNet 数据集视频问答
并行化版本：使用 asyncio 异步并发请求加速处理

支持功能：
1. 读取 test_qa_movienet.json 问题文件
2. 根据 video_id 找到对应的帧文件夹
3. 读取从第一张图片到 end_time 对应的所有帧
4. 使用 vLLM API 进行开放式 VQA 问答
5. 使用 GPT 模型对回答进行评分
"""

import json
import os
import sys
import asyncio
import glob
import base64
import io
from openai import AsyncOpenAI, OpenAI
import time
from typing import Dict, List, Set, Tuple, Optional
import argparse
from pathlib import Path
import re
import ast
from PIL import Image

# --- 1. 配置 ---

# vLLM 部署的 OpenAI 兼容服务器地址（用于 VQA）
VLLM_BASE_URL = "http://localhost:8002/v1"
VLLM_API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
VLLM_MODEL_NAME = "Qwen3-VL-2B-Instruct"

# GPT 评分 API 配置（使用 Azure OpenAI）
GPT_API_KEY = ""
GPT_API_BASE = ""
GPT_ENGINE = "gpt-4o-mini"

# 默认文件路径
DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/VStream-QA/movienet_frames"
DEFAULT_JSON_PATH = "/data/tempuser2/MMAgent/MM-Mem/Baseline/VStream-QA/Dataset/test_qa_movienet copy.json"

# 并行化配置
MAX_CONCURRENT_REQUESTS = 1  # 最大并发请求数
SEMAPHORE = None  # 信号量，控制并发数

# 图片采样配置
MAX_FRAMES = 128  # 最大帧数（如果帧数超过此值则均匀采样）

# 图片分辨率配置
IMAGE_SIZE = (224, 224)  # 图片缩放目标尺寸 (width, height)，设为 None 则不缩放

# 帧范围控制
USE_START_TIME = False  # True: 从 start_time 到 end_time; False: 从第一帧到 end_time

# --- 2. 核心功能函数 ---

def load_questions(json_path: str) -> List[dict]:
    """
    加载问题 JSON 文件。
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        list: 问题列表
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"加载问题文件: {json_path} ({len(data)} 个问题)")
            return data
    except json.JSONDecodeError as e:
        print(f"解析 JSON 文件时出错 [{json_path}]: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"读取文件时出错 [{json_path}]: {e}", file=sys.stderr)
        return []


def parse_frame_name(frame_name: str) -> Tuple[int, int]:
    """
    解析帧文件名，提取 shot 和 img 编号。
    
    例如: shot_0387_img_0.jpg -> (387, 0)
    
    Args:
        frame_name: 帧文件名
        
    Returns:
        tuple: (shot_number, img_number)
    """
    match = re.match(r'shot_(\d+)_img_(\d+)\.jpg', frame_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def get_frames_in_range(video_dir: str, video_id: str, start_time: str, end_time: str, use_start_time: bool = False) -> List[str]:
    """
    获取指定范围内的所有帧文件路径。
    
    Args:
        video_dir: 视频帧存储根目录
        video_id: 视频 ID，对应子文件夹名
        start_time: 起始帧文件名（如 "shot_0387_img_0.jpg"）
        end_time: 结束帧文件名（如 "shot_0472_img_2.jpg"）
        use_start_time: True 则从 start_time 开始，False 则从第一帧开始
        
    Returns:
        list: 帧文件路径列表（按顺序排列）
    """
    video_folder = os.path.join(video_dir, video_id)
    
    if not os.path.isdir(video_folder):
        print(f"警告：视频文件夹不存在: {video_folder}", file=sys.stderr)
        return []
    
    # 获取所有 jpg 文件
    all_frames = glob.glob(os.path.join(video_folder, "shot_*_img_*.jpg"))
    
    if not all_frames:
        print(f"警告：未找到帧文件: {video_folder}", file=sys.stderr)
        return []
    
    # 按 shot 和 img 编号排序
    def sort_key(path):
        filename = os.path.basename(path)
        shot, img = parse_frame_name(filename)
        return (shot, img)
    
    all_frames_sorted = sorted(all_frames, key=sort_key)
    
    # 解析 start_time 和 end_time
    start_shot, start_img = parse_frame_name(start_time) if use_start_time else (-1, -1)
    end_shot, end_img = parse_frame_name(end_time)
    
    if end_shot == -1:
        print(f"警告：无法解析 end_time: {end_time}", file=sys.stderr)
        return all_frames_sorted  # 返回所有帧
    
    if use_start_time and start_shot == -1:
        print(f"警告：无法解析 start_time: {start_time}，将从第一帧开始", file=sys.stderr)
        use_start_time = False
    
    # 筛选帧范围
    frames_in_range = []
    for frame_path in all_frames_sorted:
        filename = os.path.basename(frame_path)
        shot, img = parse_frame_name(filename)
        
        # 检查起始边界
        if use_start_time:
            if shot < start_shot or (shot == start_shot and img < start_img):
                continue  # 还没到起始帧
        
        # 检查结束边界（包含 end_time）
        if shot < end_shot or (shot == end_shot and img <= end_img):
            frames_in_range.append(frame_path)
        elif shot > end_shot or (shot == end_shot and img > end_img):
            break  # 超过结束帧，停止
    
    return frames_in_range


def sample_frames(frame_paths: List[str], max_frames: int = MAX_FRAMES) -> List[str]:
    """
    如果帧数超过 max_frames，则均匀采样。
    
    Args:
        frame_paths: 帧文件路径列表
        max_frames: 最大帧数
        
    Returns:
        list: 采样后的帧文件路径列表
    """
    if len(frame_paths) <= max_frames:
        return frame_paths
    
    # 均匀采样
    import numpy as np
    indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int).tolist()
    return [frame_paths[i] for i in indices]


def load_frames_as_base64(frame_paths: List[str], image_size: Tuple[int, int] = None) -> List[str]:
    """
    将帧图片加载、缩放并转换为 base64 编码。
    
    Args:
        frame_paths: 帧文件路径列表
        image_size: 目标图片尺寸 (width, height)，如 (224, 224)。设为 None 则不缩放
        
    Returns:
        list: base64 编码的图片列表
    """
    frames_base64 = []
    for frame_path in frame_paths:
        try:
            if image_size:
                # 使用 PIL 加载、缩放并编码
                with Image.open(frame_path) as img:
                    # 转换为 RGB（避免 RGBA 等格式问题）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # 缩放图片
                    img_resized = img.resize(image_size, Image.Resampling.LANCZOS)
                    # 保存到内存缓冲区
                    buffer = io.BytesIO()
                    img_resized.save(buffer, format='JPEG', quality=85)
                    image_data = buffer.getvalue()
            else:
                # 直接读取原图
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
            
            frame_base64 = base64.b64encode(image_data).decode('utf-8')
            frames_base64.append(frame_base64)
        except Exception as e:
            print(f"读取图片失败 [{frame_path}]: {e}", file=sys.stderr)
    
    return frames_base64


async def ask_question_async(
    client: AsyncOpenAI,
    frames_base64: List[str],
    question: str,
    task_info: dict
):
    """
    异步版本：使用 vLLM API 传入帧图片进行开放式问答。
    
    Args:
        client: AsyncOpenAI 客户端
        frames_base64: base64 编码的帧图片列表
        question: 问题文本
        task_info: 任务信息，用于结果记录
        
    Returns:
        dict: 包含结果的字典
    """
    global SEMAPHORE
    
    # 构建提示词
    prompt = (
        "Answer the following open-ended question based on the video. Please watch the video carefully and answer the following question based on the video content.\n"
        f"Question: {question}\n"
        # "Please provide a detailed and accurate answer."
    )
    
    # 构建消息内容
    content = []
    
    # 添加帧图片
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
                model=VLLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video analyzer. Your task is to watch the video frames provided and answer questions about the video content accurately and in detail."
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=256,
                temperature=0,
            )
            
            model_answer = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'question_id': task_info['question_id'],
                'video_id': task_info['video_id'],
                'question': question,
                'ground_truth': task_info['ground_truth'],
                'answer_type': task_info['answer_type'],
                'model_prediction': model_answer,
                'num_frames': len(frames_base64),
            }
            
        except Exception as e:
            print(f"VQA请求失败 [question_id={task_info['question_id']}]: {e}", file=sys.stderr)
            return {
                'success': False,
                'question_id': task_info['question_id'],
                'video_id': task_info['video_id'],
                'question': question,
                'ground_truth': task_info['ground_truth'],
                'answer_type': task_info['answer_type'],
                'model_prediction': None,
                'num_frames': len(frames_base64),
                'error': str(e),
            }


async def evaluate_answer_with_gpt_async(
    client: AsyncOpenAI,
    question: str,
    answer: str,
    pred: str,
    task_info: dict
):
    """
    使用 GPT 模型评估预测答案与正确答案的匹配程度。
    
    Args:
        client: AsyncOpenAI 客户端（用于 GPT API）
        question: 问题
        answer: 正确答案
        pred: 模型预测答案
        task_info: 任务信息
        
    Returns:
        dict: 评估结果
    """
    try:
        response = await client.chat.completions.create(
            model=GPT_ENGINE,
            messages=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
                }
            ],
            temperature=0.002,
            max_tokens=50,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 解析返回的字典字符串
        try:
            # 尝试使用 ast.literal_eval 解析
            eval_result = ast.literal_eval(result_text)
            is_correct = eval_result.get('pred', 'no').lower() == 'yes'
            score = int(eval_result.get('score', 0))
        except:
            # 如果解析失败，尝试正则提取
            pred_match = re.search(r"'pred'\s*:\s*'(\w+)'", result_text)
            score_match = re.search(r"'score'\s*:\s*(\d+)", result_text)
            is_correct = pred_match.group(1).lower() == 'yes' if pred_match else False
            score = int(score_match.group(1)) if score_match else 0
        
        return {
            'question_id': task_info['question_id'],
            'is_correct': is_correct,
            'score': score,
            'gpt_response': result_text,
        }
        
    except Exception as e:
        print(f"GPT评分失败 [question_id={task_info['question_id']}]: {e}", file=sys.stderr)
        return {
            'question_id': task_info['question_id'],
            'is_correct': False,
            'score': 0,
            'error': str(e),
        }


async def process_vqa_tasks(tasks: list, vllm_client: AsyncOpenAI):
    """
    并行处理所有 VQA 任务。
    
    Args:
        tasks: 任务列表
        vllm_client: vLLM AsyncOpenAI 客户端
        
    Returns:
        list: 所有 VQA 结果
    """
    async_tasks = []
    for task in tasks:
        async_task = ask_question_async(
            vllm_client,
            task['frames_base64'],
            task['question'],
            task['task_info']
        )
        async_tasks.append(async_task)
    
    print(f"\n开始并行处理 {len(async_tasks)} 个 VQA 任务（最大并发数: {MAX_CONCURRENT_REQUESTS}）...")
    
    results = []
    completed = 0
    total_tasks = len(async_tasks)
    progress_width = len(str(total_tasks))
    
    for coro in asyncio.as_completed(async_tasks):
        result = await coro
        completed += 1
        if result.get('success'):
            print(f"[{completed:>{progress_width}}/{total_tasks}] ✓ "
                  f"Q_ID={result['question_id']:<8} "
                  f"帧数={result['num_frames']:<4} "
                  f"问题: {result['question'][:40]}...")
        else:
            print(f"[{completed:>{progress_width}}/{total_tasks}] ⚠ "
                  f"Q_ID={result['question_id']:<8} 失败")
        results.append(result)
    
    return results


async def process_evaluation_tasks(vqa_results: list, gpt_client: AsyncOpenAI, eval_semaphore: asyncio.Semaphore):
    """
    并行处理所有 GPT 评分任务。
    
    Args:
        vqa_results: VQA 结果列表
        gpt_client: GPT AsyncOpenAI 客户端
        eval_semaphore: 评分任务信号量
        
    Returns:
        list: 所有评分结果
    """
    async def evaluate_with_semaphore(result):
        async with eval_semaphore:
            return await evaluate_answer_with_gpt_async(
                gpt_client,
                result['question'],
                result['ground_truth'],
                result['model_prediction'],
                {'question_id': result['question_id']}
            )
    
    successful_results = [r for r in vqa_results if r.get('success') and r.get('model_prediction')]
    
    print(f"\n开始 GPT 评分（共 {len(successful_results)} 个）...")
    
    async_tasks = [evaluate_with_semaphore(r) for r in successful_results]
    
    eval_results = []
    completed = 0
    total_tasks = len(async_tasks)
    progress_width = len(str(total_tasks))
    
    for coro in asyncio.as_completed(async_tasks):
        result = await coro
        completed += 1
        status = "✓" if result.get('is_correct') else "✗"
        score = result.get('score', 0)
        print(f"[{completed:>{progress_width}}/{total_tasks}] {status} "
              f"Q_ID={result['question_id']:<8} 得分={score}/5")
        eval_results.append(result)
    
    return eval_results


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="VStream-QA MovieNet 数据集视频问答测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Async_API_answer_HD_EPIC_questions.py
  python Async_API_answer_HD_EPIC_questions.py -j ./test_qa.json -v ./Video/movienet -c 4
        """
    )
    
    parser.add_argument(
        "-j", "--json-path",
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f"问题 JSON 文件路径 (默认: {DEFAULT_JSON_PATH})"
    )
    
    parser.add_argument(
        "-v", "--video-dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"视频帧文件夹路径 (默认: {DEFAULT_VIDEO_DIR})"
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
        help="输出结果文件路径 (默认: 当前目录下的 movienet_vqa_results.json)"
    )
    
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=VLLM_BASE_URL,
        help=f"vLLM API 基础 URL (默认: {VLLM_BASE_URL})"
    )
    
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=VLLM_MODEL_NAME,
        help=f"vLLM 模型名称 (默认: {VLLM_MODEL_NAME})"
    )
    
    parser.add_argument(
        "-n", "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help=f"最大帧数 (默认: {MAX_FRAMES})"
    )
    
    parser.add_argument(
        "--gpt-key",
        type=str,
        default=GPT_API_KEY,
        help="GPT API 密钥"
    )
    
    parser.add_argument(
        "--gpt-base",
        type=str,
        default=GPT_API_BASE,
        help=f"GPT API 基础 URL (默认: {GPT_API_BASE})"
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过 GPT 评分步骤"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的问题数量（用于测试）"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=list(IMAGE_SIZE) if IMAGE_SIZE else None,
        metavar=("WIDTH", "HEIGHT"),
        help=f"图片缩放尺寸 (默认: {IMAGE_SIZE})，设为 0 0 则不缩放"
    )
    
    parser.add_argument(
        "--use-start-time",
        action="store_true",
        default=USE_START_TIME,
        help="从 start_time 开始读取帧，而非从第一帧开始 (默认: False)"
    )
    
    return parser.parse_args()


# --- 3. 主程序 ---

async def main_async():
    """
    异步主执行函数：并行处理所有视频问答任务。
    """
    global SEMAPHORE, VLLM_BASE_URL, VLLM_MODEL_NAME, MAX_CONCURRENT_REQUESTS, MAX_FRAMES
    global GPT_API_KEY, GPT_API_BASE, GPT_ENGINE, IMAGE_SIZE, USE_START_TIME
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新全局配置
    VLLM_BASE_URL = args.vllm_url
    VLLM_MODEL_NAME = args.vllm_model
    MAX_CONCURRENT_REQUESTS = args.concurrency
    MAX_FRAMES = args.max_frames
    GPT_API_KEY = args.gpt_key
    GPT_API_BASE = args.gpt_base
    USE_START_TIME = args.use_start_time
    
    # 处理图片尺寸参数
    if args.image_size and args.image_size[0] > 0 and args.image_size[1] > 0:
        IMAGE_SIZE = tuple(args.image_size)
    else:
        IMAGE_SIZE = None  # 不缩放
    
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    json_path = args.json_path
    video_dir = args.video_dir
    output_path = args.output or os.path.join(os.getcwd(), 'movienet_vqa_results.json')
    
    timer_start = time.time()
    
    print("=" * 70)
    print("VStream-QA MovieNet 数据集视频问答测试")
    print("=" * 70)
    print(f"问题 JSON 文件: {json_path}")
    print(f"视频帧文件夹: {video_dir}")
    print(f"最大帧数: {MAX_FRAMES}")
    print(f"图片尺寸: {IMAGE_SIZE if IMAGE_SIZE else '原始尺寸'}")
    print(f"帧范围: {'start_time -> end_time' if USE_START_TIME else '第一帧 -> end_time'}")
    print(f"最大并发数: {MAX_CONCURRENT_REQUESTS}")
    print(f"vLLM API 地址: {VLLM_BASE_URL}")
    print(f"vLLM 模型名称: {VLLM_MODEL_NAME}")
    print(f"跳过 GPT 评分: {args.skip_eval}")
    print("=" * 70)
    
    # 步骤 1: 检查路径有效性
    if not os.path.isfile(json_path):
        print(f"错误：问题 JSON 文件不存在: {json_path}", file=sys.stderr)
        return
    
    if not os.path.isdir(video_dir):
        print(f"错误：视频帧文件夹不存在: {video_dir}", file=sys.stderr)
        return
    
    # 步骤 2: 加载问题文件
    print("\n[步骤 1] 加载问题数据...")
    questions = load_questions(json_path)
    if not questions:
        print("错误：未能加载任何问题", file=sys.stderr)
        return
    
    # 如果设置了限制数量
    if args.limit:
        questions = questions[:args.limit]
        print(f"  限制处理前 {args.limit} 个问题")
    
    # 步骤 3: 初始化 vLLM 客户端
    print(f"\n[步骤 2] 连接 vLLM 服务: {VLLM_BASE_URL}")
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
    )
    print("vLLM 客户端初始化成功！")
    
    # 步骤 4: 构建所有任务
    print("\n[步骤 3] 构建问答任务...")
    tasks = []
    skipped_no_frames = 0
    
    for q in questions:
        video_id = q.get("video_id", "")
        question = q.get("question", "")
        answer = q.get("answer", "")
        question_id = q.get("id", "")
        answer_type = q.get("answer_type", "")
        start_time = q.get("start_time", "")
        end_time = q.get("end_time", "")
        
        if not video_id or not question:
            print(f"  跳过无效问题: {question_id}", file=sys.stderr)
            continue
        
        # 获取帧文件
        frame_paths = get_frames_in_range(video_dir, video_id, start_time, end_time, USE_START_TIME)
        
        if not frame_paths:
            skipped_no_frames += 1
            if skipped_no_frames <= 5:
                print(f"  跳过（未找到帧）: video_id={video_id}", file=sys.stderr)
            continue
        
        # 采样帧
        sampled_frames = sample_frames(frame_paths, MAX_FRAMES)
        
        # 加载帧图片为 base64（支持缩放）
        frames_base64 = load_frames_as_base64(sampled_frames, IMAGE_SIZE)
        
        if not frames_base64:
            skipped_no_frames += 1
            continue
        
        tasks.append({
            'frames_base64': frames_base64,
            'question': question,
            'task_info': {
                'question_id': question_id,
                'video_id': video_id,
                'ground_truth': answer,
                'answer_type': answer_type,
            }
        })
    
    if skipped_no_frames > 5:
        print(f"  ... 还有 {skipped_no_frames - 5} 个问题因未找到帧而跳过", file=sys.stderr)
    
    if not tasks:
        print("没有找到有效的问答任务。")
        return
    
    print(f"\n共构建 {len(tasks)} 个问答任务（跳过 {skipped_no_frames} 个因未找到帧）")
    
    # 步骤 5: 并行处理 VQA 任务
    print("\n[步骤 4] 开始 VQA 问答...")
    vqa_results = await process_vqa_tasks(tasks, vllm_client)
    
    # 步骤 6: GPT 评分（如果没有跳过）
    eval_results = []
    if not args.skip_eval:
        print(f"\n[步骤 5] 初始化 GPT 评分客户端...")
        try:
            # 使用 OpenAI 兼容的客户端（可以是 Azure OpenAI）
            gpt_client = AsyncOpenAI(
                api_key=GPT_API_KEY,
                base_url=GPT_API_BASE if GPT_API_BASE else None,
            )
            
            eval_semaphore = asyncio.Semaphore(5)  # GPT 评分限制并发数
            eval_results = await process_evaluation_tasks(vqa_results, gpt_client, eval_semaphore)
        except Exception as e:
            print(f"GPT 客户端初始化失败: {e}", file=sys.stderr)
            print("跳过 GPT 评分步骤")
    
    # 步骤 7: 合并结果并统计
    # 创建评分结果映射
    eval_map = {r['question_id']: r for r in eval_results}
    
    # 合并 VQA 结果和评分结果
    final_results = []
    total_correct = 0
    total_score = 0
    total_evaluated = 0
    
    results_by_type = {}  # 按 answer_type 分类统计
    
    for vqa_result in vqa_results:
        if not vqa_result.get('success'):
            continue
        
        question_id = vqa_result['question_id']
        answer_type = vqa_result.get('answer_type', 'Unknown')
        
        # 获取评分结果
        eval_result = eval_map.get(question_id, {})
        is_correct = eval_result.get('is_correct', False)
        score = eval_result.get('score', 0)
        
        # 按类型统计
        if answer_type not in results_by_type:
            results_by_type[answer_type] = {'correct': 0, 'total': 0, 'score_sum': 0}
        
        results_by_type[answer_type]['total'] += 1
        
        if eval_result:
            total_evaluated += 1
            results_by_type[answer_type]['score_sum'] += score
            if is_correct:
                total_correct += 1
                results_by_type[answer_type]['correct'] += 1
            total_score += score
        
        # 组合最终结果
        final_result = {
            **vqa_result,
            'is_correct': is_correct,
            'score': score,
            'gpt_response': eval_result.get('gpt_response', ''),
        }
        final_results.append(final_result)
    
    # 步骤 8: 打印最终结果
    elapsed_time = time.time() - timer_start
    print(f"\n\n{'='*70}")
    print("最终结果")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    total_questions = len(final_results)
    if total_questions > 0:
        print(f"\n总问题数: {total_questions}")
        print(f"成功 VQA 问答: {len([r for r in vqa_results if r.get('success')])}")
        
        if total_evaluated > 0:
            accuracy = (total_correct / total_evaluated) * 100
            avg_score = total_score / total_evaluated
            
            # 先输出按问题类型分类的统计
            print("\n" + "=" * 70)
            print("按问题类型 (answer_type) 分类统计")
            print("=" * 70)
            print(f"{'问题类型':<25} {'准确率':>10} {'正确/总数':>12} {'平均分':>10}")
            print("-" * 70)
            for answer_type, stats in sorted(results_by_type.items()):
                correct = stats['correct']
                total = stats['total']
                score_sum = stats['score_sum']
                if total > 0:
                    type_accuracy = (correct / total) * 100 if total else 0
                    type_avg_score = score_sum / total if total else 0
                    print(f"{answer_type:<25} {type_accuracy:>9.2f}% {correct:>5}/{total:<5} {type_avg_score:>9.2f}")
            print("-" * 70)
            print(f"{'总计':<25} {accuracy:>9.2f}% {total_correct:>5}/{total_evaluated:<5} {avg_score:>9.2f}")
            print("=" * 70)
            
            # 再输出整体统计信息
            print(f"\nGPT 评分结果汇总:")
            print(f"  评估问题数: {total_evaluated}")
            print(f"  正确回答数: {total_correct}")
            print(f"  总体准确率: {accuracy:.2f}%")
            print(f"  总体平均得分: {avg_score:.2f}/5")
        else:
            print("\n未进行 GPT 评分。")
            
            # 即使没有评分，也输出按类型的问题分布
            print("\n" + "=" * 70)
            print("按问题类型 (answer_type) 分布")
            print("=" * 70)
            print(f"{'问题类型':<25} {'问题数':>10}")
            print("-" * 50)
            for answer_type, stats in sorted(results_by_type.items()):
                total = stats['total']
                print(f"{answer_type:<25} {total:>10}")
            print("-" * 50)
            print(f"{'总计':<25} {total_questions:>10}")
        
        print(f"\n平均每题耗时: {elapsed_time/total_questions:.2f} 秒")
    else:
        print("\n未处理任何问题。")
    
    # 步骤 9: 保存所有结果
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        output_data = {
            'summary': {
                'total_questions': total_questions,
                'total_evaluated': total_evaluated,
                'total_correct': total_correct,
                'accuracy': (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0,
                'average_score': (total_score / total_evaluated) if total_evaluated > 0 else 0,
                'elapsed_time_seconds': elapsed_time,
                'json_path': json_path,
                'video_dir': video_dir,
                'vllm_model': VLLM_MODEL_NAME,
                'gpt_engine': GPT_ENGINE,
            },
            'results_by_type': results_by_type,
            'detailed_results': final_results,
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
