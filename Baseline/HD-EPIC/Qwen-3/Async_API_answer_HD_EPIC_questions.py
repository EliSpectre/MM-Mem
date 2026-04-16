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
from openai import AsyncOpenAI
import time
from typing import Dict, List, Set, Tuple, Optional
import argparse
from pathlib import Path

# --- 1. 配置 ---
# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen3-VL-2B-Instruct"

# 默认文件路径（可通过命令行参数覆盖）
# DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/Benchmark/VideoPre/rgb_224_1_vig"
DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/Benchmark/PreVideo/fps_1_vig"
# DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/Benchmark/Video"
DEFAULT_JSON_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test"

# 并行化配置
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
SEMAPHORE = None  # 信号量，控制并发数

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


def extract_video_ids_from_inputs(inputs: dict) -> List[str]:
    """
    从 inputs 字典中提取所有唯一的视频/图像 ID。
    
    支持格式：
    - "video 1": {"id": "P01-20240202-110250"}
    - "video 2": {"id": "P01-20240203-152323"}
    - "image 1": {"id": "P01-20240202-110250"}
    
    Args:
        inputs: 问题中的 inputs 字典
        
    Returns:
        list: 去重后的视频ID列表
    """
    video_ids = []
    seen_ids = set()
    
    for key, value in inputs.items():
        # 处理 "video X" 或 "image X" 格式
        if key.startswith("video") or key.startswith("image"):
            if isinstance(value, dict) and "id" in value:
                vid_id = value["id"]
                if vid_id not in seen_ids:
                    video_ids.append(vid_id)
                    seen_ids.add(vid_id)
    
    return video_ids


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


async def ask_question_with_videos_async(
    client,
    video_paths: List[str],
    question_text: str,
    choices: List[str],
    task_info: dict
):
    """
    异步版本：使用 vLLM API 传入一个或多个视频路径进行问答。
    
    Args:
        client: AsyncOpenAI 客户端
        video_paths (list): 视频文件路径列表
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
    
    # 构建消息内容，包含所有视频
    content = []
    for video_path in video_paths:
        content.append({
            "type": "video_url",
            "video_url": {
                "url": f"file://{video_path}",
            },
        })
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
                'video_paths': video_paths,
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
                'video_paths': video_paths,
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
        tasks: 任务列表
        client: AsyncOpenAI 客户端
        
    Returns:
        list: 所有结果
    """
    # 创建所有异步任务
    async_tasks = []
    for task in tasks:
        async_task = ask_question_with_videos_async(
            client,
            task['video_paths'],
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
    
    return parser.parse_args()


# --- 3. 主程序 ---

async def main_async():
    """
    异步主执行函数：并行处理所有视频问答任务。
    """
    global SEMAPHORE, BASE_URL, MODEL_NAME, MAX_CONCURRENT_REQUESTS
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新全局配置
    BASE_URL = args.base_url
    MODEL_NAME = args.model_name
    MAX_CONCURRENT_REQUESTS = args.concurrency
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    json_dir = args.json_dir
    video_dir = args.video_dir
    output_path = args.output or os.path.join(os.getcwd(), MODEL_NAME, 'hd_epic_results.json')
    start_time = time.time()
    
    print("=" * 60)
    print("HD-EPIC 数据集视频问答测试")
    print("=" * 60)
    print(f"JSON 文件夹: {json_dir}")
    print(f"视频文件夹: {video_dir}")
    print(f"最大并发数: {MAX_CONCURRENT_REQUESTS}")
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
            
            # 提取所有视频ID（自动去重）
            video_ids = extract_video_ids_from_inputs(inputs)
            
            if not video_ids:
                print(f"  跳过无视频ID的问题: {question_key}", file=sys.stderr)
                continue
            
            # 查找所有视频路径
            video_paths = []
            missing_videos = []
            for vid_id in video_ids:
                video_path = video_cache.get(vid_id)
                if video_path:
                    video_paths.append(video_path)
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
    
    # 步骤 6: 并行处理所有任务
    print("\n[步骤 5] 开始问答测试...")
    all_results = await process_all_tasks(tasks, client)
    
    # 步骤 7: 统计结果
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
