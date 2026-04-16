"""
使用 vLLM 部署的 OpenAI 兼容 API 进行 MLVU 数据集视频问答
并行化版本：使用 asyncio 异步并发请求加速处理
支持按 question_type 分类统计准确率
"""
import json
import os
import sys
import asyncio
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time

# --- 1. 配置 ---

# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen3-VL-2B-Instruct"

# 文件路径（配置）- MLVU 数据集
VIDEO_DIR = "/data/tempuser2/MMAgent/MLVU/MLVU_Test/video"
JSON_PATH = "/data/tempuser2/MMAgent/MM-Mem/Baseline/MLVU/Dataset/test_mcq_gt.json"

# 并行化配置
MAX_CONCURRENT_REQUESTS = 15  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
SEMAPHORE = None  # 信号量，控制并发数

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
    """从答案字符串中提取唯一的字母选项（例如 "A." -> "A"），或返回原始答案用于直接比较。"""
    if not isinstance(answer_str, str):
        return ""
    # 首先尝试提取字母选项
    stripped = answer_str.strip()
    for char in stripped:
        if char.isalpha() and len(stripped) <= 3:  # 只有短答案才提取字母
            return char.upper()
    # 对于MLVU数据集，答案可能是完整的文本（如 "RoadAccidents"）
    return stripped

def normalize_answer_for_comparison(model_answer, ground_truth, candidates):
    """
    针对MLVU数据集的答案比较。
    模型可能返回选项字母或完整答案文本。
    支持整数类型的答案（如 count 类型问题）。
    """
    # 转换为字符串处理
    model_answer = str(model_answer).strip() if model_answer is not None else ""
    ground_truth = str(ground_truth).strip() if ground_truth is not None else ""
    # 将 candidates 转换为字符串列表
    candidates_str = [str(c).strip() for c in candidates]
    
    # 直接比较（忽略大小写）
    if model_answer.lower() == ground_truth.lower():
        return True
    
    # 如果模型返回的是选项索引（A, B, C, D, E, F...）
    if len(model_answer) == 1 and model_answer.isalpha():
        idx = ord(model_answer.upper()) - ord('A')
        if 0 <= idx < len(candidates_str):
            if candidates_str[idx].lower() == ground_truth.lower():
                return True
    
    # 检查模型答案是否包含正确答案（模糊匹配）
    if ground_truth.lower() in model_answer.lower():
        return True
        
    return False

async def ask_question_with_video_async(client, video_path, question_text, candidates, task_info):
    """
    异步版本：使用 vLLM API 直接传入视频路径进行问答。
    
    Args:
        client: AsyncOpenAI 客户端
        video_path (str): 视频文件路径
        question_text (str): 问题文本
        candidates (list): 候选答案列表
        task_info (dict): 任务信息，用于结果记录
        
    Returns:
        dict: 包含结果的字典
    """
    global SEMAPHORE
    
    # 构建选项字符串（MLVU格式：将candidates转换为A, B, C...选项）
    # 注意：candidates 可能是整数（如 count 类型问题），需要转换为字符串
    options_str = "\n".join([f"{chr(65+i)}. {str(opt)}" for i, opt in enumerate(candidates)])
    prompt = (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, D, E, or F) of the correct option.\n"
        f"Question: {question_text}\n"
        f"Possible answer choices:\n{options_str}\n"
        "The best answer is:"
    )
    
    # 使用信号量控制并发数
    async with SEMAPHORE:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"file://{video_path}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0,
            )
            
            model_answer_raw = response.choices[0].message.content.strip()
            ground_truth = task_info['ground_truth']
            candidates_list = task_info['candidates']
            
            is_correct = normalize_answer_for_comparison(model_answer_raw, ground_truth, candidates_list)
            
            return {
                'success': True,
                'video_file': task_info['video_file'],
                'video_id': task_info['video_id'],
                'question_id': task_info['question_id'],
                'question_type': task_info['question_type'],
                'duration': task_info['duration'],
                'question': question_text,
                'candidates': candidates,
                'ground_truth': ground_truth,
                'model_prediction': model_answer_raw,
                'is_correct': is_correct,
            }
            
        except Exception as e:
            print(f"请求失败 [video={task_info['video_id']}]: {e}", file=sys.stderr)
            return {
                'success': False,
                'video_file': task_info['video_file'],
                'video_id': task_info['video_id'],
                'question_id': task_info['question_id'],
                'question_type': task_info['question_type'],
                'duration': task_info['duration'],
                'question': question_text,
                'candidates': candidates,
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
            task['candidates'],
            task['task_info']
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
            q_type = result.get('question_type', 'unknown')
            # 格式化输出，保证对齐（转换为字符串处理）
            model_pred = str(result['model_prediction'])[:15] if result['model_prediction'] is not None else "None"
            gt = str(result['ground_truth'])[:15]
            print(f"[{completed:>{progress_width}}/{total_tasks}] {status} "
                  f"video={result['video_id']:<25} type={q_type:<15} "
                  f"预测={model_pred:<15} 答案={gt:<15} "
                  f"{prefix}正确={str(is_correct):<5}")
        else:
            print(f"[{completed:>{progress_width}}/{total_tasks}] ⚠ "
                  f"video={result['video_id']:<25} 失败")
        results.append(result)
    
    return results

# --- 3. 主程序 ---

async def main_async():
    """
    异步主执行函数：并行处理所有 MLVU 数据集视频问答任务。
    """
    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    start_time = time.time()
    
    # 步骤 1: 加载所有问题并建立 video -> questions 的映射
    all_questions = load_questions(JSON_PATH)
    if not all_questions:
        print(f"错误：无法从 {JSON_PATH} 加载问题。", file=sys.stderr)
        return

    # MLVU 数据集使用 "video" 字段作为视频文件名
    vid_map = {}
    for q in all_questions:
        vid = q.get("video")  # MLVU 格式使用 "video" 字段
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

    # 步骤 4: 构建所有任务
    tasks = []
    for video_path in sorted(mp4_files):
        mp4_file = os.path.basename(video_path)
        video_id = os.path.splitext(mp4_file)[0]
        
        # 使用完整文件名匹配（包括.mp4后缀）
        questions_for_video = vid_map.get(mp4_file)
        if not questions_for_video:
            continue

        for q_data in questions_for_video:
            duration = q_data.get("duration", 0)
            question_text = q_data.get("question", "No question text found.")
            candidates = q_data.get("candidates", [])  # MLVU 使用 "candidates"
            ground_truth_answer = q_data.get("answer", "")
            question_type = q_data.get("question_type", "unknown")
            question_id = q_data.get("question_id", "")

            tasks.append({
                'video_path': video_path,
                'question_text': question_text,
                'candidates': candidates,
                'task_info': {
                    'video_file': mp4_file,
                    'video_id': video_id,
                    'question_id': question_id,
                    'question_type': question_type,
                    'duration': duration,
                    'ground_truth': ground_truth_answer,
                    'candidates': candidates,
                }
            })

    if not tasks:
        print("没有找到匹配的视频-问题对。")
        return

    print(f"共构建 {len(tasks)} 个问答任务。")
    
    # 统计各 question_type 的任务数
    type_counts = defaultdict(int)
    for task in tasks:
        type_counts[task['task_info']['question_type']] += 1
    print("\n各 question_type 任务数分布:")
    for q_type, count in sorted(type_counts.items()):
        print(f"  - {q_type}: {count}")

    # 步骤 5: 并行处理所有任务
    all_results = await process_all_tasks(tasks, client)

    # 步骤 6: 按 question_type 统计结果
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    
    successful_results = []
    for result in all_results:
        if result.get('success'):
            q_type = result.get('question_type', 'unknown')
            total_by_type[q_type] += 1
            if result['is_correct']:
                correct_by_type[q_type] += 1
            successful_results.append(result)

    # 步骤 7: 打印最终结果
    elapsed_time = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"MLVU 数据集 VQA 测试结果")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    total_questions = sum(total_by_type.values())
    total_correct = sum(correct_by_type.values())

    if total_questions > 0:
        overall_accuracy = (total_correct / total_questions) * 100
        print(f"\n--- 总体统计 ---")
        print(f"总问题数: {total_questions}")
        print(f"正确回答数: {total_correct}")
        print(f"总准确率: {overall_accuracy:.2f}%")
        print(f"平均每题耗时: {elapsed_time/total_questions:.2f} 秒")
        
        print(f"\n--- 按 question_type 分类准确率 ---")
        print(f"{'Question Type':<25} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
        print("-" * 60)
        for q_type in sorted(total_by_type.keys()):
            total = total_by_type[q_type]
            correct = correct_by_type[q_type]
            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"{q_type:<25} {accuracy:>6.2f}%      {correct:<10} {total:<10}")
        print("-" * 60)
        print(f"{'Overall':<25} {overall_accuracy:>6.2f}%      {total_correct:<10} {total_questions:<10}")
    else:
        print("\n未处理任何问题，无法计算准确率。")

    # 步骤 8: 保存所有结果
    out_path = os.path.join(os.getcwd(), MODEL_NAME, 'mlvu_vqa_results.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(successful_results, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！所有结果已保存到: {out_path}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}", file=sys.stderr)
    
    # 保存按 question_type 分类的准确率摘要
    summary_path = os.path.join(os.getcwd(), MODEL_NAME, 'mlvu_accuracy_by_question_type.json')
    try:
        summary = {
            'total_questions': total_questions,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy if total_questions > 0 else 0,
            'elapsed_time_seconds': elapsed_time,
            'accuracy_by_question_type': {}
        }
        for q_type in sorted(total_by_type.keys()):
            total = total_by_type[q_type]
            correct = correct_by_type[q_type]
            accuracy = (correct / total) * 100 if total > 0 else 0
            summary['accuracy_by_question_type'][q_type] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"准确率摘要已保存到: {summary_path}")
    except Exception as e:
        print(f"写入摘要文件时出错: {e}", file=sys.stderr)


def main():
    """同步入口函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
