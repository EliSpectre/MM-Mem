"""
使用 vLLM 部署的 OpenAI 兼容 API 进行 MLVU 数据集视频问答
并行化版本：使用 asyncio 异步并发请求加速处理
支持按 question_type 分类统计准确率

数据集结构：
- JSON_DIR: 包含多个分类好的 JSON 文件（如 1_plotQA.json, 2_needle.json）
- VIDEO_DIR: 包含对应的视频文件夹（如 1_plotQA/, 2_needle/）
- JSON 文件名与视频文件夹名相同（不含扩展名）
"""
import json
import os
import sys
import asyncio
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time
import glob

# --- 1. 配置 ---

# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen3-VL-2B-Instruct"

# 文件路径（配置）- MLVU 数据集
# JSON 文件目录（包含按类别分好的 JSON 文件）
JSON_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/MLVU/Dataset/Image"
# 视频目录（包含与 JSON 文件同名的子文件夹）
VIDEO_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/MLVU/Dataset/Dev"

# 并行化配置
MAX_CONCURRENT_REQUESTS = 1  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
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

async def process_single_category(category_name, json_path, video_folder, client):
    """
    处理单个类别的问答任务。
    
    Args:
        category_name: 类别名称（如 "1_plotQA"）
        json_path: JSON 文件路径
        video_folder: 对应的视频文件夹路径
        client: AsyncOpenAI 客户端
        
    Returns:
        tuple: (类别名, 结果列表, 正确数, 总数)
    """
    print(f"\n{'='*60}")
    print(f"处理类别: {category_name}")
    print(f"{'='*60}")
    
    # 加载问题
    all_questions = load_questions(json_path)
    if not all_questions:
        print(f"  警告：无法从 {json_path} 加载问题，跳过此类别。")
        return category_name, [], 0, 0
    
    print(f"  加载了 {len(all_questions)} 个问题")
    
    # 建立 video -> questions 的映射
    vid_map = {}
    for q in all_questions:
        vid = q.get("video")
        if vid:
            vid_map.setdefault(vid, []).append(q)
    
    # 检查视频文件夹是否存在
    if not os.path.isdir(video_folder):
        print(f"  警告：视频目录不存在: {video_folder}，跳过此类别。")
        return category_name, [], 0, 0
    
    # 查找该文件夹下的所有 mp4 文件
    mp4_files = []
    for f in os.listdir(video_folder):
        if f.lower().endswith('.mp4'):
            mp4_files.append(os.path.join(video_folder, f))
    
    if not mp4_files:
        print(f"  警告：在 {video_folder} 中未找到 mp4 文件，跳过此类别。")
        return category_name, [], 0, 0
    
    print(f"  找到 {len(mp4_files)} 个视频文件")
    
    # 构建任务
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
            candidates = q_data.get("candidates", [])
            ground_truth_answer = q_data.get("answer", "")
            question_type = q_data.get("question_type", category_name)
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
                    'category': category_name,
                    'duration': duration,
                    'ground_truth': ground_truth_answer,
                    'candidates': candidates,
                }
            })
    
    if not tasks:
        print(f"  警告：没有找到匹配的视频-问题对，跳过此类别。")
        return category_name, [], 0, 0
    
    print(f"  共构建 {len(tasks)} 个问答任务")
    
    # 并行处理任务
    results = await process_all_tasks(tasks, client)
    
    # 统计正确数
    correct = sum(1 for r in results if r.get('success') and r.get('is_correct'))
    total = sum(1 for r in results if r.get('success'))
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n  类别 [{category_name}] 完成: 准确率 {accuracy:.2f}% ({correct}/{total})")
    
    return category_name, results, correct, total


async def main_async():
    """
    异步主执行函数：逐个处理各类别的 MLVU 数据集视频问答任务。
    """
    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    start_time = time.time()
    
    # 步骤 1: 扫描 JSON 目录，获取所有 JSON 文件
    if not os.path.isdir(JSON_DIR):
        print(f"错误：JSON 目录不存在: {JSON_DIR}", file=sys.stderr)
        return
    
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if not json_files:
        print(f"错误：在 {JSON_DIR} 中未找到 JSON 文件。", file=sys.stderr)
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件:")
    for jf in json_files:
        print(f"  - {os.path.basename(jf)}")
    
    # 步骤 2: 初始化异步 OpenAI 客户端
    print(f"\n正在连接 vLLM 服务: {BASE_URL}")
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    print("异步客户端初始化成功！")
    
    # 步骤 3: 逐个处理每个 JSON 文件（每个文件对应一个类别）
    all_results = []
    category_stats = {}  # 记录每个类别的统计信息
    
    for json_path in json_files:
        json_filename = os.path.basename(json_path)
        category_name = os.path.splitext(json_filename)[0]  # 如 "1_plotQA"
        
        # 对应的视频文件夹
        video_folder = os.path.join(VIDEO_DIR, category_name)
        
        # 处理该类别
        cat_name, results, correct, total = await process_single_category(
            category_name, json_path, video_folder, client
        )
        
        all_results.extend(results)
        category_stats[cat_name] = {
            'correct': correct,
            'total': total,
            'accuracy': (correct / total * 100) if total > 0 else 0
        }
    
    # 步骤 4: 打印最终汇总结果
    elapsed_time = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"MLVU 数据集 VQA 测试结果汇总")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    total_questions = sum(s['total'] for s in category_stats.values())
    total_correct = sum(s['correct'] for s in category_stats.values())
    
    if total_questions > 0:
        overall_accuracy = (total_correct / total_questions) * 100
        print(f"\n--- 总体统计 ---")
        print(f"总问题数: {total_questions}")
        print(f"正确回答数: {total_correct}")
        print(f"总准确率: {overall_accuracy:.2f}%")
        print(f"平均每题耗时: {elapsed_time/total_questions:.2f} 秒")
        
        print(f"\n--- 按类别（JSON文件）分类准确率 ---")
        print(f"{'Category':<30} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
        print("-" * 70)
        for cat_name in sorted(category_stats.keys()):
            stats = category_stats[cat_name]
            print(f"{cat_name:<30} {stats['accuracy']:>6.2f}%      {stats['correct']:<10} {stats['total']:<10}")
        print("-" * 70)
        print(f"{'Overall':<30} {overall_accuracy:>6.2f}%      {total_correct:<10} {total_questions:<10}")
    else:
        print("\n未处理任何问题，无法计算准确率。")
    
    # 步骤 5: 保存所有结果
    successful_results = [r for r in all_results if r.get('success')]
    out_path = os.path.join(os.getcwd(), MODEL_NAME, 'mlvu_vqa_results.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(successful_results, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！所有结果已保存到: {out_path}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}", file=sys.stderr)
    
    # 保存按类别分类的准确率摘要
    summary_path = os.path.join(os.getcwd(), MODEL_NAME, 'mlvu_accuracy_by_category.json')
    try:
        summary = {
            'total_questions': total_questions,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy if total_questions > 0 else 0,
            'elapsed_time_seconds': elapsed_time,
            'accuracy_by_category': category_stats
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
