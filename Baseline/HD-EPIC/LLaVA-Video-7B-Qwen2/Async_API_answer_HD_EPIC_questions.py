"""
使用 LLaVA 官方代码加载 LLaVA-Video-7B-Qwen2 模型进行 HD-EPIC 数据集的视频问答
本地推理版本：直接加载模型进行推理

支持功能：
1. 读取给定文件夹下的所有JSON文件
2. 处理每个JSON文件中的问题
3. 根据问题中的id递归查找对应的视频文件
4. 处理inputs中的多个"video 1"、"video 2"或"image 1"等
5. 如果video和image的id相同，自动去重
6. 按视频ID去重提取帧，优化内存使用
"""

import json
import os
import sys
import glob
import time
import copy
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import warnings
from decord import VideoReader, cpu

# 忽略一些警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# LLaVA 官方代码导入
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

# --- 1. 配置 ---

# 模型路径（可以是本地路径或 HuggingFace 模型名）
DEFAULT_MODEL_PATH = "lmms-lab/LLaVA-Video-7B-Qwen2"

# 默认文件路径（可通过命令行参数覆盖）
DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/Benchmark/VideoPre/rgb_224_1_vig"
DEFAULT_JSON_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test"

# 视频采帧配置
NUM_FRAMES_PER_VIDEO = 32  # LLaVA-Video 建议使用较少帧数（8-32帧）

# 系统提示词
SYSTEM_PROMPT = "You are an expert video analyzer, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc. You must give an answer, even if you are not sure."

# --- 2. 模型加载 ---

def load_model(model_path: str, device: str = "cuda"):
    """
    使用 LLaVA 官方代码加载模型。
    
    Args:
        model_path: 模型路径
        device: 设备（cuda 或 cpu）
        
    Returns:
        tuple: (tokenizer, model, image_processor, max_length)
    """
    print(f"正在加载模型: {model_path}")
    print(f"使用设备: {device}")
    
    model_name = "llava_qwen"
    # 使用 LLaVA 官方的 load_pretrained_model 函数
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        None,  # model_base
        model_name,  # model_name，根据你的模型类型调整
        torch_dtype="float16",
        device_map="auto",
    )
    
    model.eval()
    print(f"模型加载完成！")
    print(f"模型设备: {model.device}")
    print(f"最大长度: {max_length}")
    
    return tokenizer, model, image_processor, max_length


# --- 3. 核心功能函数 ---

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
        0 if x.startswith("video") else 1,
        int(x.split()[-1]) if x.split()[-1].isdigit() else 0
    ))
    
    label_counter = 1
    for key in sorted_keys:
        value = inputs[key]
        if key.startswith("video") or key.startswith("image"):
            if isinstance(value, dict) and "id" in value:
                vid_id = value["id"]
                if vid_id not in seen_ids:
                    video_ids.append(vid_id)
                    video_labels.append(f"video {label_counter}")
                    seen_ids.add(vid_id)
                    label_counter += 1
    
    return video_ids, video_labels


def normalize_answer(answer_str) -> str:
    """从答案字符串中提取唯一的字母选项。"""
    if isinstance(answer_str, int):
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


def load_video(video_path: str, max_frames_num: int = 32, fps: int = 1, force_sample: bool = False):
    """
    使用 decord 从视频中均匀采帧。
    
    Args:
        video_path: 视频文件路径
        max_frames_num: 要提取的最大帧数
        fps: 采帧的 fps
        force_sample: 是否强制均匀采样
        
    Returns:
        tuple: (视频帧 numpy 数组, 帧时间字符串, 视频总时长)
    """
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps_interval = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps_interval)]
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        
        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        return spare_frames, frame_time_str, video_time
    except Exception as e:
        print(f"无法读取视频: {video_path}, 错误: {e}", file=sys.stderr)
        return None, "", 0


def extract_frames_from_video(video_path: str, num_frames: int = 32) -> Tuple[np.ndarray, str, float]:
    """
    从视频中均匀采帧，返回 numpy 数组（RGB格式）。
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        
    Returns:
        tuple: (视频帧数组, 帧时间字符串, 视频总时长)
    """
    return load_video(video_path, num_frames, 1, force_sample=True)


def ask_question_with_video(
    tokenizer,
    model,
    image_processor,
    video_data: Tuple[np.ndarray, str, float],
    question_text: str,
    choices: List[str],
    task_info: dict,
    device: str = "cuda",
) -> dict:
    """
    使用 LLaVA-Video 模型对视频进行问答。
    使用 LLaVA 官方代码格式。
    
    Args:
        tokenizer: LLaVA tokenizer
        model: LLaVA-Video 模型
        image_processor: LLaVA 图像处理器
        video_data: (视频帧数组, 帧时间字符串, 视频时长) 的元组
        question_text: 问题文本
        choices: 选项列表
        task_info: 任务信息
        device: 设备
        
    Returns:
        dict: 包含结果的字典
    """
    # 构建选项字符串
    options_with_letters = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        options_with_letters.append(f"{letter}. {choice}")
    options_str = "\n".join(options_with_letters)
    
    # 构建问题文本
    question_prompt = (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
        f"Question: {question_text}\n"
        f"Possible answer choices:\n{options_str}\n"
        "The best answer is:"
    )
    
    try:
        video_frames, frame_time, video_time = video_data
        
        if video_frames is None:
            raise ValueError("视频帧为空")
        
        # 使用 image_processor 预处理视频帧
        video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].cuda().half()
        video = [video_tensor]
        
        # 构建时间指令
        num_frames = len(video_frames)
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
        
        # 使用 conv_templates 构建对话格式
        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{question_prompt}"
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # 使用 tokenizer_image_token 处理 prompt
        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        
        # 生成回答
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=10,
            )
            
            model_answer_raw = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
        
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
        print(f"推理失败 [question={task_info['question_key']}]: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
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


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="HD-EPIC 数据集视频问答测试工具 (LLaVA-Video)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python transformers_answer_HD_EPIC_questions.py --json-dir ./Dataset --video-dir /path/to/videos
  python transformers_answer_HD_EPIC_questions.py -j ./Dataset -v /data/HD-EPIC/videos -m /path/to/model
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
        "-m", "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"模型路径 (默认: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出结果文件路径 (默认: 当前目录下的 llava_video_results.json)"
    )
    
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=NUM_FRAMES_PER_VIDEO,
        help=f"每个视频均匀采帧数量 (默认: {NUM_FRAMES_PER_VIDEO})"
    )
    
    return parser.parse_args()


# --- 4. 主程序 ---

def main():
    """主执行函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    json_dir = args.json_dir
    video_dir = args.video_dir
    model_path = args.model_path
    num_frames = args.num_frames
    output_path = args.output or os.path.join(os.getcwd(), 'llava_video_results.json')
    
    start_time = time.time()
    
    print("=" * 60)
    print("HD-EPIC 数据集视频问答测试")
    print("模型: LLaVA-Video-7B-Qwen2")
    print("=" * 60)
    print(f"JSON 文件夹: {json_dir}")
    print(f"视频文件夹: {video_dir}")
    print(f"模型路径: {model_path}")
    print(f"每视频采帧数: {num_frames}")
    print("=" * 60)
    
    # 步骤 1: 检查路径有效性
    if not os.path.isdir(json_dir):
        print(f"错误：JSON 文件夹不存在: {json_dir}", file=sys.stderr)
        return
    
    if not os.path.isdir(video_dir):
        print(f"错误：视频文件夹不存在: {video_dir}", file=sys.stderr)
        return
    
    # 步骤 2: 加载模型
    print("\n[步骤 1] 加载模型...")
    tokenizer, model, image_processor, max_length = load_model(model_path)
    
    # 步骤 3: 加载所有 JSON 文件
    print("\n[步骤 2] 加载 JSON 数据集...")
    all_json_data = load_all_json_files(json_dir)
    if not all_json_data:
        print("错误：未能加载任何 JSON 数据", file=sys.stderr)
        return
    
    # 步骤 4: 预先构建视频缓存
    print("\n[步骤 3] 扫描视频文件...")
    video_cache = build_video_cache(video_dir)
    if not video_cache:
        print("警告：未找到任何视频文件", file=sys.stderr)
    
    # 步骤 5: 构建所有任务
    print("\n[步骤 4] 构建问答任务...")
    tasks = []
    skipped_no_video = 0
    
    for json_path, json_content in all_json_data.items():
        json_basename = os.path.basename(json_path)
        
        for question_key, question_data in json_content.items():
            inputs = question_data.get("inputs", {})
            question_text = question_data.get("question", "")
            choices = question_data.get("choices", [])
            correct_idx = question_data.get("correct_idx", -1)
            
            if not question_text or not choices:
                continue
            
            video_ids, video_labels = extract_video_ids_from_inputs(inputs)
            
            if not video_ids:
                continue
            
            # 只取第一个视频（LLaVA-Video 单视频输入）
            primary_video_id = video_ids[0]
            video_path = video_cache.get(primary_video_id)
            
            if not video_path:
                skipped_no_video += 1
                if skipped_no_video <= 5:
                    print(f"  跳过（视频未找到）: {question_key}, 缺少: {primary_video_id}", file=sys.stderr)
                continue
            
            ground_truth_letter = get_correct_answer_letter(correct_idx)
            
            tasks.append({
                'video_id': primary_video_id,
                'video_path': video_path,
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
    
    # 步骤 6: 收集唯一视频并预提取帧
    print("\n[步骤 5] 预提取视频帧...")
    unique_video_ids = set(task['video_id'] for task in tasks)
    print(f"  共有 {len(unique_video_ids)} 个唯一视频需要提取帧")
    
    video_frames_cache = {}
    for video_id in tqdm(unique_video_ids, desc="  提取视频帧"):
        video_path = video_cache[video_id]
        video_data = extract_frames_from_video(video_path, num_frames)
        video_frames_cache[video_id] = video_data
    
    print(f"  帧提取完成！")
    
    # 步骤 7: 进行问答推理
    print("\n[步骤 6] 开始问答测试...")
    all_results = []
    total_tasks = len(tasks)
    progress_width = len(str(total_tasks))
    completed = 0
    
    for task in tasks:
        video_id = task['video_id']
        video_data = video_frames_cache.get(video_id)
        completed += 1
        
        if video_data is None or video_data[0] is None:
            result = {
                'success': False,
                'question_key': task['task_info']['question_key'],
                'error': '无法提取视频帧'
            }
            all_results.append(result)
            print(f"[{completed:>{progress_width}}/{total_tasks}] ✗ "
                  f"Q={task['task_info']['question_key'][:50]:<50} "
                  f"错误=无法提取视频帧")
            continue
        
        result = ask_question_with_video(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            video_data=video_data,
            question_text=task['question_text'],
            choices=task['choices'],
            task_info=task['task_info']
        )
        all_results.append(result)
        
        # 输出每个样例的结果
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
            print(f"[{completed:>{progress_width}}/{total_tasks}] ✗ "
                  f"Q={result['question_key'][:50]:<50} "
                  f"错误={result.get('error', '未知错误')[:30]}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 步骤 8: 统计结果
    results_by_json = {}
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
    
    # 步骤 9: 打印最终结果
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
    
    # 步骤 10: 保存所有结果
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        output_data = {
            'summary': {
                'total_questions': total_questions,
                'total_correct': total_correct,
                'accuracy': (total_correct / total_questions * 100) if total_questions > 0 else 0,
                'elapsed_time_seconds': elapsed_time,
                'json_dir': json_dir,
                'video_dir': video_dir,
                'model_path': model_path,
                'num_frames': num_frames,
            },
            'results_by_file': results_by_json,
            'detailed_results': successful_results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！所有结果已保存到: {output_path}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
