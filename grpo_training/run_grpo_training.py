"""
GRPO 强化学习训练主脚本

完整流程:
1. Phase 1: 轨迹收集 - 生成 L1，窗口化处理，采样路径，Teacher 打分
2. Phase 2: VQA 评估 - 检索记忆，VQA 问答，计算 R_vqa 并回溯
3. Phase 3: GRPO 训练 - 使用 Swift 框架进行强化学习微调
"""

import os
import os.path as osp
import json
import argparse
import glob
from typing import List, Dict, Tuple, Optional

from .data_structures import VideoTrajectory
from .trajectory_collector import TrajectoryCollector
from .vqa_evaluator import VQAEvaluator
from .grpo_trainer import GRPOTrainer, GRPOConfig


def load_video_list(video_list_path: str) -> List[Tuple[str, str]]:
    """
    加载视频列表
    
    支持格式:
    1. JSON: [{"video_path": "...", "video_id": "..."}, ...]
    2. TXT: 每行一个视频路径
    """
    if video_list_path.endswith('.json'):
        with open(video_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_list = []
        for item in data:
            if isinstance(item, dict):
                video_path = item.get('video_path', '')
                video_id = item.get('video_id', osp.splitext(osp.basename(video_path))[0])
            else:
                video_path = str(item)
                video_id = osp.splitext(osp.basename(video_path))[0]
            
            if video_path:
                video_list.append((video_path, video_id))
        
        return video_list
    
    elif video_list_path.endswith('.txt'):
        video_list = []
        with open(video_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    video_id = osp.splitext(osp.basename(line))[0]
                    video_list.append((line, video_id))
        return video_list
    
    else:
        raise ValueError(f"不支持的文件格式: {video_list_path}")


def load_trajectories_from_path(resume_path: str) -> List[VideoTrajectory]:
    """
    从指定路径加载轨迹数据
    
    支持三种方式:
    1. 单个 JSON 文件 (如 P01-xxx_trajectory.json)
    2. 包含多个 *_trajectory.json 的文件夹
    3. 合并后的 evaluated_trajectories.json (包含 "trajectories" 键)
    
    Args:
        resume_path: 文件路径或文件夹路径
        
    Returns:
        VideoTrajectory 列表
    """
    trajectories = []
    
    if osp.isdir(resume_path):
        # 方式 2: 从文件夹加载所有 *_trajectory.json
        print(f"[恢复] 从文件夹加载轨迹: {resume_path}")
        pattern = osp.join(resume_path, "*_trajectory.json")
        traj_files = glob.glob(pattern)
        
        if not traj_files:
            print(f"[警告] 文件夹中没有找到 *_trajectory.json 文件")
            return []
        
        print(f"[恢复] 找到 {len(traj_files)} 个轨迹文件")
        
        for traj_file in sorted(traj_files):
            try:
                with open(traj_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                traj = VideoTrajectory.from_dict(data)
                trajectories.append(traj)
                print(f"  - 加载: {osp.basename(traj_file)} (video_id: {traj.video_id})")
            except Exception as e:
                print(f"  - [错误] 加载 {traj_file} 失败: {e}")
    
    elif osp.isfile(resume_path):
        print(f"[恢复] 从文件加载轨迹: {resume_path}")
        
        with open(resume_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "trajectories" in data:
            # 方式 3: 合并后的文件 (如 evaluated_trajectories.json)
            trajectories = [VideoTrajectory.from_dict(t) for t in data["trajectories"]]
            print(f"[恢复] 从合并文件加载了 {len(trajectories)} 个轨迹")
        else:
            # 方式 1: 单个轨迹文件
            traj = VideoTrajectory.from_dict(data)
            trajectories = [traj]
            print(f"[恢复] 加载了单个轨迹 (video_id: {traj.video_id})")
    
    else:
        raise FileNotFoundError(f"找不到路径: {resume_path}")
    
    return trajectories


def run_phase1_trajectory_collection(
    video_list: List[Tuple[str, str]],
    output_dir: str,
    memory_dir: str,
    config: Dict
) -> List[VideoTrajectory]:
    """Phase 1: 轨迹收集"""
    print("\n" + "="*70)
    print("                    Phase 1: 轨迹收集")
    print("="*70)
    
    collector = TrajectoryCollector(
        finetune_model_path=config.get("finetune_model_path"),
        finetune_device=config.get("finetune_device", "cuda:0"),
        teacher_api_url=config.get("teacher_api", {}).get("base_url", "http://localhost:8002/v1"),
        teacher_api_key=config.get("teacher_api", {}).get("api_key", "EMPTY"),
        teacher_model_name=config.get("teacher_api", {}).get("model_name", "Qwen3-VL-2B-Instruct"),
        base_api_url=config.get("base_api", {}).get("base_url", "http://localhost:8002/v1"),
        base_api_key=config.get("base_api", {}).get("api_key", "EMPTY"),
        base_model_name=config.get("base_api", {}).get("model_name", "Qwen3-VL-2B-Instruct"),
        window_size=config.get("grpo", {}).get("window_size", 5),
        window_stride=config.get("grpo", {}).get("window_stride", 5),
        num_sampled_paths=config.get("grpo", {}).get("num_sampled_paths", 4),
        sampling_temperature=config.get("grpo", {}).get("temperature", 0.7),
        output_dir=output_dir,
        memory_dir=memory_dir
    )
    
    trajectories = collector.collect_trajectories_batch(
        video_list=video_list,
        force_regenerate_l1=config.get("force_regenerate_l1", False)
    )
    
    print(f"\n[Phase 1 完成] 收集了 {len(trajectories)} 个视频的轨迹")
    
    return trajectories


def run_phase2_vqa_evaluation(
    trajectories: List[VideoTrajectory],
    qa_folder: str,
    output_dir: str,
    memory_dir: str,
    config: Dict
) -> List[VideoTrajectory]:
    """Phase 2: VQA 评估"""
    print("\n" + "="*70)
    print("                    Phase 2: VQA 评估")
    print("="*70)
    
    
    evaluator = VQAEvaluator(
        memory_dir=memory_dir,
        api_base_url=config.get("vqa_api", {}).get("base_url", "http://localhost:8002/v1"),
        api_key=config.get("vqa_api", {}).get("api_key", "EMPTY"),
        model_name=config.get("vqa_api", {}).get("model_name", "Qwen2.5-VL-7B-Instruct"),
        embedding_model_name=config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
        rerank_model_name=config.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
        device=config.get("device", "cuda"),
        top_k_l2=config.get("retrieval", {}).get("top_k_l2", 5),
        top_k_l1=config.get("retrieval", {}).get("top_k_l1", 3),
        alpha=config.get("reward", {}).get("alpha", 0.5)
    )
    
    # 加载 QA 数据
    qa_data = evaluator.load_qa_from_folder(qa_folder)
    
    # 运行评估
    output_path = osp.join(output_dir, "evaluated_trajectories.json")
    trajectories = evaluator.run_evaluation(
        trajectories=trajectories,
        qa_data=qa_data,
        output_path=output_path,
        verbose=config.get("verbose", False)
    )
    
    print(f"\n[Phase 2 完成] 评估完成，结果保存到: {output_path}")
    
    return trajectories


def run_phase3_grpo_training(
    trajectories: List[VideoTrajectory],
    output_dir: str,
    config: Dict
) -> str:
    """Phase 3: GRPO 训练"""
    print("\n" + "="*70)
    print("                    Phase 3: GRPO 训练")
    print("="*70)
    
    grpo_config = GRPOConfig(
        model_path=config.get("finetune_model_path", "Qwen/Qwen3-VL-2B-Instruct"),
        output_dir=osp.join(output_dir, "grpo_checkpoints"),
        use_lora=config.get("grpo", {}).get("use_lora", True),
        lora_rank=config.get("grpo", {}).get("lora_rank", 64),
        lora_alpha=config.get("grpo", {}).get("lora_alpha", 128),
        num_train_epochs=config.get("grpo", {}).get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("grpo", {}).get("batch_size", 1),
        gradient_accumulation_steps=config.get("grpo", {}).get("gradient_accumulation_steps", 8),
        learning_rate=config.get("grpo", {}).get("learning_rate", 1e-5),
        beta=config.get("grpo", {}).get("beta", 0.1),
        num_generations=config.get("grpo", {}).get("num_sampled_paths", 4),
        # PPO-Clip 和重要性采样配置
        ppo_clip_epsilon=config.get("grpo", {}).get("ppo_clip_epsilon", 0.2),
        use_importance_sampling=config.get("grpo", {}).get("use_importance_sampling", True),
        use_kl_penalty=config.get("grpo", {}).get("use_kl_penalty", False),
        kl_penalty_coef=config.get("grpo", {}).get("kl_penalty_coef", 0.1)
    )
    
    trainer = GRPOTrainer(config=grpo_config)
    
    # 准备训练数据
    train_data_path = trainer.prepare_training_data(trajectories)
    
    # 选择训练方式
    training_method = config.get("grpo", {}).get("training_method", "importance_sampling")
    
    if training_method == "importance_sampling":
        # 推荐方式：带重要性采样和 PPO-Clip 的手动 GRPO
        model_path = trainer.train_grpo_with_importance_sampling(trajectories)
    elif training_method == "swift_custom_reward":
        # 推荐方式：使用 Swift API + 自定义 reward function
        model_path = trainer.train_with_swift_api_custom_reward(trajectories)
    elif training_method == "swift_cli":
        model_path = trainer.train_with_swift_cli(train_data_path)
    elif training_method == "swift_api":
        model_path = trainer.train_with_swift_api(train_data_path)
    elif training_method == "manual":
        model_path = trainer.train_manual_grpo(trajectories)
    else:
        print(f"[警告] 未知的训练方式: {training_method}，使用 importance_sampling")
        model_path = trainer.train_grpo_with_importance_sampling(trajectories)
    
    print(f"\n[Phase 3 完成] 训练完成，模型保存到: {model_path}")
    
    return model_path


def run_full_pipeline(
    video_list_path: str,
    qa_folder: str,
    config_path: str = None,
    output_dir: str = "./output/grpo_training"
):
    """运行完整的 GRPO 训练流程"""
    
    # 加载配置
    config = load_config(config_path)
    
    # 获取项目根目录
    project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    
    # 设置输出目录 - 确保使用绝对路径
    if not osp.isabs(output_dir):
        output_dir = osp.join(project_root, output_dir)
    output_dir = osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置内存目录 - 确保使用绝对路径
    default_memory_dir = osp.join(output_dir, "memories")
    memory_dir = config.get("memory_dir", default_memory_dir)
    if not osp.isabs(memory_dir):
        memory_dir = osp.join(project_root, memory_dir)
    memory_dir = osp.abspath(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)
    
    # 打印路径信息用于调试
    print("\n" + "="*70)
    print("                    路径配置信息")
    print("="*70)
    print(f"项目根目录: {project_root}")
    print(f"输出目录: {output_dir}")
    print(f"内存目录: {memory_dir}")
    print("="*70 + "\n")
    
    # 加载视频列表
    video_list = load_video_list(video_list_path)
    print(f"[信息] 加载了 {len(video_list)} 个视频")
    
    # Phase 1: 轨迹收集
    trajectories = run_phase1_trajectory_collection(
        video_list=video_list,
        output_dir=output_dir,
        memory_dir=memory_dir,
        config=config
    )
    
    if not trajectories:
        print("[错误] 没有收集到任何轨迹")
        return
    
    # Phase 2: VQA 评估
    trajectories = run_phase2_vqa_evaluation(
        trajectories=trajectories,
        qa_folder=qa_folder,
        output_dir=output_dir,
        memory_dir=memory_dir,
        config=config
    )
    
    # Phase 3: GRPO 训练
    model_path = run_phase3_grpo_training(
        trajectories=trajectories,
        output_dir=output_dir,
        config=config
    )
    
    print("\n" + "="*70)
    print("                    训练完成!")
    print("="*70)
    print(f"输出目录: {output_dir}")
    print(f"模型路径: {model_path}")


def get_default_config_path() -> str:
    """获取默认配置文件路径"""
    current_dir = osp.dirname(osp.abspath(__file__))
    return osp.join(current_dir, "grpo_config.json")


def load_config(config_path: str = None) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    if osp.exists(config_path):
        print(f"[配置] 加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"[配置] 配置文件不存在: {config_path}，使用默认配置")
        return {}


def main():
    parser = argparse.ArgumentParser(description="GRPO 强化学习训练")
    
    # 必需参数
    parser.add_argument(
        "--video-list", type=str, required=True,
        help="视频列表文件路径 (JSON 或 TXT)"
    )
    parser.add_argument(
        "--qa-folder", type=str, required=True,
        help="QA JSON 文件夹路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (JSON)，默认使用 grpo_training/grpo_config.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/grpo_training",
        help="输出目录"
    )
    
    # 阶段控制
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "1", "2", "3", "1-2", "2-3"],
        help="运行的阶段: all, 1, 2, 3, 1-2, 2-3"
    )
    
    # 继续训练
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从之前保存的轨迹恢复。支持: 1) 单个JSON文件 2) 包含多个*_trajectory.json的文件夹 3) 合并后的evaluated_trajectories.json"
    )
    
    args = parser.parse_args()
    
    
    
    if args.phase == "all":
        run_full_pipeline(
            video_list_path=args.video_list,
            qa_folder=args.qa_folder,
            config_path=args.config,
            output_dir=args.output_dir
        )
    else:
        # 部分阶段运行（需要从文件恢复或只运行特定阶段）
        # 加载配置
        config = load_config(args.config)
        
        memory_dir = config.get("memory_dir", osp.join(args.output_dir, "memories"))
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(memory_dir, exist_ok=True)
        
        trajectories = []
        
        # 如果需要恢复
        if args.resume:
            trajectories = load_trajectories_from_path(args.resume)
            print(f"[恢复] 共加载了 {len(trajectories)} 个视频轨迹")
        
        # 运行指定阶段
        if args.phase in ["1", "1-2"]:
            video_list = load_video_list(args.video_list)
            trajectories = run_phase1_trajectory_collection(
                video_list=video_list,
                output_dir=args.output_dir,
                memory_dir=memory_dir,
                config=config
            )
        
        if args.phase in ["2", "1-2", "2-3"]:
            trajectories = run_phase2_vqa_evaluation(
                trajectories=trajectories,
                qa_folder=args.qa_folder,
                output_dir=args.output_dir,
                memory_dir=memory_dir,
                config=config
            )
        
        if args.phase in ["3", "2-3"]:
            run_phase3_grpo_training(
                trajectories=trajectories,
                output_dir=args.output_dir,
                config=config
            )


if __name__ == "__main__":
    main()
