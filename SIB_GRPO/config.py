"""
SIB-GRPO Training Configuration
- GRPO hyperparameters
- LoRA configuration
- Model paths
- Reward weights
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRPOConfig:
    """Full configuration for SIB-GRPO training"""

    # --- Model Paths ---
    policy_model_path: str = ""       # Qwen3-VL-8B (transformers + LoRA, requires gradients)
    base_model_path: str = ""         # Qwen3-VL-8B (vLLM, for inference: L1/L2 caption, VQA)
    supervisor_model_path: str = ""   # Qwen3-VL-72B etc. (vLLM, for inference: supervision signal)
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"

    # --- Data Paths ---
    data_dir: str = ""       # JSON question folder (contains multiple .json files)
    video_dir: str = ""      # Video root directory (recursive search)
    l1_cache_dir: str = "./l1_cache"  # L1 pre-built cache directory
    supervisor_cache_dir: str = "./supervisor_cache"  # Supervisor scoring cache

    # --- L1 Build Parameters (reuses build_retrieve config) ---
    pyscenedetect_threshold: float = 20.0
    l1_max_segment_duration: float = 10.0
    l1_fps: float = 5.0

    # --- L2 Build Parameters ---
    l2_fps: float = 2.0
    l2_max_input_frames: int = 64
    l2_caption_max_new_tokens: int = 512
    l2_decision_max_new_tokens: int = 64

    # --- GRPO Parameters ---
    num_generations: int = 8    # Number of rollout trajectories per sample G
    temperature: float = 0.7    # L2 decision sampling temperature
    kl_coeff: float = 0.01      # KL regularization coefficient
    clip_range: float = 0.2     # PPO-style clip range

    # --- LoRA Parameters ---
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"  # comma-separated

    # --- Reward Weights ---
    reward_correct_weight: float = 1.0     # alpha: VQA answer correctness
    reward_supervisor_weight: float = 0.5  # beta: large model supervision signal
    reward_length_weight: float = 0.1      # gamma: caption length penalty
    caption_length_threshold: int = 200    # caption length threshold (character count)

    # --- Training Parameters ---
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    save_steps: int = 100
    log_steps: int = 10
    output_dir: str = "./grpo_output"
    resume_from: str = ""  # checkpoint resume path

    # --- vLLM Parameters (base model / supervisor model) ---
    gpu_memory_utilization: float = 0.3
    max_model_len: int = 32000
    max_images_per_prompt: int = 256
    tensor_parallel_size: int = 1
    seed: int = 3407

    # --- Device ---
    device: str = "cuda"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIB-GRPO: Memory Manager Fine-tuning")

    # Model paths
    parser.add_argument("--policy_model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--supervisor_model_path", type=str, default=None)
    parser.add_argument("--embedding_model_name", type=str, default=None)

    # Data paths
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--l1_cache_dir", type=str, default=None)
    parser.add_argument("--supervisor_cache_dir", type=str, default=None)

    # L1 build
    parser.add_argument("--pyscenedetect_threshold", type=float, default=None)
    parser.add_argument("--l1_max_segment_duration", type=float, default=None)
    parser.add_argument("--l1_fps", type=float, default=None)

    # GRPO
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--kl_coeff", type=float, default=None)
    parser.add_argument("--clip_range", type=float, default=None)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    # Reward
    parser.add_argument("--reward_correct_weight", type=float, default=None)
    parser.add_argument("--reward_supervisor_weight", type=float, default=None)
    parser.add_argument("--reward_length_weight", type=float, default=None)
    parser.add_argument("--caption_length_threshold", type=int, default=None)

    # Training
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    # vLLM
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GRPOConfig:
    config = GRPOConfig()
    for field_name in vars(config):
        if hasattr(args, field_name):
            val = getattr(args, field_name)
            if val is not None:
                setattr(config, field_name, val)
    return config
