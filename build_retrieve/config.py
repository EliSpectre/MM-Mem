"""
Configuration management module - hyperparameters, command-line arguments, dataclass definitions
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryConfig:
    """Full configuration for the three-layer multimodal video memory system"""

    # --- Model paths ---
    base_model_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    finetuned_model_path: str = ""  # L2 ADD_NEW/MERGE/DISCARD fine-tuned model path
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # --- Video input/output ---
    video_path: str = ""
    output_dir: str = "./output"
    memory_dir: str = ""  # Memory directory for retrieval, defaults to output_dir
    video_id: str = ""  # Specify video_id during retrieval

    # --- L1 build parameters ---
    pyscenedetect_threshold: float = 20.0  # ContentDetector threshold
    l1_max_segment_duration: float = 10.0  # L1 segment max duration (seconds), split if exceeded
    l1_fps: float = 5.0  # L1 frame extraction rate

    # --- L2 build parameters ---
    l2_fps: float = 2.0  # L2 visual memory frame extraction rate
    l2_max_input_frames: int = 64  # Max frames for L2 decision (L1 frames + L2 frames combined)
    l2_caption_max_new_tokens: int = 512
    l2_decision_max_new_tokens: int = 64

    # --- L3 build parameters ---
    l3_fps: float = 1.0  # L3 visual memory frame extraction rate
    l3_entity_dedup_threshold: float = 0.7  # BGE similarity threshold for entity deduplication
    l3_entity_max_new_tokens: int = 512
    l3_entity_max_retries: int = 5  # Retry count on JSON parsing failure

    # --- Initial VQA ---
    initial_vqa_max_frames: int = 64  # Max frames for initial VQA
    initial_vqa_confidence: float = 0.8  # Confidence threshold for initial VQA

    # --- Retrieval parameters ---
    entropy_threshold: float = 0.5  # Entropy threshold (max entropy for 4 options = 1.386)
    num_options: int = 4  # Number of options, default 4 (A/B/C/D)

    # L3 retrieval
    l3_coarse_top_k: int = 20  # L3 coarse ranking retention count
    l3_rerank_top_k: int = 5  # L3 fine ranking retention count
    l3_retrieval_similarity_threshold: float = 0.5  # Graph retrieval similarity threshold

    # L2 retrieval
    l2_embedding_top_k: int = 10  # L2 text retrieval retention count
    l2_visual_top_k: int = 3  # L2 visual verification retention count

    # L1 retrieval
    l1_top_k: int = 3  # L1 final retention count

    # --- Generation parameters ---
    caption_max_new_tokens: int = 512  # Max tokens for caption generation
    vqa_max_new_tokens: int = 1  # VQA generates only 1 token
    top_logprobs: int = 10  # Get probabilities of top-10 tokens

    # --- vLLM parameters (aligned with infer_instruct.sh) ---
    tensor_parallel_size: int = 1  # GPU parallel count, default auto-detect
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 128000
    max_images_per_prompt: int = 256  # Max images per inference
    seed: int = 3407

    # --- vLLM generation parameters (aligned with infer_instruct.sh) ---
    temperature: float = 0.7
    top_p: float = 0.8
    top_k_sampling: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.5
    gen_max_new_tokens: int = 32768  # Max tokens for answer generation

    # --- Video processing parameters (aligned with infer_instruct.sh) ---
    fps: float = 2.0  # vLLM video processing fps
    min_pixels: int = 16 * 32 * 32        # 16384
    max_pixels: int = 512 * 32 * 32       # 524288
    min_frames: int = 4
    max_frames: int = 768
    total_pixels: int = 20480 * 32 * 32   # 20971520

    # --- Device ---
    device: str = "cuda"
    torch_dtype: str = "bfloat16"

    # --- VQA question (retrieval mode) ---
    question: str = ""
    options: List[str] = field(default_factory=list)

    # --- Dataset parameters (eval mode) ---
    dataset: str = ""  # Dataset name: "videomme"
    data_dir: str = ""  # Dataset path (containing VQA questions, etc.)
    video_dir: str = ""  # Video file directory
    duration: str = "short"  # Duration filter: "short"/"medium"/"long"/"all"
    output_file: str = "results.jsonl"  # Result output file


def get_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Three-layer multimodal video memory system")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # --- Common arguments (default=None means do not override MemoryConfig defaults) ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base_model_path", type=str, default=None)
    common.add_argument("--finetuned_model_path", type=str, default=None)
    common.add_argument("--embedding_model_name", type=str, default=None)
    common.add_argument("--reranker_model_name", type=str, default=None)
    common.add_argument("--device", type=str, default=None)
    common.add_argument("--torch_dtype", type=str, default=None)
    # vLLM parameters
    common.add_argument("--tensor_parallel_size", type=int, default=None)
    common.add_argument("--gpu_memory_utilization", type=float, default=None)
    common.add_argument("--max_model_len", type=int, default=None)
    common.add_argument("--max_images_per_prompt", type=int, default=None)
    common.add_argument("--seed", type=int, default=None)

    # --- build subcommand ---
    build_parser = subparsers.add_parser("build", parents=[common], help="Build memory for a video")
    build_parser.add_argument("--video_path", type=str, required=True)
    build_parser.add_argument("--output_dir", type=str, default=None)
    build_parser.add_argument("--pyscenedetect_threshold", type=float, default=None)
    build_parser.add_argument("--l1_max_segment_duration", type=float, default=None)
    build_parser.add_argument("--l1_fps", type=float, default=None)
    build_parser.add_argument("--l2_fps", type=float, default=None)
    build_parser.add_argument("--l2_max_input_frames", type=int, default=None)
    build_parser.add_argument("--l3_fps", type=float, default=None)
    build_parser.add_argument("--l3_entity_dedup_threshold", type=float, default=None)

    # --- retrieve subcommand ---
    retrieve_parser = subparsers.add_parser("retrieve", parents=[common], help="Answer questions using built memory")
    retrieve_parser.add_argument("--video_path", type=str, required=True)
    retrieve_parser.add_argument("--memory_dir", type=str, required=True)
    retrieve_parser.add_argument("--video_id", type=str, required=True)
    retrieve_parser.add_argument("--question", type=str, required=True)
    retrieve_parser.add_argument("--options", type=str, nargs="+", required=True)
    retrieve_parser.add_argument("--num_options", type=int, default=None)
    retrieve_parser.add_argument("--entropy_threshold", type=float, default=None)
    retrieve_parser.add_argument("--initial_vqa_confidence", type=float, default=None)
    retrieve_parser.add_argument("--initial_vqa_max_frames", type=int, default=None)
    retrieve_parser.add_argument("--l3_coarse_top_k", type=int, default=None)
    retrieve_parser.add_argument("--l3_rerank_top_k", type=int, default=None)
    retrieve_parser.add_argument("--l2_embedding_top_k", type=int, default=None)
    retrieve_parser.add_argument("--l2_visual_top_k", type=int, default=None)
    retrieve_parser.add_argument("--l1_top_k", type=int, default=None)

    # --- full subcommand ---
    full_parser = subparsers.add_parser("full", parents=[common], help="Build memory + retrieve answers")
    full_parser.add_argument("--video_path", type=str, required=True)
    full_parser.add_argument("--output_dir", type=str, default=None)
    full_parser.add_argument("--question", type=str, required=True)
    full_parser.add_argument("--options", type=str, nargs="+", required=True)
    full_parser.add_argument("--pyscenedetect_threshold", type=float, default=None)
    full_parser.add_argument("--l1_max_segment_duration", type=float, default=None)
    full_parser.add_argument("--l1_fps", type=float, default=None)
    full_parser.add_argument("--l2_fps", type=float, default=None)
    full_parser.add_argument("--l2_max_input_frames", type=int, default=None)
    full_parser.add_argument("--l3_fps", type=float, default=None)
    full_parser.add_argument("--l3_entity_dedup_threshold", type=float, default=None)
    full_parser.add_argument("--num_options", type=int, default=None)
    full_parser.add_argument("--entropy_threshold", type=float, default=None)
    full_parser.add_argument("--initial_vqa_confidence", type=float, default=None)
    full_parser.add_argument("--initial_vqa_max_frames", type=int, default=64)
    full_parser.add_argument("--l3_coarse_top_k", type=int, default=20)
    full_parser.add_argument("--l3_rerank_top_k", type=int, default=5)
    full_parser.add_argument("--l2_embedding_top_k", type=int, default=10)
    full_parser.add_argument("--l2_visual_top_k", type=int, default=3)
    full_parser.add_argument("--l1_top_k", type=int, default=3)

    # --- eval subcommand ---
    eval_parser = subparsers.add_parser("eval", parents=[common], help="Batch dataset evaluation")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Dataset name: videomme")
    eval_parser.add_argument("--data_dir", type=str, required=True, help="Dataset path")
    eval_parser.add_argument("--video_dir", type=str, required=True, help="Video file directory")
    eval_parser.add_argument("--duration", type=str, default=None, help="Duration filter: short/medium/long/all")
    eval_parser.add_argument("--memory_dir", type=str, default=None, help="Memory storage directory (default=./output)")
    eval_parser.add_argument("--output_file", type=str, default=None, help="Result output file (default=results.jsonl)")
    eval_parser.add_argument("--num_options", type=int, default=None)
    eval_parser.add_argument("--entropy_threshold", type=float, default=None)
    eval_parser.add_argument("--initial_vqa_confidence", type=float, default=None)
    eval_parser.add_argument("--initial_vqa_max_frames", type=int, default=None)
    eval_parser.add_argument("--l3_coarse_top_k", type=int, default=None)
    eval_parser.add_argument("--l3_rerank_top_k", type=int, default=None)
    eval_parser.add_argument("--l2_embedding_top_k", type=int, default=None)
    eval_parser.add_argument("--l2_visual_top_k", type=int, default=None)
    eval_parser.add_argument("--l1_top_k", type=int, default=None)
    # Build-related parameters (used when auto-building memory)
    eval_parser.add_argument("--pyscenedetect_threshold", type=float, default=None)
    eval_parser.add_argument("--l1_max_segment_duration", type=float, default=None)
    eval_parser.add_argument("--l1_fps", type=float, default=None)
    eval_parser.add_argument("--l2_fps", type=float, default=None)
    eval_parser.add_argument("--l3_fps", type=float, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MemoryConfig:
    """Build MemoryConfig from command-line arguments"""
    config = MemoryConfig()

    # Copy existing fields from args to config
    for field_name in vars(config):
        if hasattr(args, field_name):
            val = getattr(args, field_name)
            if val is not None:
                setattr(config, field_name, val)

    # memory_dir defaults to output_dir
    if not config.memory_dir and hasattr(args, "output_dir") and args.output_dir:
        config.memory_dir = args.output_dir

    return config
