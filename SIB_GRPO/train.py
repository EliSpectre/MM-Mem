"""
SIB-GRPO Training Entry Point
- Load three models (policy/base/supervisor)
- Setup LoRA
- Start training
"""

import os
import sys
import logging

import torch

from config import get_args, build_config
from dataset import load_training_data
from grpo_trainer import GRPOTrainer

# Reuse build_retrieve code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_retrieve"))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_policy_model(config):
    """
    Load policy model (transformers + LoRA).
    Setup LoRA using peft.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model

    logger = logging.getLogger(__name__)
    logger.info(f"Loading policy model: {config.policy_model_path}")

    model = AutoModelForImageTextToText.from_pretrained(
        config.policy_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Setup LoRA
    target_modules = [m.strip() for m in config.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(config.device)
    processor = AutoProcessor.from_pretrained(config.policy_model_path)

    logger.info("Policy model + LoRA loaded successfully")
    return model, processor


def load_vllm_model(model_path, config):
    """Load vLLM inference model"""
    from vllm import LLM
    from transformers import AutoProcessor

    logger = logging.getLogger(__name__)
    logger.info(f"Loading vLLM model: {model_path}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=config.max_model_len,
        limit_mm_per_prompt={"image": config.max_images_per_prompt},
        seed=config.seed,
    )
    processor = AutoProcessor.from_pretrained(model_path)

    logger.info("vLLM model loaded successfully")
    return llm, processor


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    args = get_args()
    config = build_config(args)

    logger.info("===== SIB-GRPO Training =====")
    logger.info(f"Policy model: {config.policy_model_path}")
    logger.info(f"Base model: {config.base_model_path}")
    logger.info(f"Supervisor model: {config.supervisor_model_path or '(none)'}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Video directory: {config.video_dir}")
    logger.info(f"LoRA rank: {config.lora_rank}")
    logger.info(f"Trajectories: {config.num_generations}")
    logger.info(f"Reward weights: alpha={config.reward_correct_weight}, beta={config.reward_supervisor_weight}, gamma={config.reward_length_weight}")

    # --- Load Models ---

    # 1. Policy model (transformers + LoRA)
    policy_model, policy_processor = load_policy_model(config)

    # 2. Base model (vLLM)
    base_llm, base_processor = load_vllm_model(config.base_model_path, config)

    # 3. Supervisor model (vLLM, optional)
    supervisor_llm, supervisor_processor = None, None
    if config.supervisor_model_path:
        supervisor_llm, supervisor_processor = load_vllm_model(
            config.supervisor_model_path, config
        )

    # 4. Embedding model
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(config.embedding_model_name)

    # --- Load Data ---
    dataset = load_training_data(config.data_dir)
    logger.info(f"Loaded {len(dataset)} training samples")

    # --- Training ---
    trainer = GRPOTrainer(
        config=config,
        policy_model=policy_model,
        policy_processor=policy_processor,
        base_llm=base_llm,
        base_processor=base_processor,
        supervisor_llm=supervisor_llm,
        supervisor_processor=supervisor_processor,
        embedding_model=embedding_model,
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()
