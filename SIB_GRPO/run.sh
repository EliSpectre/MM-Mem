#!/bin/bash
# SIB-GRPO: Memory Manager LoRA Fine-tuning

python train.py \
    --policy_model_path /path/to/Qwen3-VL-8B-Instruct \
    --base_model_path /path/to/Qwen3-VL-8B-Instruct \
    --supervisor_model_path /path/to/Qwen3-VL-72B \
    --embedding_model_name /path/to/bge-large-en-v1.5 \
    --data_dir /path/to/json_folder \
    --video_dir /path/to/videos \
    --l1_cache_dir ./l1_cache \
    --supervisor_cache_dir ./supervisor_cache \
    --num_generations 8 \
    --temperature 0.7 \
    --kl_coeff 0.01 \
    --clip_range 0.2 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --reward_correct_weight 1.0 \
    --reward_supervisor_weight 0.5 \
    --reward_length_weight 0.1 \
    --caption_length_threshold 200 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --output_dir ./grpo_output \
    --gpu_memory_utilization 0.3 \
    --max_model_len 32000 \
    --tensor_parallel_size 1
