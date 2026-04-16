#!/bin/bash
# Three-layer multimodal video memory system - example run script

# ============================================================
# 1. Build memory
# ============================================================
# python main.py build \
#     --video_path /path/to/video.mp4 \
#     --output_dir ./output \
#     --base_model_path /path/to/Qwen3-VL-8B-Instruct \
#     --finetuned_model_path /path/to/finetuned_model \
#     --pyscenedetect_threshold 27.0 \
#     --l1_max_segment_duration 10.0 \
#     --l1_fps 5.0 \
#     --l2_fps 2.0 \
#     --l3_fps 1.0 \
#     --l3_entity_dedup_threshold 0.7 \
#     --device cuda

# ============================================================
# 2. Retrieve and answer using built memory
# ============================================================
# python main.py retrieve \
#     --video_path /path/to/video.mp4 \
#     --memory_dir ./output \
#     --video_id video_name \
#     --question "What happens after the person picks up the cup?" \
#     --options "A. They drink from it" "B. They put it down" "C. They throw it" "D. They wash it" \
#     --base_model_path /path/to/Qwen3-VL-8B-Instruct \
#     --entropy_threshold 0.5 \
#     --initial_vqa_confidence 0.8 \
#     --num_options 4 \
#     --l3_coarse_top_k 20 \
#     --l3_rerank_top_k 5 \
#     --l2_embedding_top_k 10 \
#     --l2_visual_top_k 3 \
#     --l1_top_k 3 \
#     --device cuda
# python main.py retrieve \
#     --video_path /path/to/videos/data/6Z_XNM_iT4g.mp4 \
#     --memory_dir ./output \
#     --video_id 6Z_XNM_iT4g \
#     --question "In the legend, when the character with a fox tail goes left, what is her intention or goal?" \
#     --options "A. She wants to pick up gold." "B. She wants to ward grass." "C. She wants to slay an enemy." "D. She wants to meet her ally." \
#     --base_model_path /path/to/Qwen3-VL-8B-Instruct \
#     --entropy_threshold 0.5 \
#     --initial_vqa_confidence 0.8 \
#     --num_options 4 \
#     --l3_coarse_top_k 20 \
#     --l3_rerank_top_k 5 \
#     --l2_embedding_top_k 10 \
#     --l2_visual_top_k 3 \
#     --l1_top_k 3 \
#     --device cuda


# ============================================================
# 3. Full pipeline (build + retrieve and answer)
# ============================================================
# python main.py full \
#     --video_path /path/to/video.mp4 \
#     --output_dir ./output \
#     --question "What is the main activity shown in the video?" \
#     --options "A. Cooking" "B. Dancing" "C. Reading" "D. Swimming" \
#     --base_model_path /path/to/Qwen3-VL-8B-Instruct \
#     --finetuned_model_path /path/to/finetuned_model \
#     --device cuda

# ============================================================
# 4. Batch dataset evaluation (eval mode)
# ============================================================
python main.py eval \
    --dataset videomme \
    --data_dir /path/to/videomme \
    --video_dir /path/to/videos/data \
    --duration short \
    --memory_dir ./output \
    --output_file results.jsonl \
    --base_model_path /path/to/Qwen3-VL-8B-Instruct \
    --finetuned_model_path /path/to/Qwen3-VL-8B-Instruct \
    --entropy_threshold 0.5 \
    --initial_vqa_confidence 0.8 \
    --device cuda
