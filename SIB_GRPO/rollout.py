"""
Trajectory Collection Module
- Policy model (transformers) makes L2 decisions step by step
- Base model (vLLM) generates L2 captions
- BGE embedding retrieval + base model VQA answering
"""

import os
import sys
import math
import logging
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from config import GRPOConfig

# Reuse build_retrieve code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_retrieve"))
from models import (
    build_messages, prepare_vllm_input, generate_text,
    generate_with_logprobs, extract_option_probs,
)
from memory_build import (
    L2_DECISION_PROMPT, L2_CAPTION_PROMPT,
    parse_json_from_response, parse_decision,
)
from video_utils import extract_frames_at_fps, save_frames
from knowledge_graph import compute_text_similarity

logger = logging.getLogger(__name__)


# ============================================================
# Policy Model Decision (transformers, requires gradients)
# ============================================================

def policy_decide(
    policy_model,
    policy_processor,
    messages: List[Dict],
    images: List,
    temperature: float = 0.7,
    device: str = "cuda",
) -> Tuple[str, float, torch.Tensor]:
    """
    Make one L2 decision using policy model (transformers).
    Returns (decision, log_prob, logits)
    - decision: "ADD_NEW" / "MERGE" / "DISCARD"
    - log_prob: log probability of the sampled action (scalar)
    - logits: full logits of the first token (for later recomputation)
    """
    text = policy_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = policy_processor(
        text=[text],
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate 1 token (sampling mode)
    with torch.no_grad():
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=True,
            output_logits=True,
        )

    # Extract generated token and logits
    generated_token = outputs.sequences[0][inputs.input_ids.shape[1]:]
    first_logits = outputs.logits[0]  # (1, vocab_size) or (vocab_size,)
    if first_logits.dim() > 1:
        first_logits = first_logits[0]

    # Compute log_prob of sampled token
    log_probs = F.log_softmax(first_logits.float(), dim=-1)
    token_id = generated_token[0].item()
    log_prob = log_probs[token_id].item()

    # Decode decision
    decoded = policy_processor.decode(generated_token, skip_special_tokens=True).strip()
    decision = parse_decision(decoded)

    return decision, log_prob, first_logits.detach()


def policy_compute_log_prob(
    policy_model,
    policy_processor,
    messages: List[Dict],
    images: List,
    target_token_id: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Recompute log_prob of a specified token (with gradients, for GRPO update).
    Returns a log_prob tensor that requires gradients.
    """
    text = policy_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = policy_processor(
        text=[text],
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Forward pass (with gradients)
    outputs = policy_model(**inputs)
    # Get logits of the last token (i.e., generation position)
    last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
    log_probs = F.log_softmax(last_logits, dim=-1)

    return log_probs[target_token_id]


# ============================================================
# L2 Retrieval + VQA (base model, vLLM)
# ============================================================

def l2_retrieve_and_vqa(
    base_llm,
    base_processor,
    l2_nodes: List[Dict],
    question: str,
    options: List[str],
    embedding_model,
    l1_cache_dir: str,
    video_id: str,
) -> Dict[str, Any]:
    """
    Perform text retrieval using L2 captions, then VQA answering.
    Simplified version (no L3, no visual verification), using only BGE embedding retrieval.
    """
    if not l2_nodes:
        # No L2 nodes, answer with question directly
        messages = build_messages(
            f"Question: {question}\nOptions: {', '.join(options)}\nAnswer with the correct option:",
        )
        vllm_output = generate_with_logprobs(base_llm, base_processor, messages)
        option_labels = tuple(chr(65 + i) for i in range(len(options)))
        option_probs = extract_option_probs(vllm_output, labels=option_labels)
        predicted = max(option_probs, key=option_probs.get)
        return {"predicted_answer": predicted, "option_probs": option_probs}

    # Text retrieval: question vs L2 captions
    captions = [n.get("caption", "") for n in l2_nodes]
    if all(not c for c in captions):
        # All captions are empty, select all nodes
        selected_indices = list(range(len(l2_nodes)))
    else:
        similarities = compute_text_similarity(
            [question], captions, embedding_model
        )
        if similarities.size > 0:
            ranked = np.argsort(-similarities[0])
            selected_indices = ranked[:min(3, len(ranked))].tolist()
        else:
            selected_indices = list(range(min(3, len(l2_nodes))))

    # Build context
    context_parts = []
    all_frame_paths = []
    for idx in selected_indices:
        node = l2_nodes[idx]
        cap = node.get("caption", "")
        if cap:
            context_parts.append(f"[Episode {idx}] {cap}")
        # Load L2 frame paths
        frame_paths = node.get("frame_paths", [])
        frame_dir = node.get("frame_dir", "")
        if frame_paths and frame_dir:
            full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
            all_frame_paths.extend(full_paths)

    context_text = "\n".join(context_parts) if context_parts else ""

    # VQA
    option_labels = tuple(chr(65 + i) for i in range(len(options)))
    options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

    if context_text:
        prompt = (
            f"Retrieved context:\n{context_text}\n\n"
            f"Based on the video frames and context, select the best answer.\n"
            f"Respond with only the letter ({'/'.join(option_labels)}).\n"
            f"Question: {question}\n{options_text}\nThe best answer is:"
        )
    else:
        prompt = (
            f"Select the best answer.\n"
            f"Respond with only the letter ({'/'.join(option_labels)}).\n"
            f"Question: {question}\n{options_text}\nThe best answer is:"
        )

    messages = build_messages(prompt, images=all_frame_paths if all_frame_paths else None)
    vllm_output = generate_with_logprobs(base_llm, base_processor, messages)
    option_probs = extract_option_probs(vllm_output, labels=option_labels)
    predicted = max(option_probs, key=option_probs.get)

    return {"predicted_answer": predicted, "option_probs": option_probs}


# ============================================================
# Trajectory Collection
# ============================================================

def rollout_single_trajectory(
    policy_model,
    policy_processor,
    base_llm,
    base_processor,
    l1_nodes: List[Dict],
    question: str,
    options: List[str],
    config: GRPOConfig,
    embedding_model,
    video_path: str,
    video_id: str,
) -> Dict[str, Any]:
    """
    Collect one complete trajectory:
    1. Iterate over L1 nodes, policy model samples L2 decisions
    2. Base model generates L2 caption
    3. BGE retrieval + VQA answering

    Returns:
    {
        "steps": [
            {
                "l1_index": int,
                "decision": str,
                "log_prob": float,
                "target_token_id": int,
                "message_info": {...},  # for recomputing log_prob
            }, ...
        ],
        "l2_nodes": [...],
        "predicted_answer": str,
        "option_probs": dict,
    }
    """
    steps = []
    l2_nodes = []
    current_l2 = None
    l2_index = 0
    current_l2_frame_paths = []

    for i, l1_node in enumerate(l1_nodes):
        l1_idx = l1_node["l1_index"]
        l1_frame_dir = os.path.join(config.l1_cache_dir, video_id, "L1_frames", f"l1_{l1_idx:04d}")
        l1_frame_paths = [os.path.join(l1_frame_dir, p) for p in l1_node.get("frame_paths", [])]

        if not l1_frame_paths:
            continue

        if current_l2 is None:
            # First L1, default ADD_NEW
            current_l2 = {
                "l2_index": l2_index,
                "video_id": video_id,
                "start_sec": l1_node["start_sec"],
                "end_sec": l1_node["end_sec"],
                "constituent_l1_indices": [l1_idx],
                "caption": "",
                "frame_paths": [],
                "frame_dir": "",
            }
            current_l2_frame_paths = list(l1_frame_paths)
            steps.append({
                "l1_index": l1_idx,
                "decision": "ADD_NEW",
                "log_prob": 0.0,  # First default, not involved in gradient
                "target_token_id": -1,
                "is_default": True,
            })
            continue

        # Build decision prompt
        decision_text = L2_DECISION_PROMPT.format(
            current_l2_caption=current_l2["caption"] or "(caption not yet generated)",
            current_l2_start=current_l2["start_sec"],
            current_l2_end=current_l2["end_sec"],
            new_l1_caption=l1_node.get("caption", ""),
            new_l1_start=l1_node["start_sec"],
            new_l1_end=l1_node["end_sec"],
        )

        # Merge frame paths (limit count)
        combined_paths = current_l2_frame_paths + l1_frame_paths
        if len(combined_paths) > config.l2_max_input_frames:
            indices = np.linspace(0, len(combined_paths) - 1, config.l2_max_input_frames, dtype=int)
            combined_paths = [combined_paths[i] for i in indices]

        # Load frames as PIL (policy model uses transformers, requires PIL)
        combined_images = []
        for p in combined_paths:
            if os.path.exists(p):
                combined_images.append(Image.open(p).convert("RGB"))

        messages = build_messages(decision_text, images=combined_images)

        # Policy model samples decision
        decision, log_prob, logits = policy_decide(
            policy_model, policy_processor, messages, combined_images,
            temperature=config.temperature, device=config.device,
        )

        # Record info for recomputing log_prob
        generated_text = decision
        # Get token_id corresponding to the decision token
        decision_tokens = policy_processor.encode(generated_text, add_special_tokens=False)
        target_token_id = decision_tokens[0] if decision_tokens else -1

        steps.append({
            "l1_index": l1_idx,
            "decision": decision,
            "log_prob": log_prob,
            "target_token_id": target_token_id,
            "is_default": False,
            # Save info for recomputation
            "decision_text": decision_text,
            "combined_paths": combined_paths,
        })

        # Execute decision
        if decision == "ADD_NEW":
            # Finalize current L2: generate caption
            current_l2 = _finalize_l2_trajectory(
                current_l2, video_path, config, base_llm, base_processor
            )
            l2_nodes.append(current_l2)

            # Start new L2
            l2_index += 1
            current_l2 = {
                "l2_index": l2_index,
                "video_id": video_id,
                "start_sec": l1_node["start_sec"],
                "end_sec": l1_node["end_sec"],
                "constituent_l1_indices": [l1_idx],
                "caption": "",
                "frame_paths": [],
                "frame_dir": "",
            }
            current_l2_frame_paths = list(l1_frame_paths)

        elif decision == "MERGE":
            current_l2["constituent_l1_indices"].append(l1_idx)
            current_l2["end_sec"] = max(current_l2["end_sec"], l1_node["end_sec"])
            current_l2_frame_paths.extend(l1_frame_paths)

        elif decision == "DISCARD":
            pass  # Skip

    # Finalize the last L2
    if current_l2 is not None:
        current_l2 = _finalize_l2_trajectory(
            current_l2, video_path, config, base_llm, base_processor
        )
        l2_nodes.append(current_l2)

    # L2 retrieval + VQA
    vqa_result = l2_retrieve_and_vqa(
        base_llm, base_processor, l2_nodes, question, options,
        embedding_model, config.l1_cache_dir, video_id,
    )

    return {
        "steps": steps,
        "l2_nodes": l2_nodes,
        "predicted_answer": vqa_result["predicted_answer"],
        "option_probs": vqa_result["option_probs"],
    }


def _finalize_l2_trajectory(
    l2_node: Dict,
    video_path: str,
    config: GRPOConfig,
    base_llm,
    base_processor,
) -> Dict:
    """Generate L2 caption using base model (vLLM)"""
    frames, timestamps = extract_frames_at_fps(
        video_path, l2_node["start_sec"], l2_node["end_sec"], fps=config.l2_fps
    )

    if frames:
        # Save frames to temporary directory
        frame_dir = os.path.join(
            config.l1_cache_dir, l2_node["video_id"],
            "L2_frames_tmp", f"l2_{l2_node['l2_index']:04d}"
        )
        frame_paths = save_frames(frames, timestamps, frame_dir)
        l2_node["frame_paths"] = frame_paths
        l2_node["frame_dir"] = frame_dir

        full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
        messages = build_messages(L2_CAPTION_PROMPT, images=full_paths)
        response = generate_text(
            base_llm, base_processor, messages,
            max_new_tokens=config.l2_caption_max_new_tokens,
        )
        caption_data = parse_json_from_response(response)
        l2_node["caption"] = json.dumps(caption_data, ensure_ascii=False) if caption_data else response
    else:
        l2_node["caption"] = ""

    return l2_node


def rollout_trajectories(
    policy_model,
    policy_processor,
    base_llm,
    base_processor,
    l1_nodes: List[Dict],
    question: str,
    options: List[str],
    config: GRPOConfig,
    embedding_model,
    video_path: str,
    video_id: str,
) -> List[Dict]:
    """Collect G trajectories"""
    trajectories = []
    for k in range(config.num_generations):
        logger.info(f"    Trajectory {k+1}/{config.num_generations}")
        traj = rollout_single_trajectory(
            policy_model, policy_processor,
            base_llm, base_processor,
            l1_nodes, question, options,
            config, embedding_model,
            video_path, video_id,
        )
        trajectories.append(traj)
    return trajectories


# Need to import json (used in _finalize_l2_trajectory)
import json
