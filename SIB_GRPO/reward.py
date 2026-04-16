"""
Reward Computation Module
- R1: VQA answer correctness (0/1)
- R2: Supervisor model scoring for each step's decision
- R3: Caption length penalty
"""

import os
import sys
import json
import hashlib
import logging
from typing import Dict, List, Any

from config import GRPOConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_retrieve"))
from models import build_messages, generate_with_logprobs, extract_yes_no_probs

logger = logging.getLogger(__name__)


# ============================================================
# R1: VQA Answer Correctness
# ============================================================

def compute_reward_correct(predicted: str, correct_idx: int, choices: List[str]) -> float:
    """
    VQA answer correctness.
    predicted is the option letter (A/B/C/D/E), correct_idx is the index of the correct option.
    """
    predicted_idx = ord(predicted.upper()) - ord("A") if predicted else -1
    return 1.0 if predicted_idx == correct_idx else 0.0


# ============================================================
# R2: Supervisor Model Signal
# ============================================================

SUPERVISOR_PROMPT = """You are evaluating a memory management decision for video understanding.

Current memory episode:
- Time range: {l2_start:.1f}s - {l2_end:.1f}s
- Caption: {l2_caption}

New video segment:
- Time range: {l1_start:.1f}s - {l1_end:.1f}s
- Caption: {l1_caption}

The decision made was: {decision}

Is this a good decision? Consider:
- ADD_NEW: appropriate when the new segment is a clearly different scene
- MERGE: appropriate when the new segment continues the same scene/activity
- DISCARD: appropriate when the segment is noise, transitions, or redundant

Reply with exactly one token: yes or no."""


def _cache_key(l2_info: Dict, l1_info: Dict, decision: str) -> str:
    """Generate cache key"""
    raw = f"{l2_info.get('start_sec', 0)}-{l2_info.get('end_sec', 0)}-{l1_info.get('start_sec', 0)}-{l1_info.get('end_sec', 0)}-{decision}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def compute_reward_supervisor(
    trajectory: Dict,
    l1_nodes: List[Dict],
    supervisor_llm,
    supervisor_processor,
    config: GRPOConfig,
) -> float:
    """
    R2: Supervisor model scoring for each L2 decision step.
    For each non-default decision in the trajectory, ask supervisor if it is reasonable, extract yes_prob.
    """
    if supervisor_llm is None:
        return 0.0

    l1_by_index = {n["l1_index"]: n for n in l1_nodes}
    scores = []

    # Reconstruct L2 state for supervisor evaluation
    current_l2_info = {"start_sec": 0.0, "end_sec": 0.0, "caption": ""}

    for step in trajectory["steps"]:
        if step.get("is_default", False):
            # First default ADD_NEW, update L2 state
            l1 = l1_by_index.get(step["l1_index"], {})
            current_l2_info = {
                "start_sec": l1.get("start_sec", 0),
                "end_sec": l1.get("end_sec", 0),
                "caption": l1.get("caption", ""),
            }
            continue

        l1 = l1_by_index.get(step["l1_index"], {})
        decision = step["decision"]

        # Check cache
        cache_key = _cache_key(current_l2_info, l1, decision)
        cache_path = os.path.join(config.supervisor_cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            scores.append(cached["yes_prob"])
        else:
            # Query supervisor
            prompt = SUPERVISOR_PROMPT.format(
                l2_start=current_l2_info.get("start_sec", 0),
                l2_end=current_l2_info.get("end_sec", 0),
                l2_caption=current_l2_info.get("caption", ""),
                l1_start=l1.get("start_sec", 0),
                l1_end=l1.get("end_sec", 0),
                l1_caption=l1.get("caption", ""),
                decision=decision,
            )

            messages = build_messages(prompt)
            vllm_output = generate_with_logprobs(supervisor_llm, supervisor_processor, messages)
            yes_no = extract_yes_no_probs(vllm_output)
            yes_prob = yes_no["yes"]
            scores.append(yes_prob)

            # Cache result
            os.makedirs(config.supervisor_cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"yes_prob": yes_prob, "decision": decision}, f)

        # Update L2 state
        if decision == "ADD_NEW":
            current_l2_info = {
                "start_sec": l1.get("start_sec", 0),
                "end_sec": l1.get("end_sec", 0),
                "caption": l1.get("caption", ""),
            }
        elif decision == "MERGE":
            current_l2_info["end_sec"] = max(
                current_l2_info.get("end_sec", 0), l1.get("end_sec", 0)
            )

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================
# R3: Caption Length Penalty
# ============================================================

def compute_reward_caption_length(trajectory: Dict, threshold: int = 200) -> float:
    """
    Caption length penalty.
    For each L2 node's caption, penalize the portion exceeding the threshold.
    Returns a negative value (penalty) or 0.
    """
    l2_nodes = trajectory.get("l2_nodes", [])
    if not l2_nodes:
        return 0.0

    penalties = []
    for node in l2_nodes:
        cap_len = len(node.get("caption", ""))
        penalty = -max(0, cap_len - threshold)
        penalties.append(penalty)

    return sum(penalties) / len(penalties) if penalties else 0.0


# ============================================================
# Total Reward
# ============================================================

def compute_total_reward(
    trajectory: Dict,
    correct_idx: int,
    choices: List[str],
    l1_nodes: List[Dict],
    config: GRPOConfig,
    supervisor_llm=None,
    supervisor_processor=None,
) -> Dict[str, float]:
    """
    Compute total reward = alpha*R1 + beta*R2 + gamma*R3

    Returns:
    {
        "total": float,
        "correct": float,
        "supervisor": float,
        "caption_length": float,
    }
    """
    # R1: VQA correctness
    r_correct = compute_reward_correct(
        trajectory["predicted_answer"], correct_idx, choices
    )

    # R2: Supervisor signal
    r_supervisor = compute_reward_supervisor(
        trajectory, l1_nodes, supervisor_llm, supervisor_processor, config
    )

    # R3: Caption length
    r_length = compute_reward_caption_length(
        trajectory, threshold=config.caption_length_threshold
    )

    # Weighted sum
    total = (
        config.reward_correct_weight * r_correct
        + config.reward_supervisor_weight * r_supervisor
        + config.reward_length_weight * r_length
    )

    return {
        "total": total,
        "correct": r_correct,
        "supervisor": r_supervisor,
        "caption_length": r_length,
    }
