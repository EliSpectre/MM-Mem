"""
Memory retrieval module - top-down L3->L2->L1 retrieval with entropy gating
"""

import os
import math
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

from config import MemoryConfig
from models import (
    ModelManager,
    build_messages,
    generate_with_logprobs,
    extract_option_probs,
    extract_yes_no_probs,
)
from video_utils import extract_frames_at_fps, load_frames, subsample_frames
from memory_build import load_jsonl
from knowledge_graph import (
    load_graph,
    retrieve_from_graph,
    compute_text_similarity,
)

logger = logging.getLogger(__name__)


# ============================================================
# Prompt templates
# ============================================================

VQA_PROMPT = """Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter ({option_labels}) of the correct option.
Question: {question}
Possible answer choices:
{options_text}
The best answer is:"""

VQA_WITH_CONTEXT_PROMPT = """You are given a question about a video, along with relevant retrieved information from the video's memory.

Retrieved context:
{context_text}

Based on the video frames and retrieved context, select the best answer.
Respond with only the letter ({option_labels}) of the correct option.
Question: {question}
Possible answer choices:
{options_text}
The best answer is:"""

VISUAL_RELEVANCE_PROMPT = """You are selecting the most relevant video clip for answering a multiple-choice question.
Look at the frames and decide whether they contain information relevant for answering the question.
Reply with exactly one token: yes or no.
Question: {question}
Possible answer choices:
{options_text}
Relevant clip?"""


# ============================================================
# Entropy computation
# ============================================================

def compute_entropy(option_probs: Dict[str, float]) -> float:
    """
    Compute Shannon entropy: Hs = -sum(p_i * log(p_i))
    Max entropy for 4 options = log(4) = 1.386
    """
    entropy = 0.0
    for p in option_probs.values():
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def is_confident(
    option_probs: Dict[str, float],
    entropy_threshold: float,
) -> Tuple[bool, float, str]:
    """
    Determine whether the model is sufficiently confident.
    Returns (is_confident, entropy, predicted_option)
    """
    entropy = compute_entropy(option_probs)
    predicted = max(option_probs, key=option_probs.get) if option_probs else ""
    return entropy < entropy_threshold, entropy, predicted


# ============================================================
# Option label construction
# ============================================================

def get_option_labels(num_options: int) -> Tuple[str, ...]:
    """Generate option labels: (A, B, C, D) or (A, B, C, D, E) etc."""
    return tuple(chr(65 + i) for i in range(num_options))


def format_options_text(options: List[str]) -> str:
    """Format options text"""
    lines = []
    for i, opt in enumerate(options):
        label = chr(65 + i)
        # If the option already has a label prefix, do not duplicate
        if opt.strip().startswith(f"{label}.") or opt.strip().startswith(f"{label} "):
            lines.append(opt.strip())
        else:
            lines.append(f"{label}. {opt.strip()}")
    return "\n".join(lines)


# ============================================================
# VQA core function
# ============================================================

def do_vqa(
    llm,
    processor,
    frames: List[Image.Image],
    question: str,
    options: List[str],
    config: MemoryConfig,
    context_text: str = "",
) -> Dict[str, Any]:
    """
    Execute VQA: generate 1 token, extract option probabilities, compute entropy.
    """
    option_labels = get_option_labels(len(options))
    options_text = format_options_text(options)
    labels_str = "/".join(option_labels)

    if context_text:
        prompt = VQA_WITH_CONTEXT_PROMPT.format(
            context_text=context_text,
            option_labels=labels_str,
            question=question,
            options_text=options_text,
        )
    else:
        prompt = VQA_PROMPT.format(
            option_labels=labels_str,
            question=question,
            options_text=options_text,
        )

    messages = build_messages(prompt, images=frames if frames else None)
    vllm_output = generate_with_logprobs(llm, processor, messages)

    option_probs = extract_option_probs(vllm_output, labels=option_labels)
    confident, entropy, predicted = is_confident(option_probs, config.entropy_threshold)

    return {
        "predicted_option": predicted,
        "option_probs": option_probs,
        "entropy": entropy,
        "is_confident": confident,
        "raw_text": vllm_output.outputs[0].text.strip(),
    }


# ============================================================
# Initial VQA
# ============================================================

def initial_vqa(
    video_path: str,
    question: str,
    options: List[str],
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    Initial VQA: uniformly sample frames from the entire video for direct answering.
    Uses confidence (max option probability) for judgment, not entropy.
    """
    logger.info("===== Initial VQA =====")
    llm, processor = model_manager.get_base_model()

    # Uniformly sample frames from the entire video
    import decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    video_fps = vr.get_avg_fps()
    duration = len(vr) / video_fps

    frames, timestamps = extract_frames_at_fps(
        video_path, 0.0, duration, fps=config.initial_vqa_max_frames / max(duration, 1.0)
    )
    frames, timestamps = subsample_frames(frames, timestamps, config.initial_vqa_max_frames)

    result = do_vqa(llm, processor, frames, question, options, config)

    # Initial VQA uses confidence (max probability) for judgment
    max_prob = max(result["option_probs"].values()) if result["option_probs"] else 0.0
    result["confidence"] = max_prob
    result["is_confident"] = max_prob >= config.initial_vqa_confidence

    logger.info(
        f"  Initial VQA: predicted={result['predicted_option']}, "
        f"confidence={max_prob:.4f}, entropy={result['entropy']:.4f}, "
        f"confident={result['is_confident']}"
    )
    return result


# ============================================================
# L3 knowledge graph retrieval
# ============================================================

def retrieve_l3(
    question: str,
    options: List[str],
    video_path: str,
    memory_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    L3 layer knowledge graph retrieval -> VQA -> entropy judgment.
    """
    logger.info("===== L3 Knowledge Graph Retrieval =====")

    # Load knowledge graph
    graph_path = os.path.join(memory_dir, video_id, "knowledge_graph.pkl")
    video_graph, entity_graph = load_graph(graph_path)

    # Load L3 memory
    l3_path = os.path.join(memory_dir, video_id, "l3_memory.jsonl")
    l3_data = load_jsonl(l3_path)
    l3_by_index = {d["l2_index"]: d for d in l3_data}

    # Graph retrieval
    embedding_model = model_manager.get_embedding_model()
    reranker_model = model_manager.get_reranker_model()

    retrieved_l2_indices = retrieve_from_graph(
        query=question,
        video_graph=video_graph,
        entity_graph=entity_graph,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        top_k_embedding=config.l3_coarse_top_k,
        top_k_rerank=config.l3_rerank_top_k,
        similarity_threshold=config.l3_retrieval_similarity_threshold,
    )

    logger.info(f"  L3 retrieved {len(retrieved_l2_indices)} L2 nodes: {retrieved_l2_indices}")

    # Build context text
    context_parts = []
    all_frames = []
    for l2_idx in retrieved_l2_indices:
        data = l3_by_index.get(l2_idx, {})
        entities = data.get("entities", [])
        actions = data.get("actions", [])
        scenes = data.get("scenes", [])

        parts = []
        if entities:
            entity_strs = [f"{e.get('entity name', '')}: {e.get('description', '')}" for e in entities]
            parts.append("Entities: " + "; ".join(entity_strs))
        if actions:
            action_strs = [f"{a.get('entity name', '')}: {a.get('action description', '')}" for a in actions]
            parts.append("Actions: " + "; ".join(action_strs))
        if scenes:
            scene_strs = [s.get("location", "") for s in scenes]
            parts.append("Scenes: " + "; ".join(scene_strs))

        if parts:
            context_parts.append(f"[Segment {l2_idx}] " + " | ".join(parts))

        # Load 1fps frames
        frame_dir = os.path.join(memory_dir, video_id, "L3_frames", f"l2_{l2_idx:04d}")
        frame_paths = data.get("frame_paths", [])
        if frame_paths:
            frames = load_frames(frame_paths, frame_dir)
            all_frames.extend(frames)

    context_text = "\n".join(context_parts) if context_parts else ""

    # VQA
    llm, processor = model_manager.get_base_model()
    vqa_result = do_vqa(llm, processor, all_frames, question, options, config, context_text)

    logger.info(
        f"  L3 VQA: predicted={vqa_result['predicted_option']}, "
        f"entropy={vqa_result['entropy']:.4f}, confident={vqa_result['is_confident']}"
    )

    return {
        "vqa_result": vqa_result,
        "retrieved_l2_indices": retrieved_l2_indices,
        "context_text": context_text,
    }


# ============================================================
# L2 text + visual retrieval
# ============================================================

def retrieve_l2(
    question: str,
    options: List[str],
    video_path: str,
    candidate_l2_indices: List[int],
    memory_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    L2 layer retrieval: BGE text retrieval + yes-token visual verification -> VQA -> entropy judgment.
    """
    logger.info("===== L2 Text + Visual Retrieval =====")

    # Load L2 memory
    l2_path = os.path.join(memory_dir, video_id, "l2_memory.jsonl")
    l2_data = load_jsonl(l2_path)
    l2_by_index = {d["l2_index"]: d for d in l2_data}

    # Filter candidate L2 nodes
    candidates = [l2_by_index[idx] for idx in candidate_l2_indices if idx in l2_by_index]
    if not candidates:
        candidates = list(l2_by_index.values())
        logger.warning("  Candidate L2 nodes empty, using all L2 nodes")

    # --- Step 1: BGE text retrieval ---
    embedding_model = model_manager.get_embedding_model()
    reranker_model = model_manager.get_reranker_model()

    candidate_captions = [c.get("caption", "") for c in candidates]
    candidate_indices = [c["l2_index"] for c in candidates]

    if len(candidates) > config.l2_embedding_top_k:
        # Coarse ranking
        sim_matrix = compute_text_similarity(
            [question], candidate_captions, embedding_model
        )
        if sim_matrix.size > 0:
            ranked = sorted(
                zip(candidate_indices, candidates, sim_matrix[0]),
                key=lambda x: -x[2],
            )
            candidate_indices = [r[0] for r in ranked[:config.l2_embedding_top_k]]
            candidates = [r[1] for r in ranked[:config.l2_embedding_top_k]]
            candidate_captions = [c.get("caption", "") for c in candidates]

    # Fine ranking (cross-encoder)
    if len(candidates) > config.l2_visual_top_k:
        pairs = [(question, cap) for cap in candidate_captions]
        rerank_scores = reranker_model.predict(pairs)
        ranked = sorted(
            zip(candidate_indices, candidates, rerank_scores),
            key=lambda x: -x[2],
        )
        # Take fine-ranked candidates, then further filter with visual verification
        text_ranked_indices = [r[0] for r in ranked]
        text_ranked_candidates = [r[1] for r in ranked]
    else:
        text_ranked_indices = candidate_indices
        text_ranked_candidates = candidates

    # --- Step 2: Visual verification (yes-token probability) ---
    llm, processor = model_manager.get_base_model()
    options_text = format_options_text(options)

    visual_scores = []
    for l2_idx, l2_candidate in zip(text_ranked_indices, text_ranked_candidates):
        # Load 2fps frames
        frame_dir = os.path.join(memory_dir, video_id, "L2_frames", f"l2_{l2_idx:04d}")
        frame_paths = l2_candidate.get("frame_paths", [])
        if not frame_paths:
            visual_scores.append((l2_idx, l2_candidate, 0.0))
            continue

        frames = load_frames(frame_paths, frame_dir)
        if not frames:
            visual_scores.append((l2_idx, l2_candidate, 0.0))
            continue

        # Build relevance verification prompt
        relevance_prompt = VISUAL_RELEVANCE_PROMPT.format(
            question=question,
            options_text=options_text,
        )
        messages = build_messages(relevance_prompt, images=frames)
        vllm_out = generate_with_logprobs(llm, processor, messages)
        yes_no_probs = extract_yes_no_probs(vllm_out)
        visual_scores.append((l2_idx, l2_candidate, yes_no_probs["yes"]))

    # Sort by yes_prob descending, select top-k
    visual_scores.sort(key=lambda x: -x[2])
    selected = visual_scores[:config.l2_visual_top_k]

    selected_l2_indices = [s[0] for s in selected]
    logger.info(
        f"  L2 visual verification: "
        + ", ".join(f"L2[{s[0]}]={s[2]:.4f}" for s in selected)
    )

    # --- Step 3: VQA ---
    context_parts = []
    all_frames = []
    all_l1_indices = []

    for l2_idx, l2_candidate, _ in selected:
        caption = l2_candidate.get("caption", "")
        if caption:
            context_parts.append(f"[Episode {l2_idx}] {caption}")

        # Load 2fps frames
        frame_dir = os.path.join(memory_dir, video_id, "L2_frames", f"l2_{l2_idx:04d}")
        frame_paths = l2_candidate.get("frame_paths", [])
        if frame_paths:
            frames = load_frames(frame_paths, frame_dir)
            all_frames.extend(frames)

        # Record associated L1 indices
        l1_indices = l2_candidate.get("constituent_l1_indices", [])
        all_l1_indices.extend(l1_indices)

    context_text = "\n".join(context_parts) if context_parts else ""
    vqa_result = do_vqa(llm, processor, all_frames, question, options, config, context_text)

    logger.info(
        f"  L2 VQA: predicted={vqa_result['predicted_option']}, "
        f"entropy={vqa_result['entropy']:.4f}, confident={vqa_result['is_confident']}"
    )

    return {
        "vqa_result": vqa_result,
        "selected_l2_indices": selected_l2_indices,
        "l1_indices": list(set(all_l1_indices)),
        "context_text": context_text,
    }


# ============================================================
# L1 visual retrieval
# ============================================================

def retrieve_l1(
    question: str,
    options: List[str],
    video_path: str,
    candidate_l1_indices: List[int],
    memory_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    L1 layer retrieval: MLLM direct visual relevance judgment -> VQA (final answer).
    """
    logger.info("===== L1 Visual Retrieval =====")

    # Load L1 memory
    l1_path = os.path.join(memory_dir, video_id, "l1_memory.jsonl")
    l1_data = load_jsonl(l1_path)
    l1_by_index = {d["l1_index"]: d for d in l1_data}

    # Filter candidate L1 nodes
    candidates = [l1_by_index[idx] for idx in candidate_l1_indices if idx in l1_by_index]
    if not candidates:
        candidates = list(l1_by_index.values())
        logger.warning("  Candidate L1 nodes empty, using all L1 nodes")

    llm, processor = model_manager.get_base_model()
    options_text = format_options_text(options)

    # MLLM visual relevance verification
    visual_scores = []
    for l1_candidate in candidates:
        l1_idx = l1_candidate["l1_index"]
        frame_dir = os.path.join(memory_dir, video_id, "L1_frames", f"l1_{l1_idx:04d}")
        frame_paths = l1_candidate.get("frame_paths", [])
        if not frame_paths:
            visual_scores.append((l1_idx, l1_candidate, 0.0))
            continue

        frames = load_frames(frame_paths, frame_dir)
        if not frames:
            visual_scores.append((l1_idx, l1_candidate, 0.0))
            continue

        relevance_prompt = VISUAL_RELEVANCE_PROMPT.format(
            question=question,
            options_text=options_text,
        )
        messages = build_messages(relevance_prompt, images=frames)
        vllm_out = generate_with_logprobs(llm, processor, messages)
        yes_no_probs = extract_yes_no_probs(vllm_out)
        visual_scores.append((l1_idx, l1_candidate, yes_no_probs["yes"]))

    # Sort by yes_prob descending, select top-k
    visual_scores.sort(key=lambda x: -x[2])
    selected = visual_scores[:config.l1_top_k]

    logger.info(
        f"  L1 visual verification: "
        + ", ".join(f"L1[{s[0]}]={s[2]:.4f}" for s in selected)
    )

    # VQA (final answer)
    context_parts = []
    all_frames = []

    for l1_idx, l1_candidate, _ in selected:
        caption = l1_candidate.get("caption", "")
        if caption:
            context_parts.append(f"[Scene {l1_idx}] {caption}")

        frame_dir = os.path.join(memory_dir, video_id, "L1_frames", f"l1_{l1_idx:04d}")
        frame_paths = l1_candidate.get("frame_paths", [])
        if frame_paths:
            frames = load_frames(frame_paths, frame_dir)
            all_frames.extend(frames)

    context_text = "\n".join(context_parts) if context_parts else ""
    vqa_result = do_vqa(llm, processor, all_frames, question, options, config, context_text)

    logger.info(
        f"  L1 VQA (final): predicted={vqa_result['predicted_option']}, "
        f"entropy={vqa_result['entropy']:.4f}"
    )

    return {
        "vqa_result": vqa_result,
        "selected_l1_indices": [s[0] for s in selected],
        "context_text": context_text,
    }


# ============================================================
# Complete hierarchical retrieval pipeline
# ============================================================

def hierarchical_retrieve_and_answer(
    video_path: str,
    question: str,
    options: List[str],
    memory_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    Complete top-down retrieval pipeline:
    Initial VQA -> (auto-build memory) -> L3 knowledge graph retrieval -> L2 text + visual retrieval -> L1 visual retrieval

    Returns directly when initial VQA is confident, without building memory.
    When not confident, checks if memory already exists; builds automatically if not.
    """
    retrieval_trace = []

    # --- Stage 1: Initial VQA ---
    initial_result = initial_vqa(video_path, question, options, config, model_manager)
    retrieval_trace.append({
        "stage": "initial",
        "predicted_option": initial_result["predicted_option"],
        "confidence": initial_result.get("confidence", 0.0),
        "entropy": initial_result["entropy"],
        "is_confident": initial_result["is_confident"],
    })

    if initial_result["is_confident"]:
        return {
            "final_answer": initial_result["predicted_option"],
            "stage": "initial",
            "entropy": initial_result["entropy"],
            "option_probs": initial_result["option_probs"],
            "retrieval_trace": retrieval_trace,
        }

    # --- Auto-build memory (if not exists) ---
    graph_path = os.path.join(memory_dir, video_id, "knowledge_graph.pkl")
    if not os.path.exists(graph_path):
        logger.info(f"Memory does not exist, auto-building: {video_id}")
        from memory_build import build_all_memory
        build_all_memory(
            video_path=video_path,
            output_dir=memory_dir,
            config=config,
            model_manager=model_manager,
        )

    # --- Stage 2: L3 Knowledge Graph Retrieval ---
    l3_result = retrieve_l3(
        question, options, video_path, memory_dir, video_id, config, model_manager
    )
    retrieval_trace.append({
        "stage": "L3",
        "predicted_option": l3_result["vqa_result"]["predicted_option"],
        "entropy": l3_result["vqa_result"]["entropy"],
        "is_confident": l3_result["vqa_result"]["is_confident"],
        "retrieved_l2_indices": l3_result["retrieved_l2_indices"],
    })

    if l3_result["vqa_result"]["is_confident"]:
        return {
            "final_answer": l3_result["vqa_result"]["predicted_option"],
            "stage": "L3",
            "entropy": l3_result["vqa_result"]["entropy"],
            "option_probs": l3_result["vqa_result"]["option_probs"],
            "retrieval_trace": retrieval_trace,
        }

    # --- Stage 3: L2 Text + Visual Retrieval ---
    l2_result = retrieve_l2(
        question, options, video_path,
        candidate_l2_indices=l3_result["retrieved_l2_indices"],
        memory_dir=memory_dir,
        video_id=video_id,
        config=config,
        model_manager=model_manager,
    )
    retrieval_trace.append({
        "stage": "L2",
        "predicted_option": l2_result["vqa_result"]["predicted_option"],
        "entropy": l2_result["vqa_result"]["entropy"],
        "is_confident": l2_result["vqa_result"]["is_confident"],
        "selected_l2_indices": l2_result["selected_l2_indices"],
    })

    if l2_result["vqa_result"]["is_confident"]:
        return {
            "final_answer": l2_result["vqa_result"]["predicted_option"],
            "stage": "L2",
            "entropy": l2_result["vqa_result"]["entropy"],
            "option_probs": l2_result["vqa_result"]["option_probs"],
            "retrieval_trace": retrieval_trace,
        }

    # --- Stage 4: L1 Visual Retrieval (final) ---
    l1_result = retrieve_l1(
        question, options, video_path,
        candidate_l1_indices=l2_result["l1_indices"],
        memory_dir=memory_dir,
        video_id=video_id,
        config=config,
        model_manager=model_manager,
    )
    retrieval_trace.append({
        "stage": "L1",
        "predicted_option": l1_result["vqa_result"]["predicted_option"],
        "entropy": l1_result["vqa_result"]["entropy"],
        "selected_l1_indices": l1_result["selected_l1_indices"],
    })

    return {
        "final_answer": l1_result["vqa_result"]["predicted_option"],
        "stage": "L1",
        "entropy": l1_result["vqa_result"]["entropy"],
        "option_probs": l1_result["vqa_result"]["option_probs"],
        "retrieval_trace": retrieval_trace,
    }
