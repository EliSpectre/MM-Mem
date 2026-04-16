"""
Data Loading Module
- Load training data from JSON folder
- Video path lookup
- L1 cache management
"""

import os
import sys
import json
import glob
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add build_retrieve to path for code reuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_retrieve"))


def load_training_data(data_dir: str) -> List[Dict]:
    """
    Load all JSON files from the folder.
    Each JSON file format:
    {
        "task_name_0": {
            "inputs": {"video 1": {"id": "video_id"}},
            "question": "...",
            "choices": ["...", "..."],
            "correct_idx": 0
        }
    }
    Returns a flattened list of samples.
    """
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")

    samples = []
    for json_path in sorted(json_files):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, item in data.items():
            video_id = item["inputs"]["video 1"]["id"]
            question = item["question"]
            choices = item["choices"]
            correct_idx = item["correct_idx"]

            samples.append({
                "sample_id": key,
                "video_id": video_id,
                "question": question,
                "choices": choices,
                "correct_idx": correct_idx,
                "correct_answer": choices[correct_idx],
                "source_file": os.path.basename(json_path),
            })

    logger.info(f"Loaded {len(samples)} samples from {len(json_files)} JSON files")
    return samples


def find_video_path(video_dir: str, video_id: str) -> Optional[str]:
    """
    Recursively search the video root directory for the video file matching video_id.
    Supports .mp4, .avi, .mkv, .mov and other formats.
    """
    for ext in [".mp4", ".avi", ".mkv", ".mov", ".webm"]:
        # Try direct path first
        direct = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(direct):
            return direct

    # Recursive search
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            if name == video_id and ext.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                return os.path.join(root, f)

    return None


def load_l1_cache(l1_cache_dir: str, video_id: str) -> Optional[List[Dict]]:
    """Check and load cached L1 nodes"""
    l1_path = os.path.join(l1_cache_dir, video_id, "l1_memory.jsonl")
    if not os.path.exists(l1_path):
        return None

    nodes = []
    with open(l1_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))

    if nodes:
        logger.info(f"  Loaded L1 nodes from cache: {video_id} ({len(nodes)} nodes)")
    return nodes if nodes else None


def ensure_l1_built(
    video_path: str,
    video_id: str,
    l1_cache_dir: str,
    base_llm,
    base_processor,
    config,
) -> List[Dict]:
    """
    Ensure L1 is built. Load from cache if available, otherwise build.
    Reuses build_l1_memory from build_retrieve/memory_build.py.
    """
    # Check cache
    cached = load_l1_cache(l1_cache_dir, video_id)
    if cached is not None:
        return cached

    # Need to build -- reuse build_retrieve code
    from video_utils import get_l1_segments, extract_frames_at_fps, save_frames
    from models import build_messages, generate_text
    from memory_build import L1_CAPTION_PROMPT, parse_json_from_response

    logger.info(f"  Building L1 nodes: {video_id}")

    segments = get_l1_segments(
        video_path,
        pyscenedetect_threshold=config.pyscenedetect_threshold,
        max_segment_duration=config.l1_max_segment_duration,
    )

    l1_nodes = []
    for seg in segments:
        idx = seg["l1_index"]
        start = seg["start_sec"]
        end = seg["end_sec"]

        frames, timestamps = extract_frames_at_fps(
            video_path, start, end, fps=config.l1_fps
        )
        if not frames:
            continue

        frame_dir = os.path.join(l1_cache_dir, video_id, "L1_frames", f"l1_{idx:04d}")
        frame_paths = save_frames(frames, timestamps, frame_dir)

        # Use base model (vLLM) to generate caption
        full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
        messages = build_messages(L1_CAPTION_PROMPT, images=full_paths)
        response = generate_text(base_llm, base_processor, messages, max_new_tokens=512)

        caption_data = parse_json_from_response(response)
        caption = json.dumps(caption_data, ensure_ascii=False) if caption_data else response

        node = {
            "l1_index": idx,
            "video_id": video_id,
            "start_sec": start,
            "end_sec": end,
            "caption": caption,
            "frame_paths": frame_paths,
            "frame_timestamps": timestamps,
        }
        l1_nodes.append(node)

    # Save cache
    l1_path = os.path.join(l1_cache_dir, video_id, "l1_memory.jsonl")
    os.makedirs(os.path.dirname(l1_path), exist_ok=True)
    with open(l1_path, "w", encoding="utf-8") as f:
        for node in l1_nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")

    logger.info(f"  L1 build complete: {len(l1_nodes)} nodes")
    return l1_nodes
