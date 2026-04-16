"""
Memory building module - bottom-up construction of L1/L2/L3 three-layer multimodal memory
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from config import MemoryConfig
from models import ModelManager, build_messages, generate_text
from video_utils import (
    get_l1_segments,
    extract_frames_at_fps,
    save_frames,
)
from knowledge_graph import build_knowledge_graph, save_graph

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================

@dataclass
class L1Node:
    l1_index: int
    video_id: str
    start_sec: float
    end_sec: float
    caption: str = ""
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)


@dataclass
class L2Node:
    l2_index: int
    video_id: str
    start_sec: float
    end_sec: float
    caption: str = ""
    constituent_l1_indices: List[int] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)


@dataclass
class L3NodeData:
    l2_index: int
    entities: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    scenes: List[Dict] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)


# ============================================================
# Prompt templates
# ============================================================

L1_CAPTION_PROMPT = """Analyze the provided video frames and describe what you see in detail.
Focus on:
- Objects and people present in the scene
- Actions being performed
- Spatial relationships and positions
- Visual attributes (colors, sizes, expressions)
- Any text or symbols visible

Output your description as a JSON object:
```json
{
    "description": "A detailed description of the scene content",
    "objects": ["list", "of", "key", "objects"],
    "actions": ["list", "of", "actions", "observed"],
    "spatial_info": "Description of spatial layout and relationships"
}
```"""

L2_DECISION_PROMPT = """You are a video memory organizer. You must decide how to handle a new scene segment relative to the current memory episode.

Current memory episode:
- Caption: {current_l2_caption}
- Time range: {current_l2_start:.1f}s - {current_l2_end:.1f}s

New scene segment:
- Caption: {new_l1_caption}
- Time range: {new_l1_start:.1f}s - {new_l1_end:.1f}s

Based on the visual content of both the current episode frames and the new segment frames, decide one of:
- ADD_NEW: The new segment is significantly different from the current episode. Finalize the current episode and start a new one.
- MERGE: The new segment is a continuation of the current episode. Add it to the current episode.
- DISCARD: The new segment contains noise, transitions, or redundant content. Skip it.

Respond with exactly one word: ADD_NEW, MERGE, or DISCARD."""

L2_CAPTION_PROMPT = """Analyze the provided video frames which span a continuous episode in a video.
Focus on the narrative flow and progression:
- What events occur in sequence?
- How does the scene develop over time?
- What is the overall narrative arc of this episode?
- Key transitions or changes within the scene

Output your description as a JSON object:
```json
{
    "episode_summary": "A narrative summary of the entire episode",
    "event_sequence": ["event1", "event2", "event3"],
    "key_transitions": "Description of major changes within the episode"
}
```"""

L3_ENTITY_PROMPT = """Please analyze the given video frames and extract key information in a structured JSON format in English. Identify and describe:

Entities: List all distinct objects, people, animals, or other significant elements visible.
Actions: Describe the actions and interactions of the entities.
Scenes: Identify the locations, environments, or contexts.

If the video is filmed from a first-person point of view, describe the subject as "me" and include their actions.

Output strictly in this JSON format:
```json
{
    "entities": [{"entity name": "", "description": ""}],
    "actions": [{"entity name": "", "action description": ""}],
    "scenes": [{"location": ""}]
}
```
Each section should be detailed but concise. Avoid text outside the JSON."""


# ============================================================
# JSON parsing utilities
# ============================================================

def parse_json_from_response(response: str) -> Optional[Dict]:
    """
    Parse JSON from model response, supporting ```json...``` format.
    """
    # Try to match ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to match { ... }
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to parse directly
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def parse_decision(response: str) -> str:
    """Parse L2 decision result: ADD_NEW / MERGE / DISCARD"""
    response_upper = response.strip().upper()
    for decision in ["ADD_NEW", "MERGE", "DISCARD"]:
        if decision in response_upper:
            return decision
    # Default to MERGE (conservative strategy)
    logger.warning(f"Cannot parse L2 decision result: {response}, defaulting to MERGE")
    return "MERGE"


# ============================================================
# JSONL read/write
# ============================================================

def save_jsonl(data_list: List[Dict], filepath: str):
    """Save as JSONL format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================
# L1 memory building
# ============================================================

def build_l1_memory(
    video_path: str,
    output_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> List[L1Node]:
    """
    Build L1 layer detail memory.
    1. PySceneDetect segmentation + split segments >10s
    2. Extract frames at 5fps per segment
    3. Generate detail captions with base model
    """
    logger.info("===== Starting L1 memory building =====")
    llm, processor = model_manager.get_base_model()

    # Step 1: Segmentation
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

        logger.info(f"  L1[{idx}]: {start:.1f}s - {end:.1f}s")

        # Step 2: Extract frames at 5fps
        frames, timestamps = extract_frames_at_fps(
            video_path, start, end, fps=config.l1_fps
        )

        if not frames:
            logger.warning(f"  L1[{idx}]: Unable to extract frames, skipping")
            continue

        # Save frames
        frame_dir = os.path.join(output_dir, video_id, "L1_frames", f"l1_{idx:04d}")
        frame_paths = save_frames(frames, timestamps, frame_dir)

        # Step 3: Generate caption with base model (pass file paths to avoid vLLM cache issues)
        full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
        messages = build_messages(L1_CAPTION_PROMPT, images=full_paths)
        response = generate_text(
            llm, processor, messages,
            max_new_tokens=config.caption_max_new_tokens,
        )

        caption_data = parse_json_from_response(response)
        caption = json.dumps(caption_data, ensure_ascii=False) if caption_data else response

        node = L1Node(
            l1_index=idx,
            video_id=video_id,
            start_sec=start,
            end_sec=end,
            caption=caption,
            frame_paths=frame_paths,
            frame_timestamps=timestamps,
        )
        l1_nodes.append(node)
        logger.info(f"  L1[{idx}]: caption complete, {len(frames)} frames")

    # Save L1 memory
    l1_path = os.path.join(output_dir, video_id, "l1_memory.jsonl")
    save_jsonl([asdict(n) for n in l1_nodes], l1_path)
    logger.info(f"L1 memory building complete: {len(l1_nodes)} nodes, saved to {l1_path}")

    return l1_nodes


# ============================================================
# L2 memory building
# ============================================================

def _finalize_l2_node(
    current_l2: L2Node,
    video_path: str,
    output_dir: str,
    config: MemoryConfig,
    llm,
    processor,
) -> L2Node:
    """
    Finalize an L2 node:
    1. Extract 2fps frames from the original video as L2 visual memory
    2. Generate narrative caption with base model (vLLM)
    """
    # Extract 2fps frames
    frames, timestamps = extract_frames_at_fps(
        video_path, current_l2.start_sec, current_l2.end_sec, fps=config.l2_fps
    )

    if frames:
        frame_dir = os.path.join(
            output_dir, current_l2.video_id, "L2_frames", f"l2_{current_l2.l2_index:04d}"
        )
        frame_paths = save_frames(frames, timestamps, frame_dir)
        current_l2.frame_paths = frame_paths
        current_l2.frame_timestamps = timestamps

        # Generate narrative caption with base model (pass file paths)
        full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
        messages = build_messages(L2_CAPTION_PROMPT, images=full_paths)
        response = generate_text(
            llm, processor, messages,
            max_new_tokens=config.l2_caption_max_new_tokens,
        )
        caption_data = parse_json_from_response(response)
        current_l2.caption = json.dumps(caption_data, ensure_ascii=False) if caption_data else response

    return current_l2


def build_l2_memory(
    video_path: str,
    l1_nodes: List[L1Node],
    output_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> List[L2Node]:
    """
    Build L2 layer narrative memory.
    Process L1 nodes sequentially, using the fine-tuned model to decide ADD_NEW/MERGE/DISCARD.
    """
    logger.info("===== Starting L2 memory building =====")

    if not l1_nodes:
        logger.warning("No L1 nodes available, skipping L2 building")
        return []

    base_llm, base_processor = model_manager.get_base_model()
    ft_llm, ft_processor = model_manager.get_finetuned_model()  # Returns base model when no fine-tuned model

    l2_nodes = []
    current_l2: Optional[L2Node] = None
    l2_index = 0

    # Cache of current L2 frame paths for L2 decision
    current_l2_frame_paths: List[str] = []

    for i, l1_node in enumerate(l1_nodes):
        # Current L1 frame file paths
        l1_frame_dir = os.path.join(output_dir, video_id, "L1_frames", f"l1_{l1_node.l1_index:04d}")
        l1_frame_full_paths = [os.path.join(l1_frame_dir, p) for p in l1_node.frame_paths]

        if not l1_frame_full_paths or not os.path.exists(l1_frame_full_paths[0]):
            logger.warning(f"  L1[{l1_node.l1_index}]: No frames available, skipping")
            continue

        if current_l2 is None:
            # First L1 node, default ADD_NEW
            logger.info(f"  L1[{l1_node.l1_index}]: Default ADD_NEW (first L1)")
            current_l2 = L2Node(
                l2_index=l2_index,
                video_id=video_id,
                start_sec=l1_node.start_sec,
                end_sec=l1_node.end_sec,
                constituent_l1_indices=[l1_node.l1_index],
            )
            # Use L1 frame paths as visual reference for current L2
            current_l2_frame_paths = list(l1_frame_full_paths)
            continue

        # Subsequent L1 nodes: use fine-tuned model to decide
        decision_text = L2_DECISION_PROMPT.format(
            current_l2_caption=current_l2.caption or "(caption not yet generated)",
            current_l2_start=current_l2.start_sec,
            current_l2_end=current_l2.end_sec,
            new_l1_caption=l1_node.caption,
            new_l1_start=l1_node.start_sec,
            new_l1_end=l1_node.end_sec,
        )

        # Combine L2 frame paths and L1 frame paths as visual input
        combined_paths = current_l2_frame_paths + l1_frame_full_paths
        if len(combined_paths) > config.l2_max_input_frames:
            import numpy as np
            indices = np.linspace(0, len(combined_paths) - 1, config.l2_max_input_frames, dtype=int)
            combined_paths = [combined_paths[i] for i in indices]

        messages = build_messages(decision_text, images=combined_paths)
        response = generate_text(
            ft_llm, ft_processor, messages,
            max_new_tokens=config.l2_decision_max_new_tokens,
        )

        decision = parse_decision(response)
        logger.info(f"  L1[{l1_node.l1_index}]: Decision result={decision}")

        if decision == "ADD_NEW":
            # Finalize the current L2 node
            current_l2 = _finalize_l2_node(
                current_l2, video_path, output_dir, config, base_llm, base_processor
            )
            l2_nodes.append(current_l2)
            logger.info(f"  L2[{current_l2.l2_index}]: Completed ({current_l2.start_sec:.1f}s - {current_l2.end_sec:.1f}s)")

            # Start a new L2
            l2_index += 1
            current_l2 = L2Node(
                l2_index=l2_index,
                video_id=video_id,
                start_sec=l1_node.start_sec,
                end_sec=l1_node.end_sec,
                constituent_l1_indices=[l1_node.l1_index],
            )
            current_l2_frame_paths = list(l1_frame_full_paths)

        elif decision == "MERGE":
            # Merge L1 into current L2
            current_l2.constituent_l1_indices.append(l1_node.l1_index)
            current_l2.end_sec = max(current_l2.end_sec, l1_node.end_sec)
            current_l2_frame_paths.extend(l1_frame_full_paths)

        elif decision == "DISCARD":
            # Skip this L1 node
            logger.info(f"  L1[{l1_node.l1_index}]: DISCARD, skipping")

    # Finalize the last L2 node
    if current_l2 is not None:
        current_l2 = _finalize_l2_node(
            current_l2, video_path, output_dir, config, base_llm, base_processor
        )
        l2_nodes.append(current_l2)
        logger.info(f"  L2[{current_l2.l2_index}]: Completed ({current_l2.start_sec:.1f}s - {current_l2.end_sec:.1f}s)")

    # Save L2 memory
    l2_path = os.path.join(output_dir, video_id, "l2_memory.jsonl")
    save_jsonl([asdict(n) for n in l2_nodes], l2_path)
    logger.info(f"L2 memory building complete: {len(l2_nodes)} nodes, saved to {l2_path}")

    return l2_nodes


# ============================================================
# L3 memory building (knowledge graph)
# ============================================================

def build_l3_memory(
    video_path: str,
    l2_nodes: List[L2Node],
    output_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Tuple[Any, Any, List[L3NodeData]]:
    """
    Build L3 layer knowledge graph.
    1. Extract 1fps frames for each L2 node
    2. Extract entities/actions/scenes with base model
    3. Build knowledge graph
    """
    logger.info("===== Starting L3 memory building =====")

    if not l2_nodes:
        logger.warning("No L2 nodes available, skipping L3 building")
        import networkx as nx
        return nx.DiGraph(), {}, []

    base_llm, base_processor = model_manager.get_base_model()
    embedding_model = model_manager.get_embedding_model()

    l3_data_list = []

    for l2_node in l2_nodes:
        idx = l2_node.l2_index
        logger.info(f"  L2[{idx}]: Extracting entities ({l2_node.start_sec:.1f}s - {l2_node.end_sec:.1f}s)")

        # Extract 1fps frames
        frames, timestamps = extract_frames_at_fps(
            video_path, l2_node.start_sec, l2_node.end_sec, fps=config.l3_fps
        )

        if not frames:
            logger.warning(f"  L2[{idx}]: Unable to extract 1fps frames")
            l3_data_list.append(L3NodeData(l2_index=idx))
            continue

        # Save L3 frames
        frame_dir = os.path.join(output_dir, video_id, "L3_frames", f"l2_{idx:04d}")
        frame_paths = save_frames(frames, timestamps, frame_dir)

        # Extract entities/actions/scenes with base model (with retries)
        entity_data = None
        full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
        for attempt in range(config.l3_entity_max_retries):
            messages = build_messages(L3_ENTITY_PROMPT, images=full_paths)
            response = generate_text(
                base_llm, base_processor, messages,
                max_new_tokens=config.l3_entity_max_new_tokens,
            )
            entity_data = parse_json_from_response(response)
            if entity_data and "entities" in entity_data:
                break
            logger.warning(f"  L2[{idx}]: JSON parsing failed (attempt {attempt + 1}), retrying")

        if entity_data is None:
            entity_data = {"entities": [], "actions": [], "scenes": []}

        l3_node = L3NodeData(
            l2_index=idx,
            entities=entity_data.get("entities", []),
            actions=entity_data.get("actions", []),
            scenes=entity_data.get("scenes", []),
            frame_paths=frame_paths,
            frame_timestamps=timestamps,
        )
        l3_data_list.append(l3_node)
        logger.info(
            f"  L2[{idx}]: Extracted {len(l3_node.entities)} entities, "
            f"{len(l3_node.actions)} actions, {len(l3_node.scenes)} scenes"
        )

    # Build knowledge graph
    kg_input = [
        {
            "l2_index": d.l2_index,
            "entities": d.entities,
            "actions": d.actions,
            "scenes": d.scenes,
        }
        for d in l3_data_list
    ]

    video_graph, entity_graph = build_knowledge_graph(
        kg_input, embedding_model, dedup_threshold=config.l3_entity_dedup_threshold
    )

    # Save
    graph_path = os.path.join(output_dir, video_id, "knowledge_graph.pkl")
    save_graph(video_graph, entity_graph, graph_path)

    l3_path = os.path.join(output_dir, video_id, "l3_memory.jsonl")
    save_jsonl([asdict(d) for d in l3_data_list], l3_path)

    logger.info(f"L3 memory building complete: {len(l3_data_list)} nodes, saved to {l3_path}")
    return video_graph, entity_graph, l3_data_list


# ============================================================
# Complete memory building pipeline
# ============================================================

def build_all_memory(
    video_path: str,
    output_dir: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    Complete bottom-up memory building: L1 -> L2 -> L3
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"Starting memory building for video: {video_id}")

    # L1
    l1_nodes = build_l1_memory(video_path, output_dir, video_id, config, model_manager)

    # L2
    l2_nodes = build_l2_memory(video_path, l1_nodes, output_dir, video_id, config, model_manager)

    # L3
    video_graph, entity_graph, l3_data = build_l3_memory(
        video_path, l2_nodes, output_dir, video_id, config, model_manager
    )

    result = {
        "video_id": video_id,
        "num_l1_nodes": len(l1_nodes),
        "num_l2_nodes": len(l2_nodes),
        "num_l3_entities": len(entity_graph) if entity_graph else 0,
        "num_graph_edges": video_graph.number_of_edges() if video_graph else 0,
        "output_dir": os.path.join(output_dir, video_id),
    }

    logger.info(f"Memory building complete: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result
