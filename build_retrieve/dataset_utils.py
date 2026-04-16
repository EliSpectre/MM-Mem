"""
Dataset loading module
- Supports loading and iterating over datasets such as VideoMME
- To add new datasets, simply add a corresponding load_dataset_xxx function
"""

import os
import json
import logging
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# VideoMME dataset
# ============================================================

def load_dataset_videomme(data_dir: str, duration: str = "short") -> List[Dict]:
    """
    Load the VideoMME dataset, filtered by duration.
    Refer to reference/VideoMME/dataset_utils.py load_videomme_dataset().

    Args:
        data_dir: Dataset directory path
        duration: Filter condition "short" / "medium" / "long" / "all"

    Returns:
        List of samples, each containing:
        {
            "videoID": str,
            "question": str,
            "options": List[str],     # ["A. ...", "B. ...", "C. ...", "D. ..."]
            "answer": str,            # "A" / "B" / "C" / "D"
            "question_id": str,
            "duration": str,
            "domain": str,
            "sub_category": str,
        }
    """
    from datasets import load_dataset

    logger.info(f"Loading VideoMME dataset: {data_dir}, duration={duration}")
    dataset = load_dataset(data_dir)
    samples = []

    for item in dataset["test"]:
        if duration != "all" and item.get("duration", "") != duration:
            continue
        samples.append({
            "videoID": item["videoID"],
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"],
            "question_id": item.get("question_id", ""),
            "duration": item.get("duration", ""),
            "domain": item.get("domain", ""),
            "sub_category": item.get("sub_category", ""),
        })

    logger.info(f"Loading complete: {len(samples)} samples (duration={duration})")
    return samples


# ============================================================
# Common interface
# ============================================================

def load_dataset_by_name(
    dataset_name: str,
    data_dir: str,
    duration: str = "short",
    **kwargs,
) -> List[Dict]:
    """
    Dispatch to the corresponding loading function based on dataset name.
    Currently supported: "videomme"
    To add new datasets, simply add a new load_dataset_xxx function.
    """
    loaders = {
        "videomme": load_dataset_videomme,
    }

    if dataset_name not in loaders:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}, "
            f"available: {list(loaders.keys())}"
        )

    return loaders[dataset_name](data_dir, duration=duration, **kwargs)


def get_video_path(video_dir: str, video_id: str) -> str:
    """Build video file path: {video_dir}/{video_id}.mp4"""
    return os.path.join(video_dir, f"{video_id}.mp4")


def group_by_video(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group by videoID, placing multiple questions for the same video together"""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["videoID"]].append(sample)
    return dict(groups)


# ============================================================
# Result collection and accuracy statistics
# ============================================================

def load_completed_results(output_file: str) -> set:
    """Load completed question_ids for checkpoint resumption"""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    qid = item.get("question_id", "")
                    if qid:
                        completed.add(qid)
                except (json.JSONDecodeError, KeyError):
                    pass
    if completed:
        logger.info(f"Found {len(completed)} completed results, will skip them")
    return completed


def append_result(output_file: str, result: Dict):
    """Append one result to the JSONL file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def compute_accuracy(output_file: str) -> Dict:
    """
    Compute accuracy from the result file.
    Statistics by overall / domain / sub_category / stage.
    """
    results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        return {"overall": 0.0, "total": 0}

    # Overall accuracy
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)

    stats = {
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
    }

    # Statistics by domain
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        domain = r.get("domain", "unknown")
        domain_stats[domain]["total"] += 1
        if r.get("correct", False):
            domain_stats[domain]["correct"] += 1
    stats["by_domain"] = {
        k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
        for k, v in domain_stats.items()
    }

    # Statistics by sub_category
    sub_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        sub = r.get("sub_category", "unknown")
        sub_stats[sub]["total"] += 1
        if r.get("correct", False):
            sub_stats[sub]["correct"] += 1
    stats["by_sub_category"] = {
        k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
        for k, v in sub_stats.items()
    }

    # Statistics by stage (initial / L3 / L2 / L1)
    stage_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        stage = r.get("stage", "unknown")
        stage_stats[stage]["total"] += 1
        if r.get("correct", False):
            stage_stats[stage]["correct"] += 1
    stats["by_stage"] = {
        k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
        for k, v in stage_stats.items()
    }

    return stats
