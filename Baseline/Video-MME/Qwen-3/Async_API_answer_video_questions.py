"""
Use a vLLM OpenAI-compatible API for Video-MME question answering.

This version uses a two-stage pipeline:
1. Answer from the full video and inspect first-token confidence over A/B/C/D.
2. If confidence is low, split the video into 10-second clips, score clip relevance
   with yes/no token probabilities, keep the top clip, merge it into a retrieval
   video, and answer again from that retrieval video.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time

import cv2
from openai import AsyncOpenAI


# --- 1. Configuration ---

BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3-VL-8B-Instruct"

VIDEO_DIR = "/data/tempuser2/MMAgent/Video-MME/Video_MME_Videos"
JSON_PATH = "/data/tempuser2/MMAgent/MM-Mem/Baseline/Video-MME/Dataset/video_mme_test.json"

MAX_CONCURRENT_REQUESTS = 15
SEMAPHORE = None

OPTION_LABELS = ("A", "B", "C", "D")
RELEVANCE_LABELS = ("yes", "no")
ANSWER_CONFIDENCE_THRESHOLD = 0.8
SEGMENT_SECONDS = 10
TOP_K_SEGMENTS = 1
TOP_LOGPROBS = 10
RETRIEVAL_CACHE_DIRNAME = "retrieval_cache"


# --- 2. Utility helpers ---

def load_questions(json_path):
    """Load the question list from JSON."""
    if not os.path.exists(json_path):
        print(f"Error: question file not found at {json_path}", file=sys.stderr)
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON file: {exc}", file=sys.stderr)
        return []


def ensure_dir(path):
    """Create the directory if needed and return the same path."""
    os.makedirs(path, exist_ok=True)
    return path


def normalize_answer(answer_str):
    """Extract a multiple-choice answer label such as A/B/C/D."""
    if not isinstance(answer_str, str):
        return ""

    for char in answer_str.strip():
        if char.isalpha():
            return char.upper()
    return ""


def normalize_binary_label(text):
    """Normalize text to yes/no when possible."""
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", text.strip()).lower()
    return cleaned if cleaned in RELEVANCE_LABELS else ""


def build_task_uid(video_id, question_id, question_text):
    """Build a stable, filesystem-safe task identifier."""
    raw = f"{video_id}::{question_id}::{question_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def build_mcq_prompt(question_text, options):
    """Build the full-video or retrieval-video multiple-choice prompt."""
    options_str = "\n".join(options)
    return (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"Question: {question_text}\n"
        f"Possible answer choices:\n{options_str}\n"
        "The best answer is:"
    )


def build_relevance_prompt(question_text, options):
    """Build the yes/no relevance scoring prompt for a clip."""
    options_str = "\n".join(options)
    return (
        "You are selecting the most relevant video clip for answering a multiple-choice question.\n"
        "Look at the clip and decide whether this clip contains information that is relevant for answering the question.\n"
        "Reply with exactly one token: yes or no.\n"
        f"Question: {question_text}\n"
        f"Possible answer choices:\n{options_str}\n"
        "Relevant clip?"
    )


def extract_first_token_logprob(response):
    """Return the first token logprob object if available."""
    try:
        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            return choice.logprobs.content[0]
    except Exception:
        return None
    return None


def aggregate_label_probabilities(first_token_logprob, labels, normalizer):
    """
    Aggregate top-logprob token variants into a normalized label distribution.

    Variants such as " A" and "A" are folded into the same label.
    """
    label_masses = {label: 0.0 for label in labels}
    if first_token_logprob is None:
        return label_masses

    candidate_items = [first_token_logprob]
    candidate_items.extend(getattr(first_token_logprob, "top_logprobs", []) or [])
    seen_items = set()

    for item in candidate_items:
        token = getattr(item, "token", None)
        logprob = getattr(item, "logprob", None)
        if token is None or logprob is None:
            continue

        item_key = (token, round(float(logprob), 8))
        if item_key in seen_items:
            continue
        seen_items.add(item_key)

        label = normalizer(token)
        if label not in label_masses:
            continue

        if logprob <= -100:
            continue

        label_masses[label] += math.exp(logprob)

    total_mass = sum(label_masses.values())
    if total_mass <= 0:
        return {label: 0.0 for label in labels}

    return {label: label_masses[label] / total_mass for label in labels}


def build_segment_manifest_path(segment_dir):
    """Return the manifest path for cached clips."""
    return os.path.join(segment_dir, "segments_manifest.json")


def init_duration_stats():
    """Create an empty accuracy accumulator keyed by duration bucket."""
    return {
        "short": {"correct": 0, "total": 0},
        "medium": {"correct": 0, "total": 0},
        "long": {"correct": 0, "total": 0},
        "unknown": {"correct": 0, "total": 0},
    }


def update_duration_stats(stats, duration, is_correct):
    """Update one accuracy accumulator."""
    bucket = duration if duration in stats else "unknown"
    stats[bucket]["total"] += 1
    if is_correct:
        stats[bucket]["correct"] += 1


def print_duration_stats(title, stats):
    """Print total accuracy and per-duration accuracy for one group."""
    total_questions = sum(item["total"] for item in stats.values())
    total_correct = sum(item["correct"] for item in stats.values())

    print(f"\n--- {title} ---")
    if total_questions == 0:
        print("  No samples.")
        return

    overall_accuracy = (total_correct / total_questions) * 100
    print(f"  Total: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")

    for duration in ("short", "medium", "long", "unknown"):
        total = stats[duration]["total"]
        if total <= 0:
            continue
        correct = stats[duration]["correct"]
        accuracy = (correct / total) * 100
        print(f"  {duration.capitalize()}: {accuracy:.2f}% ({correct}/{total})")


def load_cached_segments(segment_dir):
    """Load cached segment metadata when all files still exist."""
    manifest_path = build_segment_manifest_path(segment_dir)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return None

    if not isinstance(metadata, list) or not metadata:
        return None

    for item in metadata:
        if not os.path.isfile(item.get("segment_path", "")):
            return None

    return metadata


def save_cached_segments(segment_dir, segments):
    """Save segment metadata to disk."""
    manifest_path = build_segment_manifest_path(segment_dir)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def split_video_into_segments(video_path, video_id, cache_root):
    """Split the source video into contiguous 10-second clips."""
    segment_dir = ensure_dir(os.path.join(cache_root, video_id, "segments"))
    cached_segments = load_cached_segments(segment_dir)
    if cached_segments is not None:
        return cached_segments

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video for segmentation: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Unable to read frame size from video: {video_path}")

    segment_frames = max(1, int(round(fps * SEGMENT_SECONDS)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    segments = []
    writer = None
    segment_index = -1
    segment_start_frame = 0
    frames_in_segment = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if writer is None:
                segment_index += 1
                segment_start_frame = frame_index
                frames_in_segment = 0
                segment_path = os.path.join(segment_dir, f"segment_{segment_index:04d}.mp4")
                writer = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"Unable to open segment writer: {segment_path}")

            writer.write(frame)
            frames_in_segment += 1
            frame_index += 1

            if frames_in_segment >= segment_frames:
                writer.release()
                writer = None
                segment_end_frame = frame_index
                segment_path = os.path.join(segment_dir, f"segment_{segment_index:04d}.mp4")
                segments.append(
                    {
                        "segment_index": segment_index,
                        "start_sec": round(segment_start_frame / fps, 3),
                        "end_sec": round(segment_end_frame / fps, 3),
                        "segment_path": segment_path,
                    }
                )

        if writer is not None:
            writer.release()
            writer = None
            segment_end_frame = frame_index
            segment_path = os.path.join(segment_dir, f"segment_{segment_index:04d}.mp4")
            segments.append(
                {
                    "segment_index": segment_index,
                    "start_sec": round(segment_start_frame / fps, 3),
                    "end_sec": round(segment_end_frame / fps, 3),
                    "segment_path": segment_path,
                }
            )
    finally:
        if writer is not None:
            writer.release()
        cap.release()

    if not segments:
        raise ValueError(f"No segments were created for video: {video_path}")

    save_cached_segments(segment_dir, segments)
    return segments


def merge_segments_into_video(selected_segments, task_uid, video_id, cache_root):
    """Merge selected segments into a single retrieval video."""
    if not selected_segments:
        raise ValueError("No selected segments provided for merge.")

    merged_dir = ensure_dir(os.path.join(cache_root, video_id, "merged"))
    merged_path = os.path.join(merged_dir, f"{task_uid}_top{len(selected_segments)}.mp4")
    if os.path.isfile(merged_path):
        return merged_path

    writer = None

    try:
        for segment in sorted(selected_segments, key=lambda item: item["start_sec"]):
            cap = cv2.VideoCapture(segment["segment_path"])
            if not cap.isOpened():
                raise ValueError(f"Unable to open selected segment: {segment['segment_path']}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            if width <= 0 or height <= 0:
                cap.release()
                raise ValueError(f"Unable to read segment shape: {segment['segment_path']}")

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(merged_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    cap.release()
                    raise ValueError(f"Unable to open retrieval writer: {merged_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            cap.release()
    finally:
        if writer is not None:
            writer.release()

    if not os.path.isfile(merged_path):
        raise ValueError(f"Failed to create retrieval video: {merged_path}")

    return merged_path


# --- 3. API helpers ---

async def request_video_completion(client, video_path, prompt, max_tokens=1, top_logprobs=TOP_LOGPROBS):
    """Send a single video+text request to the vLLM chat-completions API."""
    global SEMAPHORE

    async with SEMAPHORE:
        return await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"file://{video_path}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            logprobs=True,
            top_logprobs=top_logprobs,
        )


async def answer_question_once(client, video_path, question_text, options):
    """Answer the multiple-choice question from one video input."""
    prompt = build_mcq_prompt(question_text, options)
    response = await request_video_completion(
        client,
        video_path,
        prompt,
        max_tokens=1,
        top_logprobs=TOP_LOGPROBS,
    )

    raw_answer = response.choices[0].message.content.strip()
    predicted_option = normalize_answer(raw_answer)
    first_token_logprob = extract_first_token_logprob(response)
    option_probs = aggregate_label_probabilities(first_token_logprob, OPTION_LABELS, normalize_answer)
    confidence = option_probs.get(predicted_option, 0.0) if predicted_option else 0.0
    logprobs_usable = sum(option_probs.values()) > 0

    return {
        "raw_answer": raw_answer,
        "predicted_option": predicted_option,
        "option_probs": option_probs,
        "confidence": confidence,
        "logprobs_usable": logprobs_usable,
    }


async def score_segment_relevance(client, segment_info, question_text, options):
    """Score whether a segment is relevant via yes/no token probabilities."""
    prompt = build_relevance_prompt(question_text, options)

    try:
        response = await request_video_completion(
            client,
            segment_info["segment_path"],
            prompt,
            max_tokens=1,
            top_logprobs=TOP_LOGPROBS,
        )
    except Exception as exc:
        return {
            "success": False,
            "segment_index": segment_info["segment_index"],
            "start_sec": segment_info["start_sec"],
            "end_sec": segment_info["end_sec"],
            "segment_path": segment_info["segment_path"],
            "yes_prob": 0.0,
            "raw_prediction": None,
            "error": str(exc),
        }

    raw_prediction = response.choices[0].message.content.strip()
    first_token_logprob = extract_first_token_logprob(response)
    label_probs = aggregate_label_probabilities(
        first_token_logprob,
        RELEVANCE_LABELS,
        normalize_binary_label,
    )

    return {
        "success": True,
        "segment_index": segment_info["segment_index"],
        "start_sec": segment_info["start_sec"],
        "end_sec": segment_info["end_sec"],
        "segment_path": segment_info["segment_path"],
        "yes_prob": label_probs.get("yes", 0.0),
        "raw_prediction": raw_prediction,
        "label_probs": label_probs,
    }


async def ask_question_with_two_stage_pipeline(client, task):
    """Run the full two-stage inference pipeline for one question."""
    question_text = task["question_text"]
    options = task["options"]
    task_info = task["task_info"]
    video_path = task["video_path"]
    video_state = task["video_state"]
    cache_root = task["cache_root"]

    try:
        initial_result = await answer_question_once(client, video_path, question_text, options)
    except Exception as exc:
        print(f"Request failed [video={task_info['video_id']}]: {exc}", file=sys.stderr)
        return {
            "success": False,
            "video_file": task_info["video_file"],
            "videoID": task_info["video_id"],
            "question_id": task_info["question_id"],
            "duration": task_info["duration"],
            "question": question_text,
            "options": options,
            "ground_truth": task_info["ground_truth"],
            "model_prediction": None,
            "final_prediction": None,
            "is_correct": False,
            "error": str(exc),
        }

    initial_prediction = initial_result["raw_answer"]
    initial_prediction_norm = initial_result["predicted_option"]
    initial_option_probs = initial_result["option_probs"]
    initial_confidence = initial_result["confidence"]
    ground_truth_norm = normalize_answer(task_info["ground_truth"])

    retrieval_attempted = False
    used_retrieval = False
    selected_segments = []
    final_prediction = initial_prediction
    final_prediction_norm = initial_prediction_norm
    final_stage = "full_video"
    fallback_reason = None

    should_use_initial = (
        initial_result["logprobs_usable"]
        and initial_prediction_norm in OPTION_LABELS
        and initial_confidence >= ANSWER_CONFIDENCE_THRESHOLD
    )

    if not should_use_initial:
        retrieval_attempted = True

        segments = video_state.get("segments")
        if segments is None and video_state.get("segment_error") is None:
            async with video_state["segment_lock"]:
                if (
                    video_state.get("segments") is None
                    and video_state.get("segment_error") is None
                ):
                    try:
                        video_state["segments"] = await asyncio.to_thread(
                            split_video_into_segments,
                            video_path,
                            task_info["video_id"],
                            cache_root,
                        )
                        print(
                            f"  Cached {len(video_state['segments'])} segments for "
                            f"{task_info['video_id']}"
                        )
                    except Exception as exc:
                        video_state["segment_error"] = str(exc)
                        print(
                            f"  Segment cache failed for {task_info['video_id']}: {exc}",
                            file=sys.stderr,
                        )

        segments = video_state.get("segments") or []

        if not segments:
            if video_state.get("segment_error"):
                fallback_reason = f"segment_cache_unavailable: {video_state['segment_error']}"
            else:
                fallback_reason = "segment_cache_unavailable"
        else:
            segment_scores = []
            for segment in segments:
                score = await score_segment_relevance(client, segment, question_text, options)
                if score.get("success"):
                    segment_scores.append(score)

            if not segment_scores:
                fallback_reason = "all_segment_relevance_requests_failed"
            else:
                ranked_scores = sorted(
                    segment_scores,
                    key=lambda item: (-item["yes_prob"], item["start_sec"]),
                )
                selected_segments = ranked_scores[:TOP_K_SEGMENTS]

                try:
                    retrieval_video_path = merge_segments_into_video(
                        selected_segments,
                        task_info["task_uid"],
                        task_info["video_id"],
                        cache_root,
                    )
                except Exception as exc:
                    fallback_reason = f"segment_merge_failed: {exc}"
                else:
                    try:
                        retrieval_result = await answer_question_once(
                            client,
                            retrieval_video_path,
                            question_text,
                            options,
                        )
                    except Exception as exc:
                        fallback_reason = f"retrieval_answer_failed: {exc}"
                    else:
                        retrieval_prediction_norm = retrieval_result["predicted_option"]
                        if retrieval_prediction_norm in OPTION_LABELS:
                            final_prediction = retrieval_result["raw_answer"]
                            final_prediction_norm = retrieval_prediction_norm
                            final_stage = "retrieval_video"
                            used_retrieval = True
                        else:
                            fallback_reason = "retrieval_answer_invalid"

    is_correct = bool(
        final_prediction_norm
        and ground_truth_norm
        and final_prediction_norm == ground_truth_norm
    )

    return {
        "success": True,
        "video_file": task_info["video_file"],
        "videoID": task_info["video_id"],
        "question_id": task_info["question_id"],
        "task_uid": task_info["task_uid"],
        "duration": task_info["duration"],
        "question": question_text,
        "options": options,
        "ground_truth": task_info["ground_truth"],
        "model_prediction": final_prediction,
        "initial_prediction": initial_prediction,
        "initial_prediction_norm": initial_prediction_norm,
        "initial_option_probs": initial_option_probs,
        "initial_confidence": initial_confidence,
        "retrieval_attempted": retrieval_attempted,
        "used_retrieval": used_retrieval,
        "selected_segments": selected_segments,
        "final_prediction": final_prediction,
        "final_prediction_norm": final_prediction_norm,
        "final_stage": final_stage,
        "fallback_reason": fallback_reason,
        "is_correct": is_correct,
    }


async def process_all_tasks(tasks, client):
    """Run all question tasks asynchronously and report progress."""
    async_tasks = []
    for task in tasks:
        async_tasks.append(ask_question_with_two_stage_pipeline(client, task))

    print(
        f"\nStarting {len(async_tasks)} tasks "
        f"(max concurrency: {MAX_CONCURRENT_REQUESTS})..."
    )

    results = []
    completed = 0
    total_tasks = len(async_tasks)
    progress_width = len(str(total_tasks))

    for coro in asyncio.as_completed(async_tasks):
        result = await coro
        completed += 1

        if result.get("success"):
            stage_tag = "RETR" if result.get("used_retrieval") else "FULL"
            pred_display = str(result.get("final_prediction", "None"))[:6]
            gt_display = str(result.get("ground_truth", "None"))[:6]
            conf_display = result.get("initial_confidence", 0.0)
            print(
                f"[{completed:>{progress_width}}/{total_tasks}] {stage_tag:<4} "
                f"video={result['videoID']:<20} pred={pred_display:<6} "
                f"gt={gt_display:<6} conf={conf_display:.3f} "
                f"correct={str(result['is_correct']):<5}"
            )
        else:
            print(
                f"[{completed:>{progress_width}}/{total_tasks}] FAIL "
                f"video={result['videoID']:<20}"
            )

        results.append(result)

    return results


# --- 4. Main execution ---

async def main_async():
    """Async main entry point."""
    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    start_time = time.time()

    all_questions = load_questions(JSON_PATH)
    if not all_questions:
        print(f"Error: unable to load questions from {JSON_PATH}.", file=sys.stderr)
        return

    vid_map = {}
    for q in all_questions:
        vid = q.get("videoID") or q.get("video_id") or q.get("videoId")
        if vid:
            vid_map.setdefault(vid, []).append(q)

    print(f"Connecting to vLLM service: {BASE_URL}")
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    print("Async client initialized successfully.")

    if not os.path.isdir(VIDEO_DIR):
        print(f"Error: video directory does not exist: {VIDEO_DIR}", file=sys.stderr)
        return

    mp4_files = []
    for root, _, files in os.walk(VIDEO_DIR):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, filename))

    if not mp4_files:
        print(f"No mp4 files found under {VIDEO_DIR}.", file=sys.stderr)
        return

    print(f"Found {len(mp4_files)} mp4 files.")

    out_dir = ensure_dir(os.path.join(os.getcwd(), MODEL_NAME))
    cache_root = ensure_dir(os.path.join(out_dir, RETRIEVAL_CACHE_DIRNAME))

    tasks = []
    video_states = {}

    print(
        f"\nLazy segment retrieval enabled (segment length: {SEGMENT_SECONDS}s, "
        f"confidence threshold: {ANSWER_CONFIDENCE_THRESHOLD}, top-k: {TOP_K_SEGMENTS})..."
    )

    for video_path in sorted(mp4_files):
        mp4_file = os.path.basename(video_path)
        video_id = os.path.splitext(mp4_file)[0]

        questions_for_video = vid_map.get(video_id)
        if not questions_for_video:
            continue

        if video_id not in video_states:
            video_states[video_id] = {
                "segments": None,
                "segment_error": None,
                "segment_lock": asyncio.Lock(),
            }

        video_state = video_states[video_id]

        for q_data in questions_for_video:
            duration = q_data.get("duration", "unknown").lower()
            question_text = q_data.get("question", "No question text found.")
            options = q_data.get("options", [])
            ground_truth_answer = q_data.get("answer", "")
            question_id = q_data.get("question_id")
            task_uid = build_task_uid(video_id, question_id, question_text)

            tasks.append(
                {
                    "video_path": video_path,
                    "question_text": question_text,
                    "options": options,
                    "video_state": video_state,
                    "cache_root": cache_root,
                    "task_info": {
                        "video_file": mp4_file,
                        "video_id": video_id,
                        "question_id": question_id,
                        "task_uid": task_uid,
                        "duration": duration,
                        "ground_truth": ground_truth_answer,
                    },
                }
            )

    if not tasks:
        print("No matching video/question pairs were found.", file=sys.stderr)
        return

    print(f"Built {len(tasks)} question-answer tasks.")

    all_results = await process_all_tasks(tasks, client)

    overall_stats = init_duration_stats()
    full_stats = init_duration_stats()
    retrieval_stats = init_duration_stats()

    successful_results = []
    for result in all_results:
        if result.get("success"):
            duration = result["duration"]
            update_duration_stats(overall_stats, duration, result["is_correct"])
            if result.get("used_retrieval"):
                update_duration_stats(retrieval_stats, duration, result["is_correct"])
            else:
                update_duration_stats(full_stats, duration, result["is_correct"])
            successful_results.append(result)

    elapsed_time = time.time() - start_time
    print(f"\n\n{'=' * 20}\nFinal Results\n{'=' * 20}")
    print(f"Total elapsed time: {elapsed_time:.2f} s")

    total_questions = sum(item["total"] for item in overall_stats.values())
    total_correct = sum(item["correct"] for item in overall_stats.values())

    if total_questions > 0:
        overall_accuracy = (total_correct / total_questions) * 100
        print(f"Total questions: {total_questions}")
        print(f"Correct answers: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        print(f"Average time per question: {elapsed_time / total_questions:.2f} s")
        print_duration_stats("OVERALL ACCURACY", overall_stats)
        print_duration_stats("FULL-VIDEO DIRECT ANSWER ACCURACY", full_stats)
        print_duration_stats("RETRIEVAL ANSWER ACCURACY", retrieval_stats)
    else:
        print("No successful questions were processed, cannot compute accuracy.")

    out_path = os.path.join(out_dir, "video_answers_vllm_api_parallel.json")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(successful_results, f, ensure_ascii=False, indent=2)
        print(f"\nProcessing complete. Results saved to: {out_path}")
    except Exception as exc:
        print(f"Error writing result file: {exc}", file=sys.stderr)


def main():
    """Sync entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
