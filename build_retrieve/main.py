"""
Three-layer multimodal video memory system - entry point

Usage:
  python main.py build --video_path /path/to/video.mp4 --output_dir ./output
  python main.py retrieve --video_path /path/to/video.mp4 --memory_dir ./output --video_id xxx --question "..." --options "A. ..." "B. ..."
  python main.py full --video_path /path/to/video.mp4 --output_dir ./output --question "..." --options "A. ..." "B. ..."
  python main.py eval --dataset videomme --data_dir /path/to/data --video_dir /path/to/videos --duration short
"""

import os
import sys
import json
import logging
from tqdm import tqdm

from config import get_args, build_config
from models import ModelManager
from memory_build import build_all_memory
from memory_retrieve import hierarchical_retrieve_and_answer
from dataset_utils import (
    load_dataset_by_name,
    get_video_path,
    load_completed_results,
    append_result,
    compute_accuracy,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    args = get_args()
    if args.command is None:
        print("Please specify a subcommand: build / retrieve / full")
        print("Use --help to view help")
        sys.exit(1)

    config = build_config(args)
    model_manager = ModelManager(config)

    if args.command == "build":
        logger.info(f"=== Build Memory Mode ===")
        logger.info(f"Video: {config.video_path}")
        logger.info(f"Output directory: {config.output_dir}")

        result = build_all_memory(
            video_path=config.video_path,
            output_dir=config.output_dir,
            config=config,
            model_manager=model_manager,
        )
        print("\n=== Build Result ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "retrieve":
        logger.info(f"=== Retrieve and Answer Mode ===")
        logger.info(f"Video: {config.video_path}")
        logger.info(f"Memory directory: {config.memory_dir}")
        logger.info(f"Video ID: {config.video_id}")
        logger.info(f"Question: {config.question}")
        logger.info(f"Options: {config.options}")

        result = hierarchical_retrieve_and_answer(
            video_path=config.video_path,
            question=config.question,
            options=config.options,
            memory_dir=config.memory_dir,
            video_id=config.video_id,
            config=config,
            model_manager=model_manager,
        )
        print("\n=== Retrieve and Answer Result ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "full":
        logger.info(f"=== Full Mode (Build + Retrieve) ===")
        logger.info(f"Video: {config.video_path}")
        logger.info(f"Output directory: {config.output_dir}")

        # Step 1: Build memory
        build_result = build_all_memory(
            video_path=config.video_path,
            output_dir=config.output_dir,
            config=config,
            model_manager=model_manager,
        )
        print("\n=== Build Result ===")
        print(json.dumps(build_result, ensure_ascii=False, indent=2))

        # Step 2: Retrieve and answer
        video_id = build_result["video_id"]
        result = hierarchical_retrieve_and_answer(
            video_path=config.video_path,
            question=config.question,
            options=config.options,
            memory_dir=config.output_dir,
            video_id=video_id,
            config=config,
            model_manager=model_manager,
        )
        print("\n=== Retrieve and Answer Result ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "eval":
        logger.info(f"=== Batch Dataset Evaluation Mode ===")
        logger.info(f"Dataset: {config.dataset}")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Video directory: {config.video_dir}")
        logger.info(f"Duration filter: {config.duration}")
        logger.info(f"Memory directory: {config.memory_dir}")
        logger.info(f"Result file: {config.output_file}")

        # Load dataset
        samples = load_dataset_by_name(
            config.dataset, config.data_dir, duration=config.duration
        )

        # Load completed results (checkpoint resumption)
        completed_ids = load_completed_results(config.output_file)
        remaining = [s for s in samples if s["question_id"] not in completed_ids]
        logger.info(f"Total {len(samples)} samples, skipping {len(completed_ids)}, {len(remaining)} remaining")

        # Iterate and evaluate
        for i, sample in enumerate(tqdm(remaining, desc="Evaluation progress")):
            video_id = sample["videoID"]
            video_path = get_video_path(config.video_dir, video_id)
            question = sample["question"]
            options = sample["options"]
            answer = sample["answer"]

            if not os.path.exists(video_path):
                logger.warning(f"Video does not exist: {video_path}, skipping")
                continue

            try:
                result = hierarchical_retrieve_and_answer(
                    video_path=video_path,
                    question=question,
                    options=options,
                    memory_dir=config.memory_dir,
                    video_id=video_id,
                    config=config,
                    model_manager=model_manager,
                )

                # Check if correct
                correct = result["final_answer"] == answer

                # Save result
                record = {
                    "question_id": sample["question_id"],
                    "videoID": video_id,
                    "question": question,
                    "answer": answer,
                    "predicted": result["final_answer"],
                    "correct": correct,
                    "stage": result["stage"],
                    "entropy": result["entropy"],
                    "option_probs": result["option_probs"],
                    "domain": sample.get("domain", ""),
                    "sub_category": sample.get("sub_category", ""),
                    "duration": sample.get("duration", ""),
                }
                append_result(config.output_file, record)

                logger.info(
                    f"  [{i+1}/{len(remaining)}] {video_id} | "
                    f"predicted={result['final_answer']} answer={answer} "
                    f"{'correct' if correct else 'wrong'} | stage={result['stage']}"
                )

            except Exception as e:
                logger.error(f"  Processing failed {video_id}/{sample['question_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compute accuracy
        if os.path.exists(config.output_file):
            stats = compute_accuracy(config.output_file)
            print("\n=== Evaluation Results ===")
            print(json.dumps(stats, ensure_ascii=False, indent=2))

    else:
        print(f"Unknown subcommand: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
