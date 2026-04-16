# Baseline Evaluation

This directory contains baseline evaluation scripts for direct video question answering using multimodal LLMs deployed via **vLLM**. These baselines serve as reference points for comparison with the MM-Mem memory-augmented approach.

## Supported Benchmarks

| Benchmark | Subsets | Description |
|---|---|---|
| **Video-MME** | with/without subtitles | Multi-domain video understanding |
| **HD-EPIC** | — | Egocentric video understanding |
| **MLVU** | Dev set | Multi-task long video understanding |
| **VStream-QA** | Movienet, Ego_4D | Streaming video QA |

## Setup

All baselines use the same deployment method:

1. **Deploy the model** via vLLM OpenAI-compatible server (see the main [README](../README.md) for details)
2. **Configure paths** in the corresponding Python script (video directory, JSON question file, server URL)
3. **Run** the async evaluation script

Each benchmark subfolder contains a `readme.md` with specific instructions for that dataset.

## Directory Structure

```
Baseline/
  HD-EPIC/
    Qwen-2-5/        # image_url input (frame-based)
    Qwen-3/           # video_url input (native video)
    LLaVA-Video-7B-Qwen2/
  Video-MME/
    Qwen-2.5/
    Qwen-3/
  MLVU/
    Qwen-3/
  VStream-QA/
    Movienet/
    Ego_4D/
```
