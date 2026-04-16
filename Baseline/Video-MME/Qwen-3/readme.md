# Video-MME Evaluation with Qwen-3 Series

Two scripts are provided:
- `Async_API_answer_video_questions.py` — **without** subtitles
- `Async_API_answer_video_questions_w_sub.py` — **with** subtitles

## Step 1: Configure vLLM OpenAI-Compatible Server

```python
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3-VL-2B-Instruct"  # Must match --served-model-name in vLLM
```

## Step 2: Set File Paths

```python
# Video directory (use absolute paths for --allowed-local-media-path compatibility)
VIDEO_DIR = "/path/to/Video-MME/videos"

# Pre-processed JSON question file
JSON_PATH = "/path/to/filtered_questions_by_existing_videos.json"

# Concurrency (adjust based on your GPU capacity)
MAX_CONCURRENT_REQUESTS = 15

# [For subtitle version only] Subtitle directory
SUBTITLE_DIR = "/path/to/MM-Mem/Baseline/Video-MME/Dataset/subtitle"
```
