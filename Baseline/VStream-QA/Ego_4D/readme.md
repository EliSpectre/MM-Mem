# VStream-QA Evaluation — Ego_4D Subset

The VStream-QA benchmark has two subsets: **Movienet** and **Ego_4D**. This covers the Ego_4D subset.

## Dataset Decompression

The Ego_4D frames are split into multiple archive parts. Merge and extract as follows:

```bash
# Merge split archives into a single zip
cat ego4d_frames_online.partaa ego4d_frames_online.partab ego4d_frames_online.partac > ego4d_frames_online.zip

# Extract
unzip ego4d_frames_online.zip
```

## Step 1: Configure vLLM OpenAI-Compatible Server

```python
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3-VL-2B-Instruct"  # Must match --served-model-name in vLLM

# GPT scoring API (Azure OpenAI, used for evaluation)
GPT_API_KEY = ""
GPT_API_BASE = ""
GPT_ENGINE = "gpt-4o-mini"
```

## Step 2: Set File Paths

```python
# Frame directory (after extraction, each subfolder is named by video ID and contains sampled frames)
DEFAULT_VIDEO_DIR = "/path/to/VStream-QA/Video"

# Pre-processed JSON question file
JSON_PATH = "/path/to/MM-Mem/Baseline/VStream-QA/Dataset/test_qa_ego4d.json"

# Concurrency
MAX_CONCURRENT_REQUESTS = 15
```

## Step 3: Configure Sampling Parameters

```python
MAX_FRAMES = 128        # Max frames per video (uniformly sampled if exceeded)
IMAGE_SIZE = (224, 224)  # Resize target (set to None to skip resizing)
USE_START_TIME = False   # True: sample from start_time to end_time; False: from first frame to end_time
```
