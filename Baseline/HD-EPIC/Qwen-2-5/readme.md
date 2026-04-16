# HD-EPIC Evaluation with Qwen-2.5 Series

- For Qwen-2.5 series models that support `image_url` input (not `video_url`).
- Frames are uniformly sampled in advance, then fed as images.

## Step 1: Extract Frames at 1fps (224x224 Resolution)

> This step is optional if your GPU memory is sufficient to process raw videos directly.

```bash
python MM-Mem/Baseline/HD-EPIC/Dataset/convert_mp4s.py
```

## Step 2: Configure vLLM OpenAI-Compatible Server

```python
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # Must match --served-model-name in vLLM
```

Example vLLM launch command:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --max-model-len 32000 \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 258}' \
  --gpu-memory-utilization 0.8 \
  --seed 1234 \
  --port 8000 \
  --enforce-eager \
  --disable-log-requests \
  --trust-remote-code \
  --disable-mm-preprocessor-cache \
  --allowed-local-media-path /path/to/data
```

## Step 3: Set File Paths

```python
# Video directory (use absolute paths for --allowed-local-media-path compatibility)
DEFAULT_VIDEO_DIR = "/path/to/HD-EPIC/videos"

# Pre-processed JSON question files
DEFAULT_JSON_DIR = "/path/to/MM-Mem/Baseline/HD-EPIC/Dataset/test"

# Concurrency (adjust based on your GPU capacity and vLLM --max-num-seqs)
MAX_CONCURRENT_REQUESTS = 5

# Frame sampling
NUM_FRAMES_PER_VIDEO = 128
```

- **Note:** Qwen2.5-VL series has a context length limit of 32k tokens.
