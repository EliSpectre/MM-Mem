# HD-EPIC Evaluation with Qwen-3 Series

- For Qwen-3 series models that support `video_url` input.
- If your model only supports `image_url` input, use the `Qwen-2-5/` folder instead.

## Step 1: Extract Frames at 1fps (224x224 Resolution)

> This step is optional if your GPU memory is sufficient to process raw videos directly.

```bash
python MM-Mem/Baseline/HD-EPIC/Dataset/convert_mp4s.py
```

## Step 2: Configure vLLM OpenAI-Compatible Server

```python
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3-VL-2B-Instruct"  # Must match --served-model-name in vLLM
```

**Important:** Some questions require up to 27 videos. Ensure vLLM supports at least 30 video inputs:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --served-model-name Qwen3-VL-2B-Instruct \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 1024, "video": 30}' \
  --gpu-memory-utilization 0.8 \
  --seed 1234 \
  --port 8002 \
  --enforce-eager \
  --disable-log-requests \
  --trust-remote-code \
  --disable-mm-preprocessor-cache \
  --allowed-local-media-path /path/to/data
```

- **Note:** Qwen3-VL series supports a context length of up to 256k tokens.

## Step 3: Set File Paths

```python
# Video directory (use absolute paths for --allowed-local-media-path compatibility)
DEFAULT_VIDEO_DIR = "/path/to/HD-EPIC/videos"

# Pre-processed JSON question files
DEFAULT_JSON_DIR = "/path/to/MM-Mem/Baseline/HD-EPIC/Dataset/test"

# Concurrency (adjust based on your GPU capacity and vLLM --max-num-seqs)
MAX_CONCURRENT_REQUESTS = 5
```
