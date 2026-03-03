# HD-EPIC数据集测试的流程
- 针对qwen-2-5系列及其支持image_url输入的模型，不支持video_url输入的模型
- 主要是修改一下各个路径和api的相关变量
- 这个文件夹下的文件采用先均匀采帧，然后再把提前抽好的帧输入

### 第一步将原始视频抽帧为 1fps (24*224)的分辨率

- 这一步学长可以最后做,我当时可能是算力的影响，视频有点长，导致不能显存不是很够,如果够的话，就可以不自己手动降低分辨率啦

```python
# 运行这个文件，抽帧压缩分辨率即可
python MM-Mem/Baseline/HD-EPIC/Dataset/convert_mp4s.py
```

###  第二步修改 vLLM 部署的 OpenAI 兼容服务器地址

```python
# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
```

- 这一步跟Video-MME数据集是一样的，应该没什么问题

### 第三步修改文件路径（配置）

```python
# 默认文件路径（可通过命令行参数覆盖）,这个视频就是原始视频的文件夹即可,记得使用绝对路径，因为 '''allowed-local-media-path''' 是对图片或者视频的路径加以限制的
DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/Benchmark/VideoPre/rgb_224_1_vig"
# # 查看这个 "/data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test" 这个即可,这个文件夹下的json文件是我提前处理好的,学长用这个可以省去很多麻烦
DEFAULT_JSON_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test"
# 并行化配置,学长那里如果比较有算力,可以调大一些
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
# 视频采帧配置
NUM_FRAMES_PER_VIDEO = 128  # 总共均匀采帧数量（多个视频时平均分配）
```

- qwen2.5-vl 系列的上下文输入长度的限制是32k

```python
python -m vllm.entrypoints.openai.api_server \
  --model "/data/tempuser2/MMAgent/Qwen-VL/qwen_models/Qwen2.5-VL-7B-Instruct" \
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
  --allowed-local-media-path "/data/tempuser2/MMAgent"
```



修改一下上边的各种配置路径就可以，其中这个
- DEFAULT_VIDEO_DIR:数据集中视频的根目录
- DEFAULT_JSON_DIR:    /data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test   设置为这个路径即可

- 上边的  BASE_URL  API_KEY  MODEL_NAME 就是经典的VLLM部署后的变量

- 麻烦学长帮忙跑一下这个 Video-MME 数据集在 qwen2.5-vl-7b 的效果
- qwen-2.5-vl-7b、LLaVA-Video、LongVA、VideoLLaMA 3
- 有些模型我没有亲自试验过，应该是大差不差的,学长可以根据自己的经验选择video_url还是 image_url

- 说明：qwen-2.5-vl-7b 模型是没问题的，其他模型的应该没问题，有问题学长随时找我就可以