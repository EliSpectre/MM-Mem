# HD-EPIC数据集测试的流程
- 针对qwen-3系列及其支持video_url输入的模型
- 主要是修改一下各个路径和api的相关变量
- 这个文件是输入的video_url 所以其他如果调用 API 后输入的也是 Video_url 就可以继续用这个，但是倘若只能自己抽帧，则选取Qwen-2-5文件夹即可

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
MODEL_NAME = "Qwen3-VL-2B-Instruct"
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
```

- 注意：!!!!!!!!!!!!!!!!!!!!!!!!!
- 注意:由于有些问题需要访问27个视频，学长vllm部署的时候，要保证支持 30 个视频输入: --limit-mm-per-prompt '{"image": 10, "video": 30}'     
- qwen-3-vl 系列的上下文输入长度的限制是256k   --max-model-len 262144

```python
python -m vllm.entrypoints.openai.api_server \
  --model "/data/tempuser2/MMAgent/Qwen-VL/qwen_models/Qwen3-VL-2B-Instruct" \
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
  --allowed-local-media-path "/data/tempuser2/MMAgent"
```




修改一下上边的各种配置路径就可以，其中这个
- DEFAULT_VIDEO_DIR:数据集中视频的根目录
- DEFAULT_JSON_DIR:    /data/tempuser2/MMAgent/MM-Mem/Baseline/HD-EPIC/Dataset/test   设置为这个路径即可

- 上边的  BASE_URL  API_KEY  MODEL_NAME 就是经典的VLLM部署后的变量

- 麻烦学长帮忙跑一下这个 Video-MME 数据集在 Qwen/Qwen3-VL-2B-Instruct  Qwen/Qwen3-VL-4B-Instruct  Qwen/Qwen3-VL-8B-Instruct  的效果

- 如果有算力，顺手帮忙跑一下这个 Qwen/Qwen3-VL-32B-Instruct这个实验效果。

- 说明：Qwen/Qwen3-VL-2B-Instruct 模型是没问题的，其他参数量的应该没问题，有问题学长随时找我就可以

- 学长可以看看glm多模态系列,选择一到两个测一下就好(可以最后再测)
