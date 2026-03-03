# VStream-QA数据集中的Ego_4D测试的流程

- 主要是修改一下各个路径和api的相关变量
- 这个VStream-QA数据集中有两部分,一部分是 "Movienet" ,一部分是 "Ego_4D"
- 这个是 "Ego_4D" 部分


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
注意,这个下载好的数据集的解压缩流程
```python 
# 使用以下命令将它们合并为一个 .zip 文件,而不是一个 .tar.gz 文件
cat ego4d_frames_online.partaa ego4d_frames_online.partab ego4d_frames_online.partac > ego4d_frames_online.zip

# 使用 unzip 解压
unzip ego4d_frames_online.zip
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

###  第一步修改 vLLM 部署的 OpenAI 兼容服务器地址

```python
# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen3-VL-2B-Instruct"

# 这个是对应的评分的API (无需修改)
# GPT 评分 API 配置（使用 Azure OpenAI）
GPT_API_KEY = ""
GPT_API_BASE = ""
GPT_ENGINE = "gpt-4o-mini"
```


### 第二步修改文件路径（配置）

```python
# 默认文件路径（可通过命令行参数覆盖）,这个视频就是原始视频的文件夹即可,记得使用绝对路径，因为 '''allowed-local-media-path''' 是对图片或者视频的路径加以限制的
# 解压缩完后就是一个文件夹,这个文件夹下是一系列的文件夹,名字就是各个视频的id,文件夹里边是采样出来的图片
DEFAULT_VIDEO_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/VStream-QA/Video"

# 查看这个 "MM-Mem/Baseline/VStream-QA/Dataset/test_qa_ego4d.json" 这个即可,这些json文件是我提前处理好的,学长用这个可以省去很多麻烦
JSON_PATH = "MM-Mem/Baseline/VStream-QA/Dataset/test_qa_ego4d.json"
# 并行化配置,学长那里如果比较有算力,可以调大一些
MAX_CONCURRENT_REQUESTS = 15  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）

```

### 第三步修改一些变量
```python
# 图片采样配置
MAX_FRAMES = 128  # 最大帧数（如果帧数超过此值则均匀采样）

# 图片分辨率配置
IMAGE_SIZE = (224, 224)  # 图片缩放目标尺寸 (width, height)，设为 None 则不缩放

# 帧范围控制
USE_START_TIME = False  # True: 从 start_time 到 end_time; False: 从第一帧到 end_time
```

