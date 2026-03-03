# Video-MME数据集测试的流程

- 主要是修改一下各个路径和api的相关变量
- 注意这个 "Async_API_answer_video_questions"是单纯的没有字幕的
- 注意这个 "Async_API_answer_video_questions_w_sub"是单纯的没有字幕的

###  第一步修改 vLLM 部署的 OpenAI 兼容服务器地址

```python
# vLLM 部署的 OpenAI 兼容服务器地址
BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
# 您通过 vLLM 部署的模型名称（必须匹配启动命令里的 --served-model-name）
MODEL_NAME = "Qwen3-VL-2B-Instruct"
```


### 第二步修改文件路径（配置）

```python
# 默认文件路径（可通过命令行参数覆盖）,这个视频就是原始视频的文件夹即可,记得使用绝对路径，因为 '''allowed-local-media-path''' 是对图片或者视频的路径加以限制的
VIDEO_DIR = "/data/tempuser2/MMAgent/Video-MME/Video_MME_Videos"
# 查看这个 "MM-Mem/Baseline/Video-MME/Dataset/video_mme_test.json" 这个即可,这些json文件是我提前处理好的,学长用这个可以省去很多麻烦
JSON_PATH = "/data/tempuser2/MMAgent/filtered_questions_by_existing_videos.json"
# 并行化配置,学长那里如果比较有算力,可以调大一些
MAX_CONCURRENT_REQUESTS = 15  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）


# 这个是 "Async_API_answer_video_questions_w_sub" 加上的,这个文件我也上传好了,"Baseline/Video-MME/Dataset/subtitle"学长用这个就行
# 字幕目录（可选，如果存在则使用字幕辅助问答）
SUBTITLE_DIR = "/data/tempuser2/MMAgent/MM-Mem/Baseline/Video-MME/Dataset/subtitle"
```

