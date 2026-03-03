# MLVU 数据集测试的流程

- 注意:这个测试的 MLVU 的 Dev开发集，不是测试集!!!!!!
- 这个是链接 https://www.modelscope.cn/datasets/AI-ModelScope/MLVU/files
- 主要是修改一下各个路径和api的相关变量

### 注意

#### 数据集的摆放
这个是视频的分布,每一个文件夹下就是对应问题类别需要的问题

<p align="center">
  <img src="fig/image1.png" alt="Example" width="600"/>
</p>

这个是问题"json"文件的分布,这个我已经处理好啦,放到了"MM-Mem/Baseline/MLVU/Dataset/Dev" 下

<p align="center">
  <img src="fig/image2.png" alt="Example" width="600"/>
</p>

### 一定要运行 Async_API_answer_video_questions_dev.py 文件，这个才是运行 Dev 数据集的

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
# 查看这个 "Baseline/MLVU/Dataset/Dev" 这个即可,这些json文件是我提前处理好的,学长用这个可以省去很多麻烦
# JSON 文件目录（包含按类别分好的 JSON 文件）
JSON_DIR = "/MM-Mem/Baseline/MLVU/Dataset/Dev"
# 并行化配置
MAX_CONCURRENT_REQUESTS = 15  # 最大并发请求数（根据 vLLM 的 --max-num-seqs 设置）
```

