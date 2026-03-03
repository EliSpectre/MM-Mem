"""
视频多层次记忆生成器 (Multi-Level Memory Generator)

实现L1、L2、L3三层记忆的生成:
- L1 (情景记忆): 基于PySceneDetect场景划分 + 自适应关键帧采样
- L2 (语义记忆): 基于Memory Manager Agent的智能聚合 (支持RL微调)
- L3 (摘要记忆): 预留接口

"""

import os
import os.path as osp
import json
import re
import time
import math
import asyncio
import base64
from io import BytesIO
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

# OpenAI API 客户端 (用于 vLLM API 调用)
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[警告] openai 未安装，API调用将不可用，请运行: pip install openai")

# 视频处理
import decord

# 场景检测
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# K-Means聚类
try:
    from sklearn.cluster import KMeans
    # SKLEARN_AVAILABLE = True
    SKLEARN_AVAILABLE = False
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[警告] scikit-learn未安装，K-Means采样将不可用")

# 不再加载本地模型，全部通过 vLLM API 调用
# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     AutoTokenizer,
#     AutoProcessor
# )


# ============================================================================
#                           数据结构定义
# ============================================================================

class L2Action(Enum):
    """L2 Memory Manager 可执行的动作"""
    CREATE_NEW = "create_new"    # 创建新的L2节点
    MERGE = "merge"             # 归入现有L2节点
    DISCARD = "discard"         # 丢弃当前L1节点
    UPDATE = "update"           # 更新现有L2节点并触发重构


@dataclass
class L1MemoryNode:
    """L1 情景记忆节点 - 对应一个场景"""
    video_id: str
    scene_index: int
    start_sec: float
    end_sec: float
    
    # 视觉记忆
    keyframe_indices: List[int] = field(default_factory=list)
    keyframe_times: List[float] = field(default_factory=list)
    keyframe_paths: List[str] = field(default_factory=list)
    
    # 特征向量 (用于聚类)
    features: Optional[np.ndarray] = None
    
    # 文本记忆 (caption)
    caption: Optional[str] = None
    event_fact: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为可JSON序列化的字典"""
        return {
            "video_id": self.video_id,
            "scene_index": self.scene_index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "keyframe_times": self.keyframe_times,
            "keyframe_paths": self.keyframe_paths,
            "caption": self.caption,
            "event_fact": self.event_fact
        }


@dataclass
class L1DetailInL2:
    """L2节点中存储的L1详细信息"""
    scene_index: int
    start_sec: float
    end_sec: float
    # 1fps采帧的关键帧
    keyframe_paths: List[str] = field(default_factory=list)
    keyframe_times: List[float] = field(default_factory=list)
    # L1的文本记忆 (visual_evidence)
    visual_evidence: Optional[str] = None
    event_fact: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "keyframe_paths": self.keyframe_paths,
            "keyframe_times": self.keyframe_times,
            "visual_evidence": self.visual_evidence,
            "event_fact": self.event_fact
        }


@dataclass
class L2MemoryNode:
    """L2 语义记忆节点 - 聚合多个L1节点"""
    video_id: str
    l2_index: int
    
    # 包含的L1节点索引
    l1_scene_indices: List[int] = field(default_factory=list)
    
    # 包含的L1节点详细信息 (1fps采帧 + visual_evidence)
    l1_details: List[L1DetailInL2] = field(default_factory=list)
    
    # 时间范围
    start_sec: float = 0.0
    end_sec: float = 0.0
    
    # 视觉记忆 (所有L1的1fps采帧汇总)
    representative_frames: List[str] = field(default_factory=list)
    representative_times: List[float] = field(default_factory=list)
    
    # 文本记忆
    working_memory: Optional[str] = None  # 当前L2节点的工作记忆 (图文交织生成)
    # finalized_summary 已移至全局保存，不再存储在每个L2节点中
    
    # 兼容旧字段
    aggregated_caption: Optional[str] = None
    event_summary: Optional[str] = None
    
    # L3的代表帧 (更精简)
    l3_keyframe_paths: List[str] = field(default_factory=list)
    l3_keyframe_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "l2_index": self.l2_index,
            "l1_scene_indices": self.l1_scene_indices,
            "l1_details": [d.to_dict() for d in self.l1_details],
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "representative_frames": self.representative_frames,
            "representative_times": self.representative_times,
            "working_memory": self.working_memory,
            "aggregated_caption": self.aggregated_caption,
            "event_summary": self.event_summary,
            "l3_keyframe_paths": self.l3_keyframe_paths,
            "l3_keyframe_times": self.l3_keyframe_times
        }


# ============================================================================
#                       L3 知识图谱数据结构
# ============================================================================

class KGNodeType(Enum):
    """知识图谱节点类型"""
    EVENT = "event"           # 事件节点 (来自L2)
    PERSON = "person"         # 人物实体
    OBJECT = "object"         # 物体实体
    PLACE = "place"           # 地点实体
    TEXT = "text"             # 文字实体 (OCR)


class KGEdgeType(Enum):
    """知识图谱边类型"""
    PARTICIPATES_IN = "PARTICIPATES_IN"  # 实体参与事件 (person/object -> event)
    LOCATED_IN = "LOCATED_IN"            # 事件发生在地点 (event -> place)
    BEFORE = "BEFORE"                    # 时序关系 (event -> event)
    CAUSES = "CAUSES"                    # 因果关系 (event -> event)
    USES = "USES"                        # 使用关系 (person -> object)
    CONTAINS = "CONTAINS"                # 包含关系 (place -> object)


@dataclass
class KGNode:
    """知识图谱节点"""
    node_id: str                         # 唯一ID: {type}_{name}_{video_id} 或 event_{l2_index}_{video_id}
    node_type: KGNodeType                # 节点类型
    name: str                            # 节点名称
    video_id: str                        # 所属视频
    
    # 时间信息
    first_seen_sec: Optional[float] = None
    last_seen_sec: Optional[float] = None
    
    # 事件节点专有字段
    l2_index: Optional[int] = None
    event_summary: Optional[str] = None
    working_memory: Optional[str] = None
    
    # 额外属性
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "video_id": self.video_id,
            "first_seen_sec": self.first_seen_sec,
            "last_seen_sec": self.last_seen_sec,
            "l2_index": self.l2_index,
            "event_summary": self.event_summary,
            "working_memory": self.working_memory,
            "attributes": self.attributes
        }


@dataclass
class KGEdge:
    """知识图谱边"""
    edge_id: str                         # 唯一ID
    source_id: str                       # 源节点ID
    target_id: str                       # 目标节点ID
    edge_type: KGEdgeType                # 边类型
    video_id: str                        # 所属视频
    
    # 时间信息 (用于时序边)
    timestamp_sec: Optional[float] = None
    
    # 额外属性
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "video_id": self.video_id,
            "timestamp_sec": self.timestamp_sec,
            "attributes": self.attributes
        }


@dataclass
class GlobalFinalizedSummary:
    """全局因果总结 (不存储到L2节点，单独保存)"""
    summary_id: int                      # 总结ID (滑动窗口批次号)
    video_id: str                        # 所属视频
    
    # 涉及的L2节点范围
    l2_start_index: int                  # 起始L2索引
    l2_end_index: int                    # 结束L2索引
    
    # 时间范围
    start_sec: float
    end_sec: float
    
    # 因果分析内容
    causal_chain: str                    # 因果链描述
    key_entities: List[str] = field(default_factory=list)  # 关键实体列表
    overall_summary: str = ""            # 整体总结
    
    def to_dict(self) -> dict:
        return {
            "summary_id": self.summary_id,
            "video_id": self.video_id,
            "l2_start_index": self.l2_start_index,
            "l2_end_index": self.l2_end_index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "causal_chain": self.causal_chain,
            "key_entities": self.key_entities,
            "overall_summary": self.overall_summary
        }


@dataclass 
class MemoryManagerState:
    """Memory Manager的状态空间"""
    # 当前活跃的L2节点
    active_l2_nodes: List[L2MemoryNode] = field(default_factory=list)
    
    # 缓冲池中的L1节点
    l1_buffer: List[L1MemoryNode] = field(default_factory=list)
    
    # 当前处理的L1节点
    current_l1: Optional[L1MemoryNode] = None
    
    # 全局上下文
    global_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_state_representation(self) -> Dict:
        """获取状态表示 (用于RL训练)"""
        return {
            "n_active_l2": len(self.active_l2_nodes),
            "n_buffered_l1": len(self.l1_buffer),
            "current_l1_index": self.current_l1.scene_index if self.current_l1 else -1,
            "total_duration": sum(
                l2.end_sec - l2.start_sec for l2 in self.active_l2_nodes
            ) if self.active_l2_nodes else 0.0
        }


# ============================================================================
#                           Memory Manager Agent
# ============================================================================

class MemoryManagerAgent:
    """
    记忆管理器Agent (Actor)
    
    负责决策L1节点如何聚合到L2节点
    - State: 当前活跃L2节点 + 前一个L2节点 + 新L1节点
    - Action: CREATE_NEW / MERGE / DISCARD / UPDATE (不需要索引，都针对最近的L2)
    - Reward: 后续定义 (RL微调时使用)
    
    模型设计 (全部通过 vLLM API 调用，不加载本地模型):
    - base_api_client: 基础 vLLM API (用于 working_memory、L3等)
    - l2_api_client: 微调后的 vLLM API (用于L1→L2聚合决策)
    """
    
    def __init__(
        self,
        config: dict,
        output_dir: str,
        base_api_client=None,         # 基础 API 客户端 (用于 working_memory、L3等)
        base_api_model_name: str = "",  # 基础 API 模型名称
        l2_api_client=None,           # L2聚合决策的 API 客户端
        l2_api_model_name: str = "",  # L2聚合决策的 API 模型名称
        max_frames_per_l2: int = 8,   # 每个L2最多读取的代表帧数
        image_size: int = 224,        # 图片缩放大小
        similarity_threshold: float = 0.7,  # 相似度阈值
    ):
        self.config = config
        self.output_dir = output_dir
        self.max_frames_per_l2 = max_frames_per_l2
        self.image_size = image_size
        self.similarity_threshold = similarity_threshold
        
        # 基础 API 客户端（用于 working_memory、L3等）
        self.base_api_client = base_api_client
        self.base_api_model_name = base_api_model_name
        
        # L2聚合决策的 API 客户端（微调后的模型）
        self.l2_api_client = l2_api_client
        self.l2_api_model_name = l2_api_model_name
        
        # 加载提示词
        self.system_prompt = self._load_prompt(
            config.get("prompts", {}).get("L2_agent_sys_prompt", "")
        )
        self.user_prompt_template = self._load_prompt(
            config.get("prompts", {}).get("L2_agent_user_prompt", "")
        )
        # 加载working memory和finalized summary的提示词
        self.working_memory_prompt = self._load_prompt(
            config.get("prompts", {}).get("L2_working_memory_prompt", "")
        )
        self.finalized_summary_prompt = self._load_prompt(
            config.get("prompts", {}).get("L2_finalized_summary_prompt", "")
        )
        
        # 状态
        self.state = MemoryManagerState()
        
    def _load_prompt(self, prompt_path: str) -> str:
        """加载提示词文件"""
        if prompt_path and osp.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return prompt_path if prompt_path else ""
    
    def _load_image_resized(self, path: str) -> Optional[Image.Image]:
        """加载并缩放图片到指定大小"""
        try:
            full_path = osp.join(self.output_dir, path) if not osp.isabs(path) else path
            if osp.exists(full_path):
                img = Image.open(full_path).convert("RGB")
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                return img
        except Exception as e:
            print(f"[图片加载错误] {path}: {e}")
        return None
    
    def reset(self):
        """重置Agent状态"""
        self.state = MemoryManagerState()
        
    def decide_action(
        self,
        current_l1: L1MemoryNode,
        l1_frames_1fps: List[str],  # 1fps采帧的路径列表
        l1_times_1fps: List[float],  # 对应的时间戳
        use_vlm: bool = True
    ) -> L2Action:
        """
        决定对当前L1节点执行的动作
        
        Args:
            current_l1: 当前需要处理的L1节点
            l1_frames_1fps: 当前L1的1fps采帧路径
            l1_times_1fps: 对应的时间戳
            use_vlm: 是否使用VLM进行推理决策
            
        Returns:
            action: 动作 (不再需要索引，MERGE/UPDATE都针对最近的L2)
        """
        self.state.current_l1 = current_l1
        
        # 如果没有现有L2节点，直接创建新的
        if not self.state.active_l2_nodes:
            return L2Action.CREATE_NEW
        
        if use_vlm:
            return self._vlm_decision(current_l1, l1_frames_1fps, l1_times_1fps)
        else:
            return self._rule_based_decision(current_l1)
    
    def _rule_based_decision(
        self,
        current_l1: L1MemoryNode
    ) -> L2Action:
        """
        基于规则的决策 (Training-Free Baseline)
        
        规则:
        1. 如果当前L1与最近的L2在时间上连续 (< 5秒间隔)，执行MERGE
        2. 否则CREATE_NEW
        """
        if not self.state.active_l2_nodes:
            return L2Action.CREATE_NEW
        
        # 找到最近的L2节点
        last_l2 = self.state.active_l2_nodes[-1]
        time_gap = current_l1.start_sec - last_l2.end_sec
        
        # 时间连续性检查 (间隔小于5秒视为连续)
        if 0 <= time_gap < 5.0:
            return L2Action.MERGE
        
        # 时间不连续，创建新L2
        return L2Action.CREATE_NEW
    
    def _vlm_decision(
        self,
        current_l1: L1MemoryNode,
        l1_frames_1fps: List[str],
        l1_times_1fps: List[float]
    ) -> L2Action:
        """
        使用VLM进行智能决策
        
        输入构建:
        1. 前一个L2节点: 只输入文本记忆 (working_memory)
        2. 当前活跃L2节点: 224x224图片 + 已有文本记忆
        3. 当前L1节点: 1fps采帧 (k-means或均匀采样后)
        """
        try:
            all_images = []
            prompt_parts = []
            
            # 1. 前一个L2节点 (只有文本)
            prev_l2_text = ""
            if len(self.state.active_l2_nodes) >= 2:
                prev_l2 = self.state.active_l2_nodes[-2]
                prev_l2_text = prev_l2.working_memory or prev_l2.event_summary or ""
                if prev_l2_text:
                    prompt_parts.append(f"## PREVIOUS L2 NODE (Text Only):\n{prev_l2_text}")
            
            # 2. 当前活跃L2节点 (图片 + 文本)
            current_l2 = self.state.active_l2_nodes[-1]
            current_l2_images = []
            for frame_path in current_l2.representative_frames[:self.max_frames_per_l2]:
                img = self._load_image_resized(frame_path)
                if img:
                    current_l2_images.append(img)
            
            current_l2_text = current_l2.working_memory or current_l2.event_summary or ""
            n_l2_images = len(current_l2_images)
            all_images.extend(current_l2_images)
            
            if n_l2_images > 0 or current_l2_text:
                prompt_parts.append(f"## CURRENT ACTIVE L2 NODE:\n- Images: {n_l2_images} frames (index 0-{n_l2_images-1})\n- Text: {current_l2_text if current_l2_text else 'None'}")
            
            # 3. 当前L1节点 (1fps采帧)
            # 对1fps帧进行采样 (k-means或均匀)
            l1_images = []
            sampled_frames = self._sample_frames_for_decision(l1_frames_1fps, max_frames=self.max_frames_per_l2)
            for frame_path in sampled_frames:
                img = self._load_image_resized(frame_path)
                if img:
                    l1_images.append(img)
            
            n_l1_images = len(l1_images)
            all_images.extend(l1_images)
            
            l1_text = current_l1.event_fact or current_l1.caption or "No description"
            prompt_parts.append(f"## CURRENT L1 NODE TO PROCESS:\n- Images: {n_l1_images} frames (index {n_l2_images}-{n_l2_images + n_l1_images - 1})\n- Event: {l1_text}")
            
            # 4. 构建完整prompt
            prompt = self._build_decision_prompt_v2(
                prev_l2_text=prev_l2_text,
                n_l2_images=n_l2_images,
                current_l2_text=current_l2_text,
                n_l1_images=n_l1_images,
                l1_text=l1_text
            )
            
            # 5. VLM推理
            if all_images:
                response = self._vlm_inference(all_images, prompt)
                action = self._parse_decision_response_v2(response)
                return action
            else:
                return self._rule_based_decision(current_l1)
                
        except Exception as e:
            print(f"[VLM决策错误] {e}, 回退到规则决策")
            return self._rule_based_decision(current_l1)
    
    def _sample_frames_for_decision(
        self,
        frame_paths: List[str],
        max_frames: int = 8
    ) -> List[str]:
        """对帧列表进行采样 (均匀采样)"""
        if len(frame_paths) <= max_frames:
            return frame_paths
        
        indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
        return [frame_paths[i] for i in indices]
    
    def _build_decision_prompt_v2(
        self,
        prev_l2_text: str,
        n_l2_images: int,
        current_l2_text: str,
        n_l1_images: int,
        l1_text: str
    ) -> str:
        """构建决策提示词 (v2: 简化版，不需要索引)"""
        
        prompt = f"""## TASK: Memory Aggregation Decision

You are a Memory Manager Agent. Decide how to handle a new L1 (scene-level) memory node.

## CONTEXT:
"""
        if prev_l2_text:
            prompt += f"""
### Previous L2 Node (Text Only):
{prev_l2_text}
"""
        
        if n_l2_images > 0:
            prompt += f"""
### Current Active L2 Node:
- Visual: The first {n_l2_images} images are from current L2 node
- Text Memory: {current_l2_text if current_l2_text else 'None yet'}
"""
        
        prompt += f"""
### Current L1 Node to Process:
- Visual: The last {n_l1_images} images are from the new L1 node
- Event Description: {l1_text}

## AVAILABLE ACTIONS:
1. **CREATE_NEW** - Create a new L2 node (when L1 represents a distinct/new event)
2. **MERGE** - Merge L1 into the current active L2 node (when it's a continuation of the same event)
3. **DISCARD** - Discard this L1 node (when it's noise, transition, or redundant)
4. **UPDATE** - Merge L1 and rewrite/restructure the L2 memory (when significant new info requires reorganization)

## DECISION CRITERIA:
- Temporal continuity: Events close in time tend to belong together
- Semantic similarity: Similar actions/objects/scenes suggest same event
- Event boundary: Clear state changes or new activities indicate new event
- Quality: Blurry, transition, or uninformative frames should be discarded

## YOUR RESPONSE:
Output ONLY one of: CREATE_NEW, MERGE, DISCARD, UPDATE
"""
        return prompt
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 字符串"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _vlm_inference(self, images: List[Image.Image], prompt: str) -> str:
        """
        执行VLM推理 - 用于L1→L2聚合决策
        
        优先使用 l2_api_client (vLLM API，微调后的模型)
        如果 API 未配置，则回退到本地模型
        """
        # 优先使用 L2 聚合 API (微调后的模型)
        if self.l2_api_client and self.l2_api_model_name:
            return self._vlm_inference_via_api(images, prompt)
        else:
            return self._vlm_inference_local(images, prompt)
    
    def _vlm_inference_via_api(self, images: List[Image.Image], prompt: str) -> str:
        """
        通过 vLLM API 执行VLM推理 - 用于L1→L2聚合决策 (微调后的模型)
        使用同步 OpenAI 客户端，避免 asyncio.run() 导致的 Event loop is closed 错误
        """
        try:
            # 构建消息内容 (图片使用 base64)
            content_parts = []
            for img in images:
                img_base64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
            content_parts.append({"type": "text", "text": prompt})
            
            messages = [
                {"role": "system", "content": self.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": content_parts}
            ]
            
            # 同步调用 API
            response = self.l2_api_client.chat.completions.create(
                model=self.l2_api_model_name,
                messages=messages,
                max_tokens=50,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[VLM推理错误 - L2 API(sync)] {e}")
            # 回退到 base_api
            return self._vlm_inference_via_base_api(images, prompt)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图片转换为base64字符串"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _vlm_inference_via_base_api(
        self, 
        images: List[Image.Image], 
        prompt: str, 
        max_tokens: int = 50,
        system_prompt: str = None
    ) -> str:
        """执行VLM推理 - 使用 base_api (vLLM API)"""
        if not self.base_api_client:
            print("[错误] base_api_client 未初始化")
            return "CREATE_NEW"
        
        try:
            # 构建消息内容
            content_parts = []
            for img in images:
                img_base64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
            content_parts.append({"type": "text", "text": prompt})
            
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": content_parts}
            ]
            
            # 同步调用 base_api
            response = self.base_api_client.chat.completions.create(
                model=self.base_api_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[VLM推理错误 - base_api] {e}")
            return "CREATE_NEW"  # 默认创建新节点
    
    def _parse_decision_response_v2(self, response: str) -> L2Action:
        """解析VLM的决策响应 (v2: 不需要索引)"""
        response = response.strip().upper()
        
        if "CREATE_NEW" in response or "CREATE" in response:
            return L2Action.CREATE_NEW
        elif "DISCARD" in response:
            return L2Action.DISCARD
        elif "UPDATE" in response:
            return L2Action.UPDATE
        elif "MERGE" in response:
            return L2Action.MERGE
        
        # 无法解析，默认创建新节点
        return L2Action.CREATE_NEW
    
    def execute_action(
        self,
        action: L2Action,
        current_l1: L1MemoryNode,
        l1_frames_1fps: List[str],
        l1_times_1fps: List[float],
        visual_evidence: str = ""
    ) -> Tuple[Optional[L2MemoryNode], bool]:
        """
        执行决策的动作
        
        Args:
            action: 决策动作
            current_l1: 当前L1节点
            l1_frames_1fps: 1fps采帧路径
            l1_times_1fps: 对应时间戳
            visual_evidence: L1的visual_evidence文本
        
        Returns:
            (affected_l2_node, need_generate_summary): 受影响的L2节点, 是否需要生成finalized_summary
        """
        # 创建L1详细信息
        l1_detail = L1DetailInL2(
            scene_index=current_l1.scene_index,
            start_sec=current_l1.start_sec,
            end_sec=current_l1.end_sec,
            keyframe_paths=l1_frames_1fps,
            keyframe_times=l1_times_1fps,
            visual_evidence=visual_evidence,
            event_fact=current_l1.event_fact
        )
        
        if action == L2Action.CREATE_NEW:
            # 先为上一个L2生成finalized_summary (如果存在)
            need_summary = len(self.state.active_l2_nodes) > 0
            
            # 创建新L2节点
            new_l2 = L2MemoryNode(
                video_id=current_l1.video_id,
                l2_index=len(self.state.active_l2_nodes),
                l1_scene_indices=[current_l1.scene_index],
                l1_details=[l1_detail],
                start_sec=current_l1.start_sec,
                end_sec=current_l1.end_sec,
                representative_frames=l1_frames_1fps.copy(),
                representative_times=l1_times_1fps.copy(),
                aggregated_caption=current_l1.caption,
                event_summary=current_l1.event_fact
            )
            self.state.active_l2_nodes.append(new_l2)
            return new_l2, need_summary
        
        elif action == L2Action.MERGE:
            # 合并到当前活跃L2节点
            target_l2 = self.state.active_l2_nodes[-1]
            target_l2.l1_scene_indices.append(current_l1.scene_index)
            target_l2.l1_details.append(l1_detail)
            target_l2.end_sec = max(target_l2.end_sec, current_l1.end_sec)
            
            # 添加1fps帧到代表帧
            target_l2.representative_frames.extend(l1_frames_1fps)
            target_l2.representative_times.extend(l1_times_1fps)
            
            return target_l2, False
        
        elif action == L2Action.UPDATE:
            # 合并并标记需要重写
            target_l2 = self.state.active_l2_nodes[-1]
            target_l2.l1_scene_indices.append(current_l1.scene_index)
            target_l2.l1_details.append(l1_detail)
            target_l2.end_sec = max(target_l2.end_sec, current_l1.end_sec)
            
            # 添加帧
            target_l2.representative_frames.extend(l1_frames_1fps)
            target_l2.representative_times.extend(l1_times_1fps)
            
            # UPDATE需要重写working_memory
            return target_l2, False  # 返回时标记需要重写
        
        elif action == L2Action.DISCARD:
            # 丢弃，不做任何操作
            return None, False
        
        return None, False
    
    def generate_working_memory(self, l2_node: L2MemoryNode) -> str:
        """
        生成L2节点的working_memory
        
        输入: L2节点下所有L1的视觉记忆和文本记忆 (图文交织)
        输出: 整理后的当前L2事件描述
        """
        try:
            all_images = []
            interleaved_content = []
            
            # 按L1节点顺序，图文交织
            for l1_detail in l2_node.l1_details:
                # 添加该L1的图片
                l1_images = []
                for path in l1_detail.keyframe_paths[:4]:  # 每个L1最多4张
                    img = self._load_image_resized(path)
                    if img:
                        l1_images.append(img)
                        all_images.append(img)
                
                # 构建该L1的描述
                l1_desc = f"[Scene {l1_detail.scene_index}] ({l1_detail.start_sec:.1f}s - {l1_detail.end_sec:.1f}s)"
                if l1_detail.event_fact:
                    l1_desc += f"\nEvent: {l1_detail.event_fact}"
                if l1_detail.visual_evidence:
                    l1_desc += f"\nVisual Evidence: {l1_detail.visual_evidence[:200]}"
                
                interleaved_content.append({
                    "n_images": len(l1_images),
                    "text": l1_desc
                })
            
            if not all_images:
                # 没有图片，直接拼接文本
                texts = [l1.event_fact or l1.visual_evidence or "" for l1 in l2_node.l1_details]
                return " -> ".join([t for t in texts if t])
            
            # 构建prompt
            prompt = self._build_working_memory_prompt(interleaved_content, l2_node)
            
            # VLM生成
            response = self._vlm_inference_long(all_images, prompt, max_tokens=512)
            l2_node.working_memory = response
            return response
            
        except Exception as e:
            print(f"[Working Memory生成错误] {e}")
            # 回退: 简单拼接
            texts = [l1.event_fact or "" for l1 in l2_node.l1_details]
            return " -> ".join([t for t in texts if t])
    
    def _build_working_memory_prompt(
        self,
        interleaved_content: List[dict],
        l2_node: L2MemoryNode
    ) -> str:
        """构建working memory的prompt"""
        
        prompt = """## TASK: Generate Working Memory for L2 Event Node

You are summarizing a sequence of scene observations into a coherent event description.

## INPUT STRUCTURE:
The images and text are interleaved by scene order:
"""
        img_idx = 0
        for item in interleaved_content:
            n_imgs = item["n_images"]
            prompt += f"\n- Images {img_idx} to {img_idx + n_imgs - 1}: {item['text']}"
            img_idx += n_imgs
        
        prompt += f"""

## TIME RANGE:
{l2_node.start_sec:.1f}s - {l2_node.end_sec:.1f}s

## INSTRUCTIONS:
1. Analyze all scenes in temporal order
2. Identify the main event/activity happening
3. Note key objects, people, actions, and state changes
4. Summarize into a coherent paragraph describing what happened

## OUTPUT:
Provide a concise but comprehensive description of the event (2-4 sentences).
Focus on: WHO did WHAT, WHERE, with WHAT objects, and any notable state changes.
"""
        return prompt
    
    def generate_finalized_summary(
        self,
        l2_node: L2MemoryNode,
        n_prev_nodes: int = 3
    ) -> str:
        """
        生成finalized_summary (已废弃，保留用于向后兼容)
        
        注意: finalized_summary 已移至 L3 阶段全局生成
        此方法仅返回working_memory，不再存储到L2节点
        """
        return l2_node.working_memory or l2_node.event_summary or ""
    
    def _build_finalized_summary_prompt(self, memories: List[dict]) -> str:
        """
        构建finalized summary的prompt (已废弃)
        
        注意: 此方法已废弃，finalized_summary 现在在 L3 阶段全局生成
        保留此方法仅用于向后兼容
        """
        return ""
    
    def _vlm_inference_long(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int = 512
    ) -> str:
        """执行VLM推理 (长输出版本) - 使用 base_api"""
        return self._vlm_inference_via_base_api(
            images=images,
            prompt=prompt,
            max_tokens=max_tokens,
            system_prompt="You are a precise visual event summarizer."
        )
    
    def _text_inference(self, prompt: str, max_tokens: int = 256) -> str:
        """纯文本推理 (不含图片) - 使用 base_api"""
        if not self.base_api_client:
            print("[错误] base_api_client 未初始化")
            return ""
        
        try:
            messages = [
                {"role": "system", "content": "You are a precise event summarizer for knowledge graph construction."},
                {"role": "user", "content": prompt}
            ]
            
            # 同步调用 base_api
            response = self.base_api_client.chat.completions.create(
                model=self.base_api_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[文本推理错误 - base_api] {e}")
            return ""
    
    def compute_reward(self, action: L2Action, result: Optional[L2MemoryNode]) -> float:
        """
        计算奖励 (用于RL训练)
        """
        # TODO: 实现奖励函数
        return 0.0


# ============================================================================
#                           Memory Generator
# ============================================================================

class VideoMemoryGenerator:
    """
    视频记忆生成器
    
    整合L1、L2、L3三层记忆的生成流程
    
    模型设计 (全部通过 vLLM API 调用，不加载本地模型):
    - base_api_client: 基础 vLLM API (用于 L1 caption、working_memory、L3等)
    - l2_api_client: 微调后的 vLLM API (用于L1→L2聚合决策)
    """
    
    def __init__(
        self,
        config: dict,
        output_dir: str,
        down_sample: int = 2
    ):
        self.config = config
        self.output_dir = output_dir
        self.down_sample = down_sample
        self.fps = None  # 将在 generate_l1_memories 中从视频动态读取
        
        # 配置参数
        self.pyscenesdetect_threshold = float(config.get("pyscenesdetect_threshold", 20.0))
        self.adaptive_k = float(config.get("adaptive_k", 2.0))
        self.frames_per_l2 = int(config.get("frames_per_window", 10))
        self.frames_per_l3 = int(config.get("frames_per_L3", 3))
        
        # 加载提示词
        self.l1_sys_prompt = self._load_prompt(
            config.get("prompts", {}).get("L1_memory_sys_prompt", "")
        )
        self.l1_user_prompt_template = self._load_prompt(
            config.get("prompts", {}).get("L1_memory_user_prompt", "")
        )
        
        # ========== Base API 客户端配置 (用于 L1 caption、working_memory、L3等) ==========
        base_api_cfg = config.get("base_api", {})
        base_api_client = None
        base_api_model_name = ""
        
        if base_api_cfg.get("enabled", False):
            if not OPENAI_AVAILABLE:
                raise RuntimeError("[错误] openai 库未安装，请运行: pip install openai")
            else:
                base_api_client = OpenAI(
                    base_url=base_api_cfg.get("base_url", "http://localhost:8002/v1"),
                    api_key=base_api_cfg.get("api_key", "EMPTY"),
                )
                base_api_model_name = base_api_cfg.get("model_name", "Qwen3-VL-2B-Instruct")
                print(f"[Base API 客户端已初始化] base_url={base_api_cfg.get('base_url')}, model={base_api_model_name}")
        else:
            raise RuntimeError("[错误] base_api 未启用，请在配置文件中启用 base_api")
        
        self.base_api_client = base_api_client
        self.base_api_model_name = base_api_model_name
        
        # 异步 API 客户端 (用于并行 L1 caption 生成)
        self.async_api_client = AsyncOpenAI(
            base_url=base_api_cfg.get("base_url", "http://localhost:8002/v1"),
            api_key=base_api_cfg.get("api_key", "EMPTY"),
        )
        self.api_max_concurrent = int(base_api_cfg.get("max_concurrent", 4))
        print(f"[异步 API 客户端已初始化] 最大并发数: {self.api_max_concurrent}")
        
        # ========== L2 聚合 API 客户端配置 (用于 L1→L2 聚合决策，微调后的模型) ==========
        # 注意：使用同步 OpenAI 客户端，避免 asyncio.run() 导致的 Event loop is closed 错误
        l2_api_cfg = config.get("l2_aggregation_api", {})
        l2_api_client = None
        l2_api_model_name = ""
        
        if l2_api_cfg.get("enabled", False):
            if not OPENAI_AVAILABLE:
                print("[警告] openai 库未安装，L2聚合将使用 base_api")
                l2_api_client = base_api_client
                l2_api_model_name = base_api_model_name
            else:
                # 使用同步 OpenAI 客户端（不要 AsyncOpenAI）
                l2_api_client = OpenAI(
                    base_url=l2_api_cfg.get("base_url", "http://localhost:8003/v1"),
                    api_key=l2_api_cfg.get("api_key", "EMPTY"),
                )
                l2_api_model_name = l2_api_cfg.get("model_name", "Qwen3-VL-2B-L2-Finetuned")
                print(f"[L2聚合 API 客户端已初始化(sync)] base_url={l2_api_cfg.get('base_url')}, model={l2_api_model_name}")
        else:
            print("[L2聚合] 未启用 L2 API 模式，将使用 base_api 进行聚合决策")
            l2_api_client = base_api_client
            l2_api_model_name = base_api_model_name
        
        # 初始化Memory Manager Agent
        self.memory_agent = MemoryManagerAgent(
            config=config,
            output_dir=output_dir,
            base_api_client=base_api_client,
            base_api_model_name=base_api_model_name,
            l2_api_client=l2_api_client,
            l2_api_model_name=l2_api_model_name
        )
        
    def _load_prompt(self, prompt_path: str) -> str:
        """加载提示词文件"""
        if prompt_path and osp.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return prompt_path if prompt_path else ""
    
    # ========================================================================
    #                           L1 生成 (并行版本)
    # ========================================================================
    
    def generate_l1_memories(self, video_path: str, skip_if_exists: bool = True) -> List[L1MemoryNode]:
        """
        并行生成L1情景记忆 (通过 vLLM API)
        
        流程:
        1. PySceneDetect场景划分 (串行)
        2. 并行处理每个场景: 自适应关键帧采样 + API 生成 caption
        3. 写入临时文件，最后按 scene_index 排序后写入最终文件
        
        Args:
            video_path: 视频文件路径
            skip_if_exists: 如果L1文件已存在是否跳过
            
        Returns:
            L1节点列表 (如果跳过则返回空列表，需要时可从文件重新加载)
        """
        # 调用异步版本
        return asyncio.run(self._generate_l1_memories_async(video_path, skip_if_exists))
    
    async def _generate_l1_memories_async(self, video_path: str, skip_if_exists: bool = True) -> List[L1MemoryNode]:
        """
        异步并行生成 L1 情景记忆
        
        采用"边生成边写入临时文件"策略，减少内存压力：
        1. 每处理完一个场景，立即将结果追加写入临时文件
        2. 所有场景处理完成后，读取临时文件，按 scene_index 排序
        3. 写入最终文件，删除临时文件
        """
        vid_id = osp.splitext(osp.basename(video_path))[0]
        print(f"\n[L1生成-并行] 处理视频: {vid_id}")
        
        # 输出文件路径
        video_output_dir = osp.join(self.output_dir, vid_id)
        l1_output_file = osp.join(video_output_dir, "episodic_memories_L1.json")
        l1_temp_file = osp.join(video_output_dir, "episodic_memories_L1_temp.json")  # 临时文件
        
        # 检查是否已经处理过该视频
        if skip_if_exists and osp.exists(l1_output_file):
            print(f"  [跳过] 视频 {vid_id} L1记忆已存在，跳过处理")
            print(f"    L1: {l1_output_file}")
            return []
        
        # 从视频动态读取 fps
        video = open_video(video_path)
        fps = video._frame_rate
        self.fps = fps
        print(f"  视频 FPS: {fps}")
        
        # 创建输出目录
        l1_keyframe_dir = osp.join(video_output_dir, "L1_keyframes")
        os.makedirs(l1_keyframe_dir, exist_ok=True)
        os.makedirs(osp.dirname(l1_output_file), exist_ok=True)
        
        # Phase 1: 场景检测 (串行)
        print("  Phase 1: 运行 PySceneDetect...")
        scene_list = self._run_scene_detection(video_path)
        
        if not scene_list:
            print(f"  [警告] 未检测到场景")
            return []
        
        print(f"  检测到 {len(scene_list)} 个原始场景")
        
        # Phase 1.5: 切割超过10秒的场景
        max_scene_duration = 10.0  # 最大场景持续时间（秒）
        split_scene_list = self._split_long_scenes(scene_list, fps, max_scene_duration)
        print(f"  切割后共 {len(split_scene_list)} 个场景片段")
        
        # Phase 2: 加载视频
        print("  Phase 2: 并行处理场景...")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(self.api_max_concurrent)
        
        # 创建文件写入锁（防止并发写入冲突）
        file_lock = asyncio.Lock()
        
        # 清空临时文件（如果存在）
        if osp.exists(l1_temp_file):
            os.remove(l1_temp_file)
        
        # 创建所有场景的异步任务
        tasks = []
        for scene_idx, (start_f_num, end_f_num) in enumerate(split_scene_list):
            task = self._process_single_scene_and_save_async(
                scene_idx=scene_idx,
                start_f_num=start_f_num,
                end_f_num=end_f_num,
                vr=vr,
                vid_id=vid_id,
                semaphore=semaphore,
                temp_file=l1_temp_file,
                file_lock=file_lock
            )
            tasks.append(task)
        
        # 并行执行所有任务
        print(f"  开始并行处理 {len(tasks)} 个场景（最大并发数: {self.api_max_concurrent}）...")
        start_time = time.time()
        
        # 收集结果
        completed = 0
        success_count = 0
        total = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            result = await coro  # result 是 (scene_index, success, event_preview) 元组
            completed += 1
            if result[1]:  # success
                success_count += 1
                print(f"  [{completed}/{total}] 场景 {result[0]}: {result[2]}")
            else:
                print(f"  [{completed}/{total}] 场景 {result[0]} 处理失败或跳过")
        
        elapsed = time.time() - start_time
        print(f"  并行处理完成，耗时: {elapsed:.2f}s，成功: {success_count}/{total}")
        
        # 显式关闭异步客户端，避免 Event loop is closed 错误
        try:
            await self.async_api_client.close()
        except Exception as e:
            # 忽略关闭错误，不影响主流程
            pass
        
        # 释放视频读取器
        del vr
        
        # Phase 3: 从临时文件读取，排序后写入最终文件
        print("  Phase 3: 整理临时文件，按场景索引排序...")
        l1_nodes_data = []
        if osp.exists(l1_temp_file):
            with open(l1_temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            l1_nodes_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        # 按 scene_index 排序
        l1_nodes_data.sort(key=lambda x: x.get('scene_index', 0))
        
        # Phase 4: 写入最终文件
        print("  Phase 4: 写入最终文件...")
        with open(l1_output_file, 'w', encoding='utf-8') as f:
            for node_data in l1_nodes_data:
                line = json.dumps(node_data, ensure_ascii=False)
                f.write(line + '\n')
        
        # Phase 5: 删除临时文件
        if osp.exists(l1_temp_file):
            os.remove(l1_temp_file)
            print(f"  [清理] 已删除临时文件: {l1_temp_file}")
        
        print(f"\n[L1生成完成] 共 {len(l1_nodes_data)} 个节点, 保存到: {l1_output_file}")
        
        return []  # 返回空列表，需要时可通过 load_l1_memories 从文件加载
    
    async def _process_single_scene_and_save_async(
        self,
        scene_idx: int,
        start_f_num: int,
        end_f_num: int,
        vr,
        vid_id: str,
        semaphore: asyncio.Semaphore,
        temp_file: str,
        file_lock: asyncio.Lock
    ) -> tuple:
        """
        异步处理单个场景并立即写入临时文件
        
        Returns:
            (scene_index, success, event_preview) 元组
        """
        try:
            # 调用场景处理逻辑
            l1_node = await self._process_single_scene_async(
                scene_idx=scene_idx,
                start_f_num=start_f_num,
                end_f_num=end_f_num,
                vr=vr,
                vid_id=vid_id,
                semaphore=semaphore
            )
            
            if l1_node:
                # 立即写入临时文件（使用锁保证线程安全）
                async with file_lock:
                    with open(temp_file, 'a', encoding='utf-8') as f:
                        line = json.dumps(l1_node.to_dict(), ensure_ascii=False)
                        f.write(line + '\n')
                
                # 返回成功信息
                event_preview = (l1_node.event_fact[:50] + "...") if l1_node.event_fact and len(l1_node.event_fact) > 50 else (l1_node.event_fact or "N/A")
                return (scene_idx, True, event_preview)
            else:
                return (scene_idx, False, "")
                
        except Exception as e:
            print(f"  [错误] 场景 {scene_idx} 处理失败: {e}")
            return (scene_idx, False, "")
    
    async def _process_single_scene_async(
        self,
        scene_idx: int,
        start_f_num: int,
        end_f_num: int,
        vr,
        vid_id: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[L1MemoryNode]:
        """
        异步处理单个场景
        """
        try:
            # 采样帧索引
            scene_frame_idxs = list(range(start_f_num, end_f_num, self.down_sample))
            
            if len(scene_frame_idxs) < 2:
                return None
            
            # 读取场景帧 (CPU/IO 密集型，保持同步)
            scene_frames = vr.get_batch(scene_frame_idxs).asnumpy()
            pil_images = [Image.fromarray(f.astype(np.uint8)) for f in scene_frames]
            
            # 自适应采样
            keyframe_data = self._adaptive_sampling(
                scene_frames=scene_frames,
                scene_frame_idxs=scene_frame_idxs,
                pil_images=pil_images,
                vid_id=vid_id,
                scene_idx=scene_idx
            )
            
            # 及时释放内存
            del scene_frames
            del pil_images
            
            if not keyframe_data:
                return None
            
            # 创建 L1 节点
            l1_node = L1MemoryNode(
                video_id=vid_id,
                scene_index=scene_idx,
                start_sec=round(start_f_num / self.fps, 2),
                end_sec=round(end_f_num / self.fps, 2),
                keyframe_indices=keyframe_data['indices'],
                keyframe_times=keyframe_data['times'],
                keyframe_paths=keyframe_data['paths']
            )
            
            # 选择用于生成 caption 的帧
            caption_images = self._select_frames_for_caption(
                keyframe_data['images'],
                keyframe_data['indices']
            )
            
            if caption_images:
                # 异步生成 caption (通过 API)，支持重试
                max_retries = 2
                parsed = None
                caption = ""
                
                for retry_attempt in range(max_retries + 1):  # 0, 1, 2 共3次尝试
                    caption = await self._generate_caption_async(caption_images, semaphore)
                    parsed = self._parse_caption(caption)
                    
                    # 检查是否解析成功
                    if self._is_parse_successful(parsed):
                        break
                    
                    if retry_attempt < max_retries:
                        print(f"  [重试] 场景 {scene_idx}: 解析失败，第 {retry_attempt + 1}/{max_retries} 次重试...")
                    else:
                        # 重试失败后，改为分步生成每个字段
                        print(f"  [分步生成] 场景 {scene_idx}: 切换为分步生成模式...")
                        caption, parsed = await self._generate_caption_stepwise_async(caption_images, semaphore)
                        n_valid = self._count_valid_fields(parsed)
                        print(f"  [分步完成] 场景 {scene_idx}: 成功生成 {n_valid} 个有效字段")
                
                l1_node.caption = caption
                l1_node.event_fact = parsed.get("event_fact", "")
            else:
                print(f"  [警告] 场景 {scene_idx}: 没有可用的图片")
            
            # 释放图像数据
            del keyframe_data['images']
            if caption_images:
                del caption_images
            
            return l1_node
            
        except Exception as e:
            print(f"  [错误] 场景 {scene_idx} 处理失败: {e}")
            return None
    
    async def _generate_caption_async(
        self,
        images: List[Image.Image],
        semaphore: asyncio.Semaphore
    ) -> str:
        """
        异步生成 caption (通过 vLLM API 调用)
        
        说明：并发下 vLLM 抖动/限流会导致一段时间内连续失败。
        这里做"可用性检测 + 指数退避重试"，避免把异常统一伪装成 "{}" 造成"连着解析失败"的错觉。
        """
        # 如果图片超过12张，均匀采样
        max_images = 12
        if len(images) > max_images:
            indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
            images = [images[i] for i in indices]
        
        # 构建 prompt
        user_prompt = self.l1_user_prompt_template.replace(
            "{{EVENT_NODE_NAME}}", "Scene Event"
        )
        
        # 构建消息内容 (图片使用 base64)
        content_parts = []
        for img in images:
            img_base64 = self._image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        content_parts.append({"type": "text", "text": user_prompt})
        
        messages = [
            {"role": "system", "content": self.l1_sys_prompt or "You are a helpful assistant."},
            {"role": "user", "content": content_parts}
        ]
        
        # 指数退避重试参数
        max_attempts = 3
        last_err = None
        
        for attempt in range(max_attempts):
            try:
                # 使用信号量控制并发
                async with semaphore:
                    response = await self.async_api_client.chat.completions.create(
                        model=self.base_api_model_name,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.0001, 
                        extra_body={
                            # 【关键 1】Top_P (核采样): 
                            # 只从概率累计最高的前 10% 里选。
                            "top_p": 0.1,
                            # 【关键 2】Repetition Penalty (重复惩罚): 
                            # 它会强行降低已经出现过的词的概率，迫使模型说新词。
                            "repetition_penalty": 1.15,
                            # 【注意】stop_token_ids 很容易把 JSON 提前截断（且 token id 与 served model 可能不一致）
                            # 如非确认一致，建议先关闭：
                            # "stop_token_ids": [151645, 151643],
                        }
                    )
                
                text = (response.choices[0].message.content or "").strip()
                
                # 检查返回内容是否可解析
                if self._caption_seems_parseable(text):
                    return text
                
                # 内容不可用：退避后重试
                backoff = 0.5 * (2 ** attempt)
                await asyncio.sleep(backoff)
                last_err = RuntimeError(f"返回内容不可解析 (len={len(text)})")
                
            except Exception as e:
                last_err = e
                backoff = 0.5 * (2 ** attempt)
                await asyncio.sleep(backoff)
        
        print(f"    [Caption生成错误 - async API] 重试 {max_attempts} 次失败: {last_err}")
        return ""  # 返回空串，让上层 parse/重试逻辑识别为失败
    
    def _caption_seems_parseable(self, caption: str) -> bool:
        """
        粗判断：返回内容是否像是可解析的 JSON / json codeblock。
        
        Args:
            caption: VLM 返回的原始字符串
            
        Returns:
            是否像可解析的 JSON
        """
        if not caption or not caption.strip():
            return False
        s = caption.strip()
        # 常见两种：直接 JSON 或 ```json ... ```
        if s.startswith("{") and ("event_fact" in s or "visual_evidence" in s):
            return True
        if "```json" in s:
            return True
        return False
    
    async def _generate_caption_stepwise_async(
        self,
        images: List[Image.Image],
        semaphore: asyncio.Semaphore
    ) -> Tuple[str, dict]:
        """
        分步生成 caption：分别调用大模型生成每个字段，然后拼接成 JSON
        
        这是备用方案，只有在正常生成失败时才使用
        
        Args:
            images: 图片列表
            semaphore: 并发控制信号量
            
        Returns:
            (caption_json_string, parsed_dict)
        """
        # 如果图片超过12张，均匀采样
        max_images = 12
        if len(images) > max_images:
            indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
            images = [images[i] for i in indices]
        
        # 构建图片消息部分（复用）
        image_content_parts = []
        for img in images:
            img_base64 = self._image_to_base64(img)
            image_content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        # 分步生成每个字段
        event_fact = await self._generate_single_field_async(
            image_content_parts, semaphore,
            field_name="event_fact",
            prompt="Based on the keyframes, describe the main action in 2-5 words (verb-noun format, e.g., 'cutting vegetables', 'typing on keyboard'). Output ONLY the action phrase, nothing else."
        )
        
        visual_evidence = await self._generate_single_field_async(
            image_content_parts, semaphore,
            field_name="visual_evidence",
            prompt="Describe the visual scene objectively in 1-2 sentences. Include: who/what is visible, their positions, colors, and what they are doing. Output ONLY the description, nothing else."
        )
        
        ocr_content = await self._generate_single_field_async(
            image_content_parts, semaphore,
            field_name="ocr_content",
            prompt="List any visible text/words in the images. IGNORE timestamps, timers, and frame numbers. If no readable text, output 'none'. Output ONLY the text strings separated by comma, nothing else."
        )
        
        # 处理 ocr_content 为列表格式
        ocr_list = []
        if ocr_content and ocr_content.lower() not in ["none", "n/a", "no text", "no visible text", ""]:
            # 分割并清理
            ocr_list = [item.strip().strip('"').strip("'") for item in ocr_content.split(",") if item.strip()]
        
        # 拼接成 JSON 字典
        result = {
            "event_fact": event_fact or "",
            "visual_evidence": visual_evidence or "",
            "ocr_content": ocr_list
        }
        
        caption_json = json.dumps(result, ensure_ascii=False)
        return caption_json, result
    
    async def _generate_single_field_async(
        self,
        image_content_parts: List[dict],
        semaphore: asyncio.Semaphore,
        field_name: str,
        prompt: str,
        max_attempts: int = 2
    ) -> str:
        """
        异步生成单个字段
        
        Args:
            image_content_parts: 图片内容列表（已编码为 base64）
            semaphore: 并发控制信号量
            field_name: 字段名（用于日志）
            prompt: 针对该字段的提示词
            max_attempts: 最大重试次数
            
        Returns:
            生成的字段值
        """
        # 构建消息
        content_parts = image_content_parts.copy()
        content_parts.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": "You are a precise visual analyzer. Follow instructions exactly. Be concise."},
            {"role": "user", "content": content_parts}
        ]
        
        for attempt in range(max_attempts):
            try:
                async with semaphore:
                    response = await self.async_api_client.chat.completions.create(
                        model=self.base_api_model_name,
                        messages=messages,
                        max_tokens=150,  # 单字段不需要太多 token
                        temperature=0.0001,
                        extra_body={
                            "top_p": 0.1,
                            "repetition_penalty": 1.15,
                        }
                    )
                
                text = (response.choices[0].message.content or "").strip()
                
                # 清理常见的格式问题
                text = text.strip('"').strip("'").strip()
                # 移除可能的 markdown 格式
                if text.startswith("```") and text.endswith("```"):
                    text = text[3:-3].strip()
                if text.startswith("`") and text.endswith("`"):
                    text = text[1:-1].strip()
                
                if text:
                    return text
                
                # 空响应，退避重试
                backoff = 0.3 * (2 ** attempt)
                await asyncio.sleep(backoff)
                
            except Exception as e:
                backoff = 0.3 * (2 ** attempt)
                await asyncio.sleep(backoff)
        
        # 重试失败，返回空字符串
        return ""
    
    def _run_scene_detection(self, video_path: str):
        """运行PySceneDetect场景检测"""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.pyscenesdetect_threshold))
        scene_manager.detect_scenes(video, show_progress=True)
        return scene_manager.get_scene_list()
    
    def _split_long_scenes(
        self,
        scene_list: list,
        fps: float,
        max_duration: float = 10.0
    ) -> List[Tuple[int, int]]:
        """
        切割超过指定时长的场景
        
        Args:
            scene_list: PySceneDetect返回的场景列表
            fps: 视频帧率
            max_duration: 最大场景持续时间（秒），默认10秒
            
        Returns:
            切割后的场景列表 [(start_frame, end_frame), ...]
        """
        max_frames = int(max_duration * fps)  # 最大帧数
        split_scenes = []
        
        for start_frame, end_frame in scene_list:
            start_f_num = start_frame.get_frames()
            end_f_num = end_frame.get_frames()
            scene_length = end_f_num - start_f_num
            
            if scene_length <= max_frames:
                # 场景时长不超过阈值，直接添加
                split_scenes.append((start_f_num, end_f_num))
            else:
                # 场景时长超过阈值，进行切割
                current_start = start_f_num
                while current_start < end_f_num:
                    current_end = min(current_start + max_frames, end_f_num)
                    split_scenes.append((current_start, current_end))
                    current_start = current_end
        
        return split_scenes
    
    def _adaptive_sampling(
        self,
        scene_frames: np.ndarray,
        scene_frame_idxs: List[int],
        pil_images: List[Image.Image],
        vid_id: str,
        scene_idx: int
    ) -> Optional[dict]:
        """
        自适应关键帧采样
        
        基于相邻帧的像素L2距离，选择变化显著的帧
        使用 CPU 进行计算 (不再依赖本地模型的 device)
        """
        try:
            # 使用 CPU 进行计算
            device = torch.device('cpu')
            
            # 计算特征 (使用像素作为特征)
            scene_tensor = torch.from_numpy(scene_frames).to(device).float()
            num_frames = scene_tensor.shape[0]
            scene_features = scene_tensor.reshape(num_frames, -1)
            
            # 计算自适应阈值
            distances = torch.norm(scene_features[1:] - scene_features[:-1], p=2, dim=1)
            mu = distances.mean().item()
            sigma = distances.std().item()
            tau = mu + self.adaptive_k * sigma
            
            # 采样关键帧
            selected_indices = [0]  # 总是包含第一帧
            last_feature = scene_features[0]
            
            for j in range(1, len(scene_features)):
                dist = torch.norm(scene_features[j] - last_feature, p=2).item()
                if dist > tau:
                    selected_indices.append(j)
                    last_feature = scene_features[j]
            
            # 保存关键帧
            scene_keyframe_dir = osp.join(
                self.output_dir, vid_id, "L1_keyframes", f"scene_{scene_idx:04d}"
            )
            os.makedirs(scene_keyframe_dir, exist_ok=True)
            
            keyframe_data = {
                'indices': [],
                'times': [],
                'paths': [],
                'images': []
            }
            
            for idx in selected_indices:
                frame_idx = scene_frame_idxs[idx]
                frame_img = pil_images[idx]
                
                # 保存图像
                img_filename = f"frame_{frame_idx:07d}.jpg"
                relative_path = osp.join(
                    vid_id, "L1_keyframes", f"scene_{scene_idx:04d}", img_filename
                )
                absolute_path = osp.join(self.output_dir, relative_path)
                frame_img.save(absolute_path, quality=90)
                
                keyframe_data['indices'].append(frame_idx)
                keyframe_data['times'].append(round(frame_idx / self.fps, 3))
                keyframe_data['paths'].append(relative_path)
                keyframe_data['images'].append(frame_img)
            
            return keyframe_data
            
        except Exception as e:
            print(f"    [采样错误] {e}")
            return None
    
    def _select_frames_for_caption(
        self,
        images: List[Image.Image],
        indices: List[int]
    ) -> List[Image.Image]:
        """
        选择用于生成caption的帧
        
        使用K-Means聚类选择代表性帧
        """
        if len(images) <= self.frames_per_l2:
            return images
        
        if not SKLEARN_AVAILABLE:
            # 均匀采样
            sample_indices = np.linspace(0, len(images) - 1, self.frames_per_l2, dtype=int)
            return [images[i] for i in sample_indices]
        
        try:
            # 提取特征
            features = self._get_visual_features(images)
            if features is None or len(features) == 0:
                sample_indices = np.linspace(0, len(images) - 1, self.frames_per_l2, dtype=int)
                return [images[i] for i in sample_indices]
            
            features_np = features.cpu().numpy()
            
            # K-Means聚类
            n_clusters = min(self.frames_per_l2, len(images))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features_np)
            
            # 选择每个聚类中心最近的帧
            selected_indices = set()
            for i in range(n_clusters):
                cluster_indices = np.where(kmeans.labels_ == i)[0]
                if len(cluster_indices) == 0:
                    continue
                
                cluster_features = features_np[cluster_indices]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.add(closest_idx)
            
            # 按时序排序
            selected_indices = sorted(list(selected_indices))
            return [images[i] for i in selected_indices]
            
        except Exception as e:
            print(f"    [K-Means错误] {e}")
            sample_indices = np.linspace(0, len(images) - 1, self.frames_per_l2, dtype=int)
            return [images[i] for i in sample_indices]
    
    def _get_visual_features(self, images: List[Image.Image]) -> Optional[torch.Tensor]:
        """
        提取视觉特征 (使用简单的像素特征，不再依赖本地模型)
        
        由于不再加载本地模型，改用简单的像素平均值作为特征
        """
        if not images:
            return None
        
        all_features = []
        
        for frame in images:
            try:
                # 缩放到固定大小并转换为tensor
                resized = frame.resize((224, 224), Image.LANCZOS)
                arr = np.array(resized).astype(np.float32) / 255.0
                # 使用每个颜色通道的均值和标准差作为简单特征
                feature = torch.tensor([
                    arr[:, :, 0].mean(), arr[:, :, 0].std(),
                    arr[:, :, 1].mean(), arr[:, :, 1].std(),
                    arr[:, :, 2].mean(), arr[:, :, 2].std(),
                ])
                all_features.append(feature)
            except Exception as e:
                continue
        
        if not all_features:
            return None
        
        return torch.stack(all_features, dim=0)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图片转换为base64字符串"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _generate_caption(self, images: List[Image.Image]) -> str:
        """生成caption - 使用 base_api (vLLM API)"""
        try:
            # 如果图片超过12张，均匀采样12张
            max_images = 12
            if len(images) > max_images:
                indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
                images = [images[i] for i in indices]
            
            # 构建消息内容
            content_parts = []
            for img in images:
                img_base64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
            
            user_prompt = self.l1_user_prompt_template.replace(
                "{{EVENT_NODE_NAME}}", "Scene Event"
            )
            content_parts.append({"type": "text", "text": user_prompt})
            
            messages = [
                {"role": "system", "content": self.l1_sys_prompt},
                {"role": "user", "content": content_parts}
            ]
            
            # 同步调用 base_api
            response = self.base_api_client.chat.completions.create(
                model=self.base_api_model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"    [Caption生成错误 - base_api] {e}")
            return "{}"
    
    def _parse_caption(self, caption: str) -> dict:
        """
        从 VLM 返回的 caption 中提取 JSON 数据
        支持处理被截断的 JSON（由于输出长度限制）
        
        采用从前往后逐字段解析策略：
        - 能解析出来的字段保留
        - 解析不出来的字段设为空
        
        Args:
            caption: VLM 返回的原始字符串
            
        Returns:
            解析后的字典
        """
        # 1. 定义简化后的 3 字段默认结构
        default_result = {
            "event_fact": "",
            "visual_evidence": "",
            "ocr_content": []
        }
        
        # 检查输入是否为空
        if not caption or not caption.strip():
            print(f"  [警告] VLM 返回了空的 caption")
            return default_result
        
        # 2. [提取逻辑] 尝试提取 ```json 包裹的内容
        clean_json_string = ""
        # 优先匹配完整的代码块
        match = re.search(r'```json\s*(.*?)\s*```', caption, re.DOTALL)
        if match:
            clean_json_string = match.group(1)
        else:
            # 尝试匹配未闭合的代码块 (截断情况)
            match_start = re.search(r'```json\s*(.*)', caption, re.DOTALL)
            if match_start:
                clean_json_string = match_start.group(1)
            else:
                # 尝试直接清理 markdown 标记
                clean_json_string = caption.strip().replace('```json', '').replace('```', '').strip()

        # 3. 尝试标准解析
        try:
            parsed_caption_data = json.loads(clean_json_string)
            
            # 展开嵌套结构 (event_log, action_log, log_entry 等)
            parsed_caption_data = self._flatten_nested_json(parsed_caption_data)
            
            # 数据清洗与合并（支持字段名变体）
            final_data = self._normalize_field_names(parsed_caption_data, default_result)
            
            return final_data
            
        except json.JSONDecodeError:
            # 4. 进入鲁棒修复模式 (针对截断或格式错误的 JSON)
            return self._repair_truncated_json(clean_json_string, default_result)
    
    def _flatten_nested_json(self, data: dict) -> dict:
        """
        展开嵌套的 JSON 结构
        
        处理模型返回的嵌套包装，如：
        - {"event_log": {"event_fact": "..."}} -> {"event_fact": "..."}
        - {"action_log": {"fact_event": "..."}} -> {"fact_event": "..."}
        """
        if not isinstance(data, dict):
            return data
        
        # 常见的嵌套包装 key
        wrapper_keys = ["event_log", "action_log", "log_entry", "output", "result", "response", "data"]
        
        for wrapper_key in wrapper_keys:
            if wrapper_key in data and isinstance(data[wrapper_key], dict):
                # 展开嵌套，将内层字段提升到顶层
                inner = data[wrapper_key]
                # 递归展开
                inner = self._flatten_nested_json(inner)
                # 合并到顶层（内层优先）
                for k, v in inner.items():
                    if k not in data or k == wrapper_key:
                        data[k] = v
        
        return data
    
    def _normalize_field_names(self, parsed_data: dict, default_result: dict) -> dict:
        """
        标准化字段名，支持常见变体
        
        如：fact_event -> event_fact
        """
        result = default_result.copy()
        
        # 字段名映射：变体名 -> 标准名
        field_aliases = {
            "event_fact": ["event_fact", "fact_event", "event", "action", "action_summary", "summary", "fact"],
            "visual_evidence": ["visual_evidence", "evidence", "visual", "description", "scene", "scene_description", "log"],
            "ocr_content": ["ocr_content", "ocr", "text", "texts", "ocr_text", "visible_text"]
        }
        
        for standard_field, aliases in field_aliases.items():
            for alias in aliases:
                if alias in parsed_data:
                    value = parsed_data[alias]
                    # 确保类型正确
                    if standard_field == "ocr_content":
                        if isinstance(value, list):
                            result[standard_field] = value
                        elif isinstance(value, str):
                            result[standard_field] = [value] if value else []
                    else:
                        if isinstance(value, str):
                            result[standard_field] = value
                        elif isinstance(value, dict):
                            # 如果值是字典，尝试提取其中的文本
                            result[standard_field] = str(value)
                    break  # 找到第一个匹配的就停止
        
        return result

    def _repair_truncated_json(self, json_string: str, default_result: dict) -> dict:
        """
        通过正则强行提取字段，即使 JSON 不完整
        支持字段名变体和嵌套结构
        """
        result = default_result.copy()
        
        # 字段名变体映射（正则版本）
        # 标准字段: (是否为列表, [变体名列表])
        fields_config = {
            "event_fact": (False, ["event_fact", "fact_event", "event", "action", "action_summary", "summary", "fact"]),
            "visual_evidence": (False, ["visual_evidence", "evidence", "visual", "description", "scene", "scene_description", "log"]),
            "ocr_content": (True, ["ocr_content", "ocr", "text", "texts", "ocr_text", "visible_text"])
        }
        
        extracted_count = 0
        
        for standard_field, (is_list, aliases) in fields_config.items():
            # 尝试每个变体名
            for alias in aliases:
                try:
                    if is_list:
                        # --- 提取列表 (List) ---
                        pattern = rf'"{alias}"\s*:\s*\[(.*?)\]'
                        match = re.search(pattern, json_string, re.DOTALL)
                        
                        if match:
                            content = match.group(1)
                            items = re.findall(r'"(.*?)(?<!\\)"', content)
                            result[standard_field] = items
                            extracted_count += 1
                            break  # 找到就停止
                            
                    else:
                        # --- 提取字符串 (String) ---
                        pattern = rf'"{alias}"\s*:\s*"(.*?)(?<!\\)"'
                        match = re.search(pattern, json_string, re.DOTALL)
                        
                        if match:
                            result[standard_field] = match.group(1)
                            extracted_count += 1
                            break  # 找到就停止
                            
                except Exception:
                    continue

        if extracted_count == 0 and len(json_string) > 20:
            print(f"  [修复失败] 无法提取有效字段。原始内容预览: {json_string[:100]}...")
             
        return result
    
    def _is_parse_successful(self, parsed: dict) -> bool:
        """
        检查解析结果是否成功
        
        Args:
            parsed: 解析后的字典
            
        Returns:
            是否解析成功
        """
        # 检查 event_fact 字段是否有效（不是解析失败标记且不为空）
        event_fact = parsed.get("event_fact", "")
        if not event_fact or event_fact == "[解析失败]":
            return False
        return True
    
    def _get_empty_caption_result(self) -> dict:
        """
        返回保留所有字段但内容为空的默认结构
        用于重试失败后保持数据结构一致性
        
        Returns:
            空内容的字典
        """
        return {
            "event_fact": "",
            "visual_evidence": "",
            "ocr_content": []
        }
    
    def _clean_parsed_result(self, parsed: dict) -> dict:
        """
        清理解析结果，将失败标记替换为空值
        保留能成功解析的字段，只清理失败的部分
        
        Args:
            parsed: 解析后的字典（可能包含部分成功、部分失败的字段）
            
        Returns:
            清理后的字典
        """
        result = parsed.copy()
        
        # 字符串字段：检查是否为失败标记或无效值
        string_fields = ["event_fact", "visual_evidence"]
        invalid_markers = ["[解析失败]", "[Parse Failed]", "null", "None"]
        
        for field in string_fields:
            value = result.get(field, "")
            if not value or value in invalid_markers:
                result[field] = ""
        
        # 列表字段：确保是列表类型，清理无效元素
        list_fields = ["ocr_content"]
        
        for field in list_fields:
            value = result.get(field, [])
            if not isinstance(value, list):
                result[field] = []
            else:
                # 清理列表中的无效元素
                result[field] = [item for item in value if item and item not in invalid_markers]
        
        return result
    
    def _count_valid_fields(self, parsed: dict) -> int:
        """
        统计解析结果中有效字段的数量
        
        Args:
            parsed: 解析后的字典
            
        Returns:
            有效字段数量
        """
        count = 0
        
        # 检查字符串字段
        if parsed.get("event_fact"):
            count += 1
        if parsed.get("visual_evidence"):
            count += 1
        
        # 检查列表字段
        ocr = parsed.get("ocr_content", [])
        if isinstance(ocr, list) and len(ocr) > 0:
            count += 1
        
        return count
    
    # ========================================================================
    #                           L1 加载
    # ========================================================================
    
    def load_l1_memories(self, video_path: str) -> List[L1MemoryNode]:
        """
        从文件加载L1记忆
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            L1节点列表
        """
        vid_id = osp.splitext(osp.basename(video_path))[0]
        l1_output_file = osp.join(self.output_dir, vid_id, "episodic_memories_L1.json")
        
        if not osp.exists(l1_output_file):
            print(f"  [错误] L1文件不存在: {l1_output_file}")
            return []
        
        l1_memories = []
        with open(l1_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                l1_node = L1MemoryNode(
                    video_id=data.get("video_id", vid_id),
                    scene_index=data.get("scene_index", 0),
                    start_sec=data.get("start_sec", 0.0),
                    end_sec=data.get("end_sec", 0.0),
                    keyframe_indices=data.get("keyframe_indices", []),
                    keyframe_times=data.get("keyframe_times", []),
                    keyframe_paths=data.get("keyframe_paths", []),
                    caption=data.get("caption"),
                    event_fact=data.get("event_fact")
                )
                l1_memories.append(l1_node)
        
        print(f"  [加载] 从文件加载了 {len(l1_memories)} 个L1节点")
        return l1_memories
    
    # ========================================================================
    #                           L2 生成
    # ========================================================================
    
    def generate_l2_memories(
        self,
        l1_memories: List[L1MemoryNode],
        video_path: str,
        use_vlm_decision: bool = True,
        skip_if_exists: bool = True
    ) -> List[L2MemoryNode]:
        """
        生成L2语义记忆
        
        流程:
        1. 遍历L1节点
        2. 对每个L1进行1fps采帧
        3. 使用Memory Manager Agent决定动作 (CREATE_NEW/MERGE/DISCARD/UPDATE)
        4. 执行动作并更新L2记忆
        5. 生成working_memory (finalized_summary已移至L3阶段全局生成)
        
        Args:
            l1_memories: L1节点列表
            video_path: 视频文件路径 (用于1fps采帧)
            use_vlm_decision: 是否使用VLM进行决策
            skip_if_exists: 如果L2文件已存在是否跳过
            
        Returns:
            L2节点列表 (如果跳过则返回空列表)
        """
        if not l1_memories:
            return []
        
        vid_id = l1_memories[0].video_id
        
        # 输出文件路径
        video_output_dir = osp.join(self.output_dir, vid_id)
        l2_output_file = osp.join(video_output_dir, "semantic_memories_L2.json")
        
        # 检查是否已经处理过该视频
        if skip_if_exists and osp.exists(l2_output_file):
            print(f"  [跳过] 视频 {vid_id} L2记忆已存在，跳过处理")
            print(f"    L2: {l2_output_file}")
            return []
        
        print(f"\n[L2生成] 处理视频: {vid_id}, 共 {len(l1_memories)} 个L1节点")
        
        # 重置Agent
        self.memory_agent.reset()
        
        # 打开文件用于边处理边写入 (仿照L1的保存方式)
        f_l2_out = None
        l2_count = 0
        finalized_l2_indices = set()  # 记录已经保存过的L2节点索引
        
        try:
            f_l2_out = open(l2_output_file, 'w', encoding='utf-8')
            
            # 遍历L1节点
            for l1_node in tqdm(l1_memories, desc="聚合L1到L2"):
                # 1. 从L1的已有关键帧中按1fps规则采样
                l1_frames_1fps, l1_times_1fps = self._sample_l1_at_1fps(
                    l1_node, vid_id
                )
                
                if not l1_frames_1fps:
                    print(f"    [跳过] L1[{l1_node.scene_index}] 1fps采帧失败")
                    continue
                
                # 2. 提取visual_evidence
                visual_evidence = self._extract_visual_evidence(l1_node)
                
                # 3. 决策
                action = self.memory_agent.decide_action(
                    l1_node,
                    l1_frames_1fps=l1_frames_1fps,
                    l1_times_1fps=l1_times_1fps,
                    use_vlm=use_vlm_decision
                )
                
                # 4. 执行动作
                affected_l2, need_summary = self.memory_agent.execute_action(
                    action=action,
                    current_l1=l1_node,
                    l1_frames_1fps=l1_frames_1fps,
                    l1_times_1fps=l1_times_1fps,
                    visual_evidence=visual_evidence
                )
                 
                # 5. 根据动作生成文本记忆，并边处理边保存已完成的L2节点
                if action == L2Action.CREATE_NEW and need_summary:
                    # 新L2创建时，为刚结束的L2生成working_memory (finalized_summary移到L3阶段全局生成)
                    if len(self.memory_agent.state.active_l2_nodes) >= 2:
                        prev_l2 = self.memory_agent.state.active_l2_nodes[-2]
                        # 生成working_memory
                        self.memory_agent.generate_working_memory(prev_l2)
                        # 选择L3代表帧
                        self._finalize_l2_node_v2(prev_l2)
                        print(f"      生成L2[{prev_l2.l2_index}] working_memory")
                        
                        # 边处理边写入文件 (该L2节点已完成)
                        if prev_l2.l2_index not in finalized_l2_indices:
                            line = json.dumps(prev_l2.to_dict(), ensure_ascii=False)
                            f_l2_out.write(line + '\n')
                            f_l2_out.flush()
                            finalized_l2_indices.add(prev_l2.l2_index)
                            l2_count += 1
                
                elif action == L2Action.UPDATE and affected_l2:
                    # UPDATE需要重写working_memory
                    self.memory_agent.generate_working_memory(affected_l2)
                    print(f"      重写L2[{affected_l2.l2_index}] working_memory")
                
                print(f"    L1[{l1_node.scene_index}] -> {action.value}")
            
            # 后处理: 为最后一个L2生成working_memory并保存
            l2_memories = self.memory_agent.state.active_l2_nodes
            if l2_memories:
                last_l2 = l2_memories[-1]
                if not last_l2.working_memory:
                    self.memory_agent.generate_working_memory(last_l2)
                # 选择L3代表帧
                self._finalize_l2_node_v2(last_l2)
                
                # 保存最后一个L2节点
                if last_l2.l2_index not in finalized_l2_indices:
                    line = json.dumps(last_l2.to_dict(), ensure_ascii=False)
                    f_l2_out.write(line + '\n')
                    f_l2_out.flush()
                    finalized_l2_indices.add(last_l2.l2_index)
                    l2_count += 1
                    
        finally:
            if f_l2_out is not None:
                f_l2_out.close()
        
        print(f"\n[L2生成完成] 共 {l2_count} 个L2节点, 保存到: {l2_output_file}")
        
        return []  # 返回空列表，数据已保存到文件
    
    def _sample_l1_at_1fps(
        self,
        l1_node: L1MemoryNode,
        vid_id: str
    ) -> Tuple[List[str], List[float]]:
        """
        从L1节点的已有关键帧中按1fps规则采样
        
        不从原始视频重新采帧，而是从L1节点中已存储的关键帧中，
        根据L1节点的时间跨度按照1fps规则采样。
        
        采样规则:
        - 时间跨度不足1秒按1秒计算（至少采1帧）
        - 时间跨度向下取整（如0.1-3.2秒按3秒计算，采3帧）
        
        Args:
            vr: 视频读取器（未使用，保留参数兼容性）
            l1_node: L1节点
            fps: 视频帧率（未使用，保留参数兼容性）
            vid_id: 视频ID（未使用，保留参数兼容性）
            output_dir: 输出目录（未使用，保留参数兼容性）
            
        Returns:
            (frame_paths, frame_times): 帧路径列表和时间戳列表
        """
        try:
            # 检查L1节点是否有关键帧
            if not l1_node.keyframe_paths or not l1_node.keyframe_times:
                print(f"    [警告] L1[{l1_node.scene_index}] 没有关键帧")
                return [], []
            
            # 计算时间跨度
            duration = l1_node.end_sec - l1_node.start_sec
            
            # 计算需要采样的帧数（1fps规则）
            # 不足1秒按1秒计算，其他向上取整
            if duration < 1.0:
                n_frames_needed = 1
            else:
                n_frames_needed = math.ceil(duration)  # 向上取整
            
            # 获取L1的所有关键帧
            total_frames = len(l1_node.keyframe_paths)
            
            # 如果L1关键帧数量少于等于需要的帧数，直接返回所有帧
            if total_frames <= n_frames_needed:
                return l1_node.keyframe_paths.copy(), l1_node.keyframe_times.copy()
            
            # 从L1关键帧中均匀采样
            indices = np.linspace(0, total_frames - 1, n_frames_needed, dtype=int)
            
            sampled_paths = [l1_node.keyframe_paths[i] for i in indices]
            sampled_times = [l1_node.keyframe_times[i] for i in indices]
            
            return sampled_paths, sampled_times
            
        except Exception as e:
            print(f"    [1fps采样错误] L1[{l1_node.scene_index}]: {e}")
            return [], []
    
    def _extract_visual_evidence(self, l1_node: L1MemoryNode) -> str:
        """从L1的caption中提取visual_evidence"""
        if not l1_node.caption:
            return ""
        
        try:
            parsed = self._parse_caption(l1_node.caption)
            visual_evidence = parsed.get("visual_evidence", "")
            if isinstance(visual_evidence, dict):
                # 如果是字典，转换为字符串
                visual_evidence = json.dumps(visual_evidence, ensure_ascii=False)
            return visual_evidence
        except:
            return ""
    
    def _finalize_l2_node_v2(self, l2_node: L2MemoryNode):
        """
        完善L2节点 (v2版本)
        
        从l1_details中选择L3代表帧
        """
        # 收集所有帧路径和时间
        all_paths = l2_node.representative_frames
        all_times = l2_node.representative_times
        
        if not all_paths:
            return
        
        # L3代表帧: 均匀采样
        if len(all_paths) > self.frames_per_l3:
            indices = np.linspace(0, len(all_paths) - 1, self.frames_per_l3, dtype=int)
            l2_node.l3_keyframe_paths = [all_paths[i] for i in indices]
            l2_node.l3_keyframe_times = [all_times[i] if i < len(all_times) else 0 for i in indices]
        else:
            l2_node.l3_keyframe_paths = all_paths.copy()
            l2_node.l3_keyframe_times = all_times.copy() if all_times else []
        
        # 聚合event_summary (从l1_details)
        if not l2_node.event_summary:
            facts = [d.event_fact for d in l2_node.l1_details if d.event_fact]
            if facts:
                l2_node.event_summary = " -> ".join(facts)
    
    def _finalize_l2_node(
        self,
        l2_node: L2MemoryNode,
        l1_memories: List[L1MemoryNode]
    ):
        """
        完善L2节点
        
        使用K-Means从所有包含的L1帧中选择代表帧
        """
        # 收集所有关联L1的关键帧
        all_frames = []
        all_paths = []
        all_times = []
        
        for scene_idx in l2_node.l1_scene_indices:
            for l1 in l1_memories:
                if l1.scene_index == scene_idx:
                    for path in l1.keyframe_paths:
                        full_path = osp.join(self.output_dir, path)
                        if osp.exists(full_path):
                            img = Image.open(full_path).convert("RGB")
                            all_frames.append(img)
                            all_paths.append(path)
                    all_times.extend(l1.keyframe_times)
        
        if not all_frames:
            return
        
        # K-Means选择代表帧
        if len(all_frames) > self.frames_per_l2 and SKLEARN_AVAILABLE:
            try:
                features = self._get_visual_features(all_frames)
                if features is not None:
                    features_np = features.cpu().numpy()
                    n_clusters = min(self.frames_per_l2, len(all_frames))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features_np)
                    
                    selected_indices = set()
                    for i in range(n_clusters):
                        cluster_indices = np.where(kmeans.labels_ == i)[0]
                        if len(cluster_indices) == 0:
                            continue
                        cluster_features = features_np[cluster_indices]
                        center = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_features - center, axis=1)
                        closest_idx = cluster_indices[np.argmin(distances)]
                        selected_indices.add(closest_idx)
                    
                    selected_indices = sorted(list(selected_indices))
                    l2_node.representative_frames = [all_paths[i] for i in selected_indices]
                    l2_node.representative_times = [all_times[i] if i < len(all_times) else 0 for i in selected_indices]
            except:
                pass
        
        if not l2_node.representative_frames:
            # 均匀采样
            indices = np.linspace(0, len(all_paths) - 1, min(self.frames_per_l2, len(all_paths)), dtype=int)
            l2_node.representative_frames = [all_paths[i] for i in indices]
            l2_node.representative_times = [all_times[i] if i < len(all_times) else 0 for i in indices]
        
        # L3代表帧 (更精简)
        if len(l2_node.representative_frames) > self.frames_per_l3:
            indices = np.linspace(0, len(l2_node.representative_frames) - 1, self.frames_per_l3, dtype=int)
            l2_node.l3_keyframe_paths = [l2_node.representative_frames[i] for i in indices]
            l2_node.l3_keyframe_times = [l2_node.representative_times[i] for i in indices]
        else:
            l2_node.l3_keyframe_paths = l2_node.representative_frames.copy()
            l2_node.l3_keyframe_times = l2_node.representative_times.copy()
        
        # 聚合caption
        captions = []
        for scene_idx in l2_node.l1_scene_indices:
            for l1 in l1_memories:
                if l1.scene_index == scene_idx and l1.event_fact:
                    captions.append(l1.event_fact)
        
        if captions:
            l2_node.event_summary = " -> ".join(captions)
    
    # ========================================================================
    #                           主流程
    # ========================================================================
    
    def generate_all_memories(
        self,
        video_path: str,
        use_vlm_decision: bool = False,  # 默认使用规则决策 (更快)
        skip_if_exists: bool = True
    ) -> Tuple[List[L1MemoryNode], List[L2MemoryNode]]:
        """
        生成完整的多层次记忆
        
        流程:
        1. 检查是否已处理过 (L1和L2都存在则跳过)
        2. 生成L1记忆 (边处理边存储)
        3. 从文件重新加载L1记忆
        4. 生成L2记忆
        
        Args:
            video_path: 视频文件路径
            use_vlm_decision: 是否使用VLM进行L2聚合决策
            skip_if_exists: 如果记忆文件已存在是否跳过
            
        Returns:
            (l1_memories, l2_memories) - 如果跳过则返回空列表
        """
        vid_id = osp.splitext(osp.basename(video_path))[0]
        print("=" * 60)
        print(f"  视频多层次记忆生成: {vid_id}")
        print("=" * 60)
        
        # 输出文件路径
        video_output_dir = osp.join(self.output_dir, vid_id)
        l1_output_file = osp.join(video_output_dir, "episodic_memories_L1.json")
        l2_output_file = osp.join(video_output_dir, "semantic_memories_L2.json")
        
        # 检查是否已经处理过该视频
        if skip_if_exists and osp.exists(l1_output_file) and osp.exists(l2_output_file):
            print(f"  [跳过] 视频 {vid_id} 已处理完成，跳过")
            print(f"    L1: {l1_output_file}")
            print(f"    L2: {l2_output_file}")
            return [], []
        
        # 生成L1记忆 (边处理边存储，返回空列表)
        self.generate_l1_memories(video_path, skip_if_exists=skip_if_exists)
        
        # 从文件重新加载L1记忆 (避免内存占用过大)
        l1_memories = self.load_l1_memories(video_path)
        
        if not l1_memories:
            print("[警告] L1记忆为空，跳过L2生成")
            return [], []
        
        # 生成L2记忆
        l2_memories = self.generate_l2_memories(
            l1_memories,
            video_path=video_path,
            use_vlm_decision=use_vlm_decision,
            skip_if_exists=skip_if_exists
        )
        
        print("\n" + "=" * 60)
        print(f"  完成! L1: {len(l1_memories)} 个节点, L2: {len(l2_memories)} 个节点")
        print("=" * 60)

        # 释放内存
        del l1_memories

    # ============================================================================
    #                       L3 知识图谱生成
    # ============================================================================
    
    def generate_l3_knowledge_graph(
        self,
        video_path: str,
        window_size: int = 4,
        stride: int = 2,
        skip_if_exists: bool = True
    ) -> Tuple[List[GlobalFinalizedSummary], List[KGNode], List[KGEdge]]:
        """
        生成L3知识图谱
        
        流程:
        1. 加载L1和L2记忆
        2. 用滑动窗口生成全局finalized_summary (边生成边保存)
        3. 从L1中提取实体节点 (person, object, place, text)
        4. 从L2中创建事件节点
        5. 建立实体与事件之间的边
        6. 建立事件之间的时序和因果边
        
        Args:
            video_path: 视频文件路径
            window_size: 滑动窗口大小 (包含多少个L2节点)
            stride: 滑动步长
            skip_if_exists: 如果L3文件已存在是否跳过
            
        Returns:
            (global_summaries, kg_nodes, kg_edges)
        """
        vid_id = osp.splitext(osp.basename(video_path))[0]
        print("\n" + "=" * 60)
        print(f"  L3知识图谱生成: {vid_id}")
        print("=" * 60)
        
        # 输出文件路径
        video_output_dir = osp.join(self.output_dir, vid_id)
        summary_output_file = osp.join(video_output_dir, "global_finalized_summaries_L3.json")
        kg_output_file = osp.join(video_output_dir, "knowledge_graph_L3.json")
        
        # 检查是否已经处理过
        if skip_if_exists and osp.exists(summary_output_file) and osp.exists(kg_output_file):
            print(f"  [跳过] 视频 {vid_id} L3已存在，跳过处理")
            return [], [], []
        
        # 加载L1和L2记忆
        l1_memories = self.load_l1_memories(video_path)
        l2_memories = self.load_l2_memories(video_path)
        
        if not l2_memories:
            print("[警告] L2记忆为空，跳过L3生成")
            return [], [], []
        
        # 1. 生成全局finalized_summary (边生成边保存)
        global_summaries = self._generate_global_finalized_summaries(
            l2_memories, vid_id, summary_output_file, window_size, stride
        )
        
        # 2. 构建知识图谱
        kg_nodes, kg_edges = self._build_knowledge_graph(
            l1_memories, l2_memories, global_summaries, vid_id, kg_output_file
        )
        
        print(f"\n[L3生成完成]")
        print(f"  全局Summary: {len(global_summaries)} 个")
        print(f"  KG节点: {len(kg_nodes)} 个")
        print(f"  KG边: {len(kg_edges)} 个")
        print(f"  保存到: {summary_output_file}")
        print(f"         {kg_output_file}")
        
        return global_summaries, kg_nodes, kg_edges
    
    def _generate_global_finalized_summaries(
        self,
        l2_memories: List[L2MemoryNode],
        vid_id: str,
        output_file: str,
        window_size: int = 4,
        stride: int = 2
    ) -> List[GlobalFinalizedSummary]:
        """
        用滑动窗口生成全局finalized_summary
        
        Args:
            l2_memories: L2节点列表
            vid_id: 视频ID
            output_file: 输出文件路径
            window_size: 窗口大小
            stride: 滑动步长
            
        Returns:
            GlobalFinalizedSummary列表
        """
        global_summaries = []
        summary_id = 0
        
        os.makedirs(osp.dirname(output_file), exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                # 滑动窗口遍历L2节点
                i = 0
                while i < len(l2_memories):
                    # 获取当前窗口内的L2节点
                    window_end = min(i + window_size, len(l2_memories))
                    window_l2s = l2_memories[i:window_end]
                    
                    if len(window_l2s) < 2:
                        # 窗口内节点太少，跳过
                        i += stride
                        continue
                    
                    # 构建prompt
                    memories_for_prompt = []
                    for l2 in window_l2s:
                        memory_text = l2.working_memory or l2.event_summary or ""
                        if memory_text:
                            memories_for_prompt.append({
                                "l2_index": l2.l2_index,
                                "time_range": f"{l2.start_sec:.1f}s - {l2.end_sec:.1f}s",
                                "memory": memory_text
                            })
                    
                    if len(memories_for_prompt) < 2:
                        i += stride
                        continue
                    
                    # 生成因果分析
                    prompt = self._build_global_summary_prompt(memories_for_prompt)
                    response = self.memory_agent._text_inference(prompt, max_tokens=512)
                    
                    # 解析响应 (简单解析，后续可改进)
                    causal_chain, key_entities, overall_summary = self._parse_causal_response(response)
                    
                    # 创建GlobalFinalizedSummary对象
                    summary = GlobalFinalizedSummary(
                        summary_id=summary_id,
                        video_id=vid_id,
                        l2_start_index=window_l2s[0].l2_index,
                        l2_end_index=window_l2s[-1].l2_index,
                        start_sec=window_l2s[0].start_sec,
                        end_sec=window_l2s[-1].end_sec,
                        causal_chain=causal_chain,
                        key_entities=key_entities,
                        overall_summary=overall_summary
                    )
                    
                    global_summaries.append(summary)
                    
                    # 边生成边保存
                    line = json.dumps(summary.to_dict(), ensure_ascii=False)
                    f_out.write(line + '\n')
                    f_out.flush()
                    
                    print(f"  [Summary {summary_id}] L2[{summary.l2_start_index}-{summary.l2_end_index}]: {overall_summary[:50]}...")
                    
                    summary_id += 1
                    i += stride
                    
        except Exception as e:
            print(f"[全局Summary生成错误] {e}")
            import traceback
            traceback.print_exc()
        
        return global_summaries
    
    def _build_global_summary_prompt(self, memories: List[dict]) -> str:
        """构建全局因果总结的prompt"""
        
        prompt = """## TASK: Generate Global Causal Summary for Knowledge Graph

Analyze the sequence of video events and identify causal/temporal relationships.

## EVENT SEQUENCE:
"""
        for mem in memories:
            prompt += f"\n[L2-{mem['l2_index']}] ({mem['time_range']}): {mem['memory']}"
        
        prompt += """

## INSTRUCTIONS:
1. Identify CAUSAL relationships between events (cause -> effect)
2. Identify TEMPORAL relationships (before -> after)
3. Extract KEY ENTITIES that appear across events
4. Provide a high-level summary

## OUTPUT FORMAT (use exactly this format):
CAUSAL_CHAIN: Event A -> Event B -> Event C (describe cause-effect links)
KEY_ENTITIES: entity1, entity2, entity3
OVERALL_SUMMARY: One paragraph summary of this sequence

Keep it concise for knowledge graph construction.
"""
        return prompt
    
    def _parse_causal_response(self, response: str) -> Tuple[str, List[str], str]:
        """
        解析VLM的因果分析响应
        
        Returns:
            (causal_chain, key_entities, overall_summary)
        """
        causal_chain = ""
        key_entities = []
        overall_summary = ""
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("CAUSAL_CHAIN:"):
                causal_chain = line[len("CAUSAL_CHAIN:"):].strip()
            elif line.startswith("KEY_ENTITIES:"):
                entities_str = line[len("KEY_ENTITIES:"):].strip()
                key_entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif line.startswith("OVERALL_SUMMARY:"):
                overall_summary = line[len("OVERALL_SUMMARY:"):].strip()
        
        # 如果解析失败，使用整个响应作为overall_summary
        if not overall_summary:
            overall_summary = response.strip()
        
        return causal_chain, key_entities, overall_summary
    
    def _build_knowledge_graph(
        self,
        l1_memories: List[L1MemoryNode],
        l2_memories: List[L2MemoryNode],
        global_summaries: List[GlobalFinalizedSummary],
        vid_id: str,
        output_file: str
    ) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        构建知识图谱
        
        1. 从L1的caption中提取实体节点
        2. 从L2创建事件节点
        3. 建立边 (PARTICIPATES_IN, LOCATED_IN, BEFORE, CAUSES)
        
        Args:
            l1_memories: L1节点列表
            l2_memories: L2节点列表  
            global_summaries: 全局因果总结列表
            vid_id: 视频ID
            output_file: 输出文件路径
            
        Returns:
            (kg_nodes, kg_edges)
        """
        kg_nodes: List[KGNode] = []
        kg_edges: List[KGEdge] = []
        
        # 实体去重字典 {entity_name_type: KGNode}
        entity_dict: Dict[str, KGNode] = {}
        
        # L1索引到L2索引的映射
        l1_to_l2_map: Dict[int, int] = {}
        for l2 in l2_memories:
            for l1_idx in l2.l1_scene_indices:
                l1_to_l2_map[l1_idx] = l2.l2_index
        
        edge_id_counter = 0
        
        # 1. 从L1中提取实体节点
        print("\n  [Step 1] 提取实体节点...")
        for l1 in tqdm(l1_memories, desc="提取实体"):
            # 解析L1的caption
            entities = self._extract_entities_from_l1(l1)
            
            for entity_type, entity_names in entities.items():
                for name in entity_names:
                    # 生成唯一key
                    entity_key = f"{entity_type}_{name}_{vid_id}"
                    
                    if entity_key not in entity_dict:
                        # 创建新的实体节点
                        node = KGNode(
                            node_id=entity_key,
                            node_type=KGNodeType[entity_type.upper()],
                            name=name,
                            video_id=vid_id,
                            first_seen_sec=l1.start_sec,
                            last_seen_sec=l1.end_sec
                        )
                        entity_dict[entity_key] = node
                        kg_nodes.append(node)
                    else:
                        # 更新时间范围
                        existing_node = entity_dict[entity_key]
                        if l1.start_sec < existing_node.first_seen_sec:
                            existing_node.first_seen_sec = l1.start_sec
                        if l1.end_sec > existing_node.last_seen_sec:
                            existing_node.last_seen_sec = l1.end_sec
        
        # 2. 从L2创建事件节点
        print("  [Step 2] 创建事件节点...")
        event_nodes: Dict[int, KGNode] = {}  # l2_index -> KGNode
        for l2 in l2_memories:
            event_node = KGNode(
                node_id=f"event_{l2.l2_index}_{vid_id}",
                node_type=KGNodeType.EVENT,
                name=f"Event_{l2.l2_index}",
                video_id=vid_id,
                first_seen_sec=l2.start_sec,
                last_seen_sec=l2.end_sec,
                l2_index=l2.l2_index,
                event_summary=l2.event_summary,
                working_memory=l2.working_memory
            )
            kg_nodes.append(event_node)
            event_nodes[l2.l2_index] = event_node
        
        # 3. 建立实体与事件之间的边 (PARTICIPATES_IN)
        print("  [Step 3] 建立实体-事件边...")
        for l1 in l1_memories:
            l2_idx = l1_to_l2_map.get(l1.scene_index)
            if l2_idx is None:
                continue
            
            event_node_id = f"event_{l2_idx}_{vid_id}"
            entities = self._extract_entities_from_l1(l1)
            
            for entity_type, entity_names in entities.items():
                for name in entity_names:
                    entity_key = f"{entity_type}_{name}_{vid_id}"
                    if entity_key in entity_dict:
                        # 创建 PARTICIPATES_IN 边
                        edge = KGEdge(
                            edge_id=f"edge_{edge_id_counter}",
                            source_id=entity_key,
                            target_id=event_node_id,
                            edge_type=KGEdgeType.PARTICIPATES_IN,
                            video_id=vid_id,
                            timestamp_sec=l1.start_sec
                        )
                        kg_edges.append(edge)
                        edge_id_counter += 1
                        
                        # 如果是place类型，额外创建 LOCATED_IN 边
                        if entity_type == "place":
                            edge_loc = KGEdge(
                                edge_id=f"edge_{edge_id_counter}",
                                source_id=event_node_id,
                                target_id=entity_key,
                                edge_type=KGEdgeType.LOCATED_IN,
                                video_id=vid_id,
                                timestamp_sec=l1.start_sec
                            )
                            kg_edges.append(edge_loc)
                            edge_id_counter += 1
        
        # 4. 建立事件之间的时序边 (BEFORE)
        print("  [Step 4] 建立事件时序边...")
        for i in range(len(l2_memories) - 1):
            curr_l2 = l2_memories[i]
            next_l2 = l2_memories[i + 1]
            
            edge = KGEdge(
                edge_id=f"edge_{edge_id_counter}",
                source_id=f"event_{curr_l2.l2_index}_{vid_id}",
                target_id=f"event_{next_l2.l2_index}_{vid_id}",
                edge_type=KGEdgeType.BEFORE,
                video_id=vid_id,
                timestamp_sec=curr_l2.end_sec
            )
            kg_edges.append(edge)
            edge_id_counter += 1
        
        # 5. 从global_summaries中提取因果边 (CAUSES)
        print("  [Step 5] 提取因果边...")
        for summary in global_summaries:
            # 如果causal_chain包含明确的因果描述，可以进一步解析
            # 简单实现：将窗口内相邻事件标记为潜在因果
            if summary.causal_chain and "->" in summary.causal_chain:
                # 在窗口内的事件之间建立因果边
                for l2_idx in range(summary.l2_start_index, summary.l2_end_index):
                    if l2_idx + 1 <= summary.l2_end_index:
                        edge = KGEdge(
                            edge_id=f"edge_{edge_id_counter}",
                            source_id=f"event_{l2_idx}_{vid_id}",
                            target_id=f"event_{l2_idx + 1}_{vid_id}",
                            edge_type=KGEdgeType.CAUSES,
                            video_id=vid_id,
                            attributes={"causal_chain": summary.causal_chain}
                        )
                        kg_edges.append(edge)
                        edge_id_counter += 1
        
        # 6. 保存知识图谱
        print("  [Step 6] 保存知识图谱...")
        os.makedirs(osp.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            kg_data = {
                "video_id": vid_id,
                "nodes": [node.to_dict() for node in kg_nodes],
                "edges": [edge.to_dict() for edge in kg_edges],
                "statistics": {
                    "total_nodes": len(kg_nodes),
                    "event_nodes": len([n for n in kg_nodes if n.node_type == KGNodeType.EVENT]),
                    "entity_nodes": len([n for n in kg_nodes if n.node_type != KGNodeType.EVENT]),
                    "total_edges": len(kg_edges)
                }
            }
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        return kg_nodes, kg_edges
    
    def _extract_entities_from_l1(self, l1: L1MemoryNode) -> Dict[str, List[str]]:
        """
        从L1节点的caption中提取实体
        
        解析caption中的JSON字段:
        - subjects -> person
        - scene_objects -> object
        - target_objects -> object
        - instruments -> object
        - scene -> place
        - ocr_content -> text
        
        Returns:
            {entity_type: [entity_names]}
        """
        entities = {
            "person": [],
            "object": [],
            "place": [],
            "text": []
        }
        
        try:
            # 尝试解析caption为JSON
            caption = l1.caption
            if isinstance(caption, str):
                # 尝试解析JSON
                try:
                    caption_data = json.loads(caption)
                except json.JSONDecodeError:
                    # 不是JSON格式，跳过
                    return entities
            elif isinstance(caption, dict):
                caption_data = caption
            else:
                return entities
            
            # 提取subjects -> person
            subjects = caption_data.get("subjects", [])
            if isinstance(subjects, list):
                entities["person"].extend([s for s in subjects if s])
            elif isinstance(subjects, str) and subjects:
                entities["person"].append(subjects)
            
            # 提取scene_objects -> object
            scene_objects = caption_data.get("scene_objects", [])
            if isinstance(scene_objects, list):
                entities["object"].extend([o for o in scene_objects if o])
            elif isinstance(scene_objects, str) and scene_objects:
                entities["object"].append(scene_objects)
            
            # 提取target_objects -> object
            target_objects = caption_data.get("target_objects", [])
            if isinstance(target_objects, list):
                entities["object"].extend([o for o in target_objects if o])
            elif isinstance(target_objects, str) and target_objects:
                entities["object"].append(target_objects)
            
            # 提取instruments -> object
            instruments = caption_data.get("instruments", [])
            if isinstance(instruments, list):
                entities["object"].extend([i for i in instruments if i])
            elif isinstance(instruments, str) and instruments:
                entities["object"].append(instruments)
            
            # 提取scene -> place
            scene = caption_data.get("scene", "")
            if scene:
                entities["place"].append(scene)
            
            # 提取ocr_content -> text
            ocr_content = caption_data.get("ocr_content", [])
            if isinstance(ocr_content, list):
                entities["text"].extend([t for t in ocr_content if t])
            elif isinstance(ocr_content, str) and ocr_content:
                entities["text"].append(ocr_content)
            
            # 去重
            for key in entities:
                entities[key] = list(set(entities[key]))
                
        except Exception as e:
            print(f"[实体提取警告] L1[{l1.scene_index}]: {e}")
        
        return entities
    
    def load_l2_memories(self, video_path: str) -> List[L2MemoryNode]:
        """
        从文件加载L2记忆
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            L2MemoryNode列表
        """
        vid_id = osp.splitext(osp.basename(video_path))[0]
        l2_file = osp.join(self.output_dir, vid_id, "semantic_memories_L2.json")
        
        if not osp.exists(l2_file):
            print(f"[警告] L2文件不存在: {l2_file}")
            return []
        
        l2_memories = []
        try:
            with open(l2_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    
                    # 重建L1DetailInL2对象
                    l1_details = []
                    for detail_data in data.get("l1_details", []):
                        l1_detail = L1DetailInL2(
                            scene_index=detail_data.get("scene_index", detail_data.get("l1_scene_index", 0)),
                            start_sec=detail_data.get("start_sec", 0.0),
                            end_sec=detail_data.get("end_sec", 0.0),
                            keyframe_paths=detail_data.get("keyframe_paths", detail_data.get("l1_1fps_frames", [])),
                            keyframe_times=detail_data.get("keyframe_times", detail_data.get("l1_1fps_times", [])),
                            visual_evidence=detail_data.get("visual_evidence", ""),
                            event_fact=detail_data.get("event_fact", "")
                        )
                        l1_details.append(l1_detail)
                    
                    l2_node = L2MemoryNode(
                        video_id=data.get("video_id", vid_id),
                        l2_index=data.get("l2_index", 0),
                        l1_scene_indices=data.get("l1_scene_indices", []),
                        l1_details=l1_details,
                        start_sec=data.get("start_sec", 0.0),
                        end_sec=data.get("end_sec", 0.0),
                        representative_frames=data.get("representative_frames", []),
                        representative_times=data.get("representative_times", []),
                        working_memory=data.get("working_memory"),
                        aggregated_caption=data.get("aggregated_caption"),
                        event_summary=data.get("event_summary"),
                        l3_keyframe_paths=data.get("l3_keyframe_paths", []),
                        l3_keyframe_times=data.get("l3_keyframe_times", [])
                    )
                    l2_memories.append(l2_node)
                    
        except Exception as e:
            print(f"[L2加载错误] {e}")
            import traceback
            traceback.print_exc()
        
        print(f"  加载L2记忆: {len(l2_memories)} 个节点")
        return l2_memories
    
    def generate_all_memories_with_l3(
        self,
        video_path: str,
        use_vlm_decision: bool = False,
        skip_if_exists: bool = True,
        l3_window_size: int = 4,
        l3_stride: int = 2
    ):
        """
        生成完整的多层次记忆 (L1 + L2 + L3)
        
        流程:
        1. 生成L1记忆
        2. 生成L2记忆
        3. 生成L3知识图谱
        
        Args:
            video_path: 视频文件路径
            use_vlm_decision: 是否使用VLM进行L2聚合决策
            skip_if_exists: 如果记忆文件已存在是否跳过
            l3_window_size: L3滑动窗口大小
            l3_stride: L3滑动步长
        """
        # 先生成L1和L2
        self.generate_all_memories(
            video_path=video_path,
            use_vlm_decision=use_vlm_decision,
            skip_if_exists=skip_if_exists
        )
        
        # 再生成L3知识图谱
        self.generate_l3_knowledge_graph(
            video_path=video_path,
            window_size=l3_window_size,
            stride=l3_stride,
            skip_if_exists=skip_if_exists
        )
        
    
    def _save_memories_jsonl(self, memories: list, output_file: str):
        """保存记忆到JSONL文件"""
        os.makedirs(osp.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for mem in memories:
                if hasattr(mem, 'to_dict'):
                    line = json.dumps(mem.to_dict(), ensure_ascii=False)
                else:
                    line = json.dumps(mem, ensure_ascii=False)
                f.write(line + '\n')


# ============================================================================
#                           便捷函数
# ============================================================================

def find_all_videos(video_dir: str, extensions: List[str] = [".mp4", ".avi", ".mkv", ".mov"]) -> List[str]:
    """
    递归查找目录下所有视频文件
    
    Args:
        video_dir: 视频文件夹路径
        extensions: 视频文件扩展名列表
        
    Returns:
        视频文件路径列表
    """
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(osp.join(root, file))
    
    video_files.sort()  # 按文件名排序
    return video_files


def process_single_video(args_tuple):
    """
    处理单个视频的包装函数 (用于并行处理)
    
    Args:
        args_tuple: (video_path, config_path, output_dir, use_vlm, with_l3, l3_window, l3_stride)
    """
    video_path, config_path, output_dir, use_vlm, with_l3, l3_window, l3_stride = args_tuple
    
    vid_id = osp.splitext(osp.basename(video_path))[0]
    
    try:
        print(f"\n{'='*60}")
        print(f"[开始处理] {vid_id}")
        print(f"{'='*60}")
        
        # 每个进程创建独立的生成器实例
        generator = create_memory_generator(
            config_path=config_path,
            output_dir=output_dir
        )
        
        # 生成记忆
        if with_l3:
            generator.generate_all_memories_with_l3(
                video_path=video_path,
                use_vlm_decision=use_vlm,
                skip_if_exists=True,
                l3_window_size=l3_window,
                l3_stride=l3_stride
            )
        else:
            generator.generate_all_memories(
                video_path=video_path,
                use_vlm_decision=use_vlm,
                skip_if_exists=True
            )
        
        print(f"[完成] {vid_id}")
        return (vid_id, True, None)
        
    except Exception as e:
        print(f"[错误] {vid_id}: {e}")
        import traceback
        traceback.print_exc()
        return (vid_id, False, str(e))


def process_video_folder(
    video_dir: str,
    config_path: str,
    output_dir: str,
    use_vlm: bool = False,
    with_l3: bool = False,
    l3_window: int = 4,
    l3_stride: int = 2,
    num_workers: int = 4
):
    """
    并行处理视频文件夹中的所有视频
    
    Args:
        video_dir: 视频文件夹路径
        config_path: 配置文件路径
        output_dir: 输出目录
        use_vlm: 是否使用VLM进行L2聚合决策
        with_l3: 是否生成L3
        l3_window: L3滑动窗口大小
        l3_stride: L3滑动步长
        num_workers: 并行工作进程数
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 查找所有视频文件
    video_files = find_all_videos(video_dir)
    
    if not video_files:
        print(f"[警告] 在 {video_dir} 中未找到视频文件")
        return
    
    print(f"\n{'='*60}")
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"并行工作线程数: {num_workers}")
    print(f"{'='*60}\n")
    
    for i, vf in enumerate(video_files):
        print(f"  [{i+1}] {osp.basename(vf)}")
    print()
    
    # 准备参数
    args_list = [
        (video_path, config_path, output_dir, use_vlm, with_l3, l3_window, l3_stride)
        for video_path in video_files
    ]
    
    # 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理视频"):
            video_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                vid_id = osp.splitext(osp.basename(video_path))[0]
                results.append((vid_id, False, str(e)))
    
    # 统计结果
    success_count = sum(1 for r in results if r[1])
    fail_count = len(results) - success_count
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  失败: {fail_count}/{len(results)}")
    print(f"{'='*60}")
    
    if fail_count > 0:
        print("\n失败的视频:")
        for vid_id, success, error in results:
            if not success:
                print(f"  - {vid_id}: {error}")


def create_memory_generator(
    config_path: str,
    output_dir: str
) -> VideoMemoryGenerator:
    """
    创建VideoMemoryGenerator实例
    
    通过 vLLM API 调用模型，不再加载本地模型
    
    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("[初始化] 使用 vLLM API 模式，不加载本地模型")
    
    # 创建生成器 (使用 API 调用)
    generator = VideoMemoryGenerator(
        config=config,
        output_dir=output_dir
    )
    
    return generator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="视频多层次记忆生成 (vLLM API 模式)")
    parser.add_argument("--video-dir", type=str, help="视频文件夹路径 (递归查找所有视频)")
    parser.add_argument("--video", type=str, help="单个视频文件路径 (与 --video-dir 二选一)")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--use-vlm", action="store_true", help="使用VLM进行L2聚合决策")
    parser.add_argument("--with-l3", action="store_true", help="同时生成L3知识图谱")
    parser.add_argument("--l3-window", type=int, default=4, help="L3滑动窗口大小")
    parser.add_argument("--l3-stride", type=int, default=2, help="L3滑动步长")
    parser.add_argument("--num-workers", type=int, default=4, help="并行处理的工作线程数")
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.video_dir and not args.video:
        parser.error("必须指定 --video-dir 或 --video 参数之一")
    
    if args.video_dir and args.video:
        parser.error("--video-dir 和 --video 参数只能选择其一")
    
    if args.video_dir:
        # 并行处理视频文件夹
        process_video_folder(
            video_dir=args.video_dir,
            config_path=args.config,
            output_dir=args.output,
            use_vlm=args.use_vlm,
            with_l3=args.with_l3,
            l3_window=args.l3_window,
            l3_stride=args.l3_stride,
            num_workers=args.num_workers
        )
    else:
        # 处理单个视频
        generator = create_memory_generator(
            config_path=args.config,
            output_dir=args.output
        )
        
        if args.with_l3:
            generator.generate_all_memories_with_l3(
                video_path=args.video,
                use_vlm_decision=args.use_vlm,
                skip_if_exists=True,
                l3_window_size=args.l3_window,
                l3_stride=args.l3_stride
            )
        else:
            generator.generate_all_memories(
                video_path=args.video,
                use_vlm_decision=args.use_vlm,
                skip_if_exists=True
            )
    
