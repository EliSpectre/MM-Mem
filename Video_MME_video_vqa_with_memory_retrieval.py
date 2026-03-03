"""
视频VQA问答系统 (带记忆检索回退机制)

流程:
1. 首先使用视频直接回答问题
2. 如果选项置信度低于阈值，触发记忆检索:
   - L2 文本检索 (Embedding 粗排 + Rerank 细排)
   - L1 视觉重排 (MLLM 判断场景相关性)
   - 使用筛选后的 L1 场景图片重新回答
3. 记录两次回答结果，统计准确率

依赖:
- openai: vLLM API 调用
- sentence_transformers: Embedding 和 Rerank
- numpy: 数值计算
"""

import json
import os
import os.path as osp
import sys
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from openai import OpenAI

# 尝试导入 sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[警告] sentence_transformers 未安装，将无法使用 Embedding 检索")


# ============================================================================
#                           配置加载
# ============================================================================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
#                           数据结构
# ============================================================================

@dataclass
class L2MemoryEmbedding:
    """L2 记忆嵌入数据结构"""
    l2_index: int
    text_fields: Dict[str, str]
    combined_text: str
    embedding: Optional[np.ndarray] = None
    l1_scene_indices: List[int] = field(default_factory=list)
    start_sec: float = 0.0
    end_sec: float = 0.0
    working_memory: str = ""
    event_summary: str = ""


@dataclass
class L1SceneInfo:
    """L1 场景信息"""
    scene_index: int
    video_id: str
    start_sec: float
    end_sec: float
    keyframe_paths: List[str]
    keyframe_times: List[float]
    event_fact: str = ""
    caption: str = ""


# ============================================================================
#                           工具函数
# ============================================================================

def normalize_answer(answer_str: str) -> str:
    """从答案字符串中提取唯一的字母选项"""
    if not isinstance(answer_str, str):
        return ""
    for char in answer_str.strip():
        if char.isalpha() and char.upper() in 'ABCD':
            return char.upper()
    return ""


def extract_option_confidences(response) -> Dict[str, Any]:
    """从 API 响应的 logprobs 中提取 A/B/C/D 各选项的置信度分数"""
    result = {
        'answer': '',
        'confidence': 0.0,
        'option_scores': {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    }
    
    try:
        raw_text = response.choices[0].message.content.strip()
        result['answer'] = normalize_answer(raw_text)
        
        # 检查是否有 logprobs
        if not hasattr(response.choices[0], 'logprobs') or response.choices[0].logprobs is None:
            return result
        
        logprobs_content = response.choices[0].logprobs.content
        if not logprobs_content:
            return result
        
        # 获取第一个 token 的 logprobs
        first_token_info = logprobs_content[0]
        
        # 获取当前选择的 token 的概率
        if hasattr(first_token_info, 'logprob'):
            result['confidence'] = math.exp(first_token_info.logprob)
        
        # 从 top_logprobs 中提取所有选项的概率
        if hasattr(first_token_info, 'top_logprobs') and first_token_info.top_logprobs:
            for token_info in first_token_info.top_logprobs:
                token_text = token_info.token.strip().upper()
                if token_text and token_text[0] in 'ABCD':
                    option = token_text[0]
                    prob = math.exp(token_info.logprob)
                    if prob > result['option_scores'][option]:
                        result['option_scores'][option] = prob
        
        # 如果答案不在 top_logprobs 中，使用实际输出的概率
        if result['answer'] in 'ABCD' and result['option_scores'][result['answer']] == 0.0:
            result['option_scores'][result['answer']] = result['confidence']
                
    except Exception as e:
        print(f"  [警告] 提取置信度失败: {e}", file=sys.stderr)
    
    return result


def extract_yes_confidence(response) -> float:
    """从 API 响应中提取 Yes 的置信度分数"""
    try:
        logprobs_content = response.choices[0].logprobs.content
        if not logprobs_content:
            return 0.0
        
        first_token_logprobs = logprobs_content[0]
        top_logprobs = first_token_logprobs.top_logprobs
        
        yes_prob = 0.0
        no_prob = 0.0
        
        for token_info in top_logprobs:
            token_text = token_info.token.lower().strip()
            prob = math.exp(token_info.logprob)
            
            if token_text in ['yes', 'y']:
                yes_prob += prob
            elif token_text in ['no', 'n']:
                no_prob += prob
        
        if yes_prob == 0.0 and no_prob == 0.0:
            first_prob = math.exp(first_token_logprobs.logprob)
            first_token = first_token_logprobs.token.lower().strip()
            return first_prob if first_token in ['yes', 'y'] else 0.0
        
        total_prob = yes_prob + no_prob
        return yes_prob / total_prob if total_prob > 0 else yes_prob
        
    except Exception as e:
        print(f"[警告] 提取置信度失败: {e}", file=sys.stderr)
        return 0.0


# ============================================================================
#                           L2 文本检索器
# ============================================================================

class L2TextRetriever:
    """L2 记忆文本检索器 (Embedding + Rerank)"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        rerank_model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda"
    ):
        self.device = device
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name
        
        self.embedding_model = None
        self.rerank_model = None
        self._embedding_loaded = False
        self._rerank_loaded = False
    
    def _load_embedding_model(self):
        """延迟加载 Embedding 模型"""
        if self._embedding_loaded:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence_transformers 未安装")
        
        print(f"[加载] Embedding 模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        self._embedding_loaded = True
        print("[完成] Embedding 模型加载成功")
    
    def _load_rerank_model(self):
        """延迟加载 Rerank 模型"""
        if self._rerank_loaded:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence_transformers 未安装")
        
        print(f"[加载] Rerank 模型: {self.rerank_model_name}")
        self.rerank_model = CrossEncoder(
            self.rerank_model_name,
            device=self.device
        )
        self._rerank_loaded = True
        print("[完成] Rerank 模型加载成功")
    
    def extract_text_from_l2(self, l2_memory: Dict) -> Dict[str, str]:
        """从 L2 记忆中提取文本字段"""
        text_fields = {}
        
        if "event_summary" in l2_memory:
            text_fields["event_summary"] = str(l2_memory["event_summary"])
        
        if "working_memory" in l2_memory:
            text_fields["working_memory"] = str(l2_memory["working_memory"])
        
        if "l1_details" in l2_memory:
            event_facts = []
            for detail in l2_memory["l1_details"]:
                if "event_fact" in detail and detail["event_fact"]:
                    event_facts.append(str(detail["event_fact"]))
            if event_facts:
                text_fields["event_facts"] = " | ".join(event_facts)
        
        return text_fields
    
    def combine_text_fields(self, text_fields: Dict[str, str]) -> str:
        """合并文本字段"""
        parts = []
        field_order = [
            ("working_memory", "Memory"),
            ("event_summary", "Summary"),
            ("event_facts", "Events"),
        ]
        
        for field_key, field_label in field_order:
            if field_key in text_fields and text_fields[field_key]:
                parts.append(f"{field_label}: {text_fields[field_key]}")
        
        return " | ".join(parts) if parts else ""
    
    def load_l2_memories(self, l2_json_path: str) -> List[L2MemoryEmbedding]:
        """从 JSON 文件加载 L2 记忆"""
        print(f"[加载] L2 记忆文件: {l2_json_path}")
        
        memories = []
        with open(l2_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        l2_embeddings = []
        for memory in memories:
            text_fields = self.extract_text_from_l2(memory)
            combined_text = self.combine_text_fields(text_fields)
            
            l1_indices = memory.get("l1_scene_indices", [])
            
            l2_emb = L2MemoryEmbedding(
                l2_index=memory.get("l2_index", 0),
                text_fields=text_fields,
                combined_text=combined_text,
                l1_scene_indices=l1_indices,
                start_sec=memory.get("start_sec", 0.0),
                end_sec=memory.get("end_sec", 0.0),
                working_memory=memory.get("working_memory", ""),
                event_summary=memory.get("event_summary", "")
            )
            l2_embeddings.append(l2_emb)
        
        print(f"[完成] 加载了 {len(l2_embeddings)} 条 L2 记忆")
        return l2_embeddings
    
    def generate_embeddings(
        self,
        l2_memories: List[L2MemoryEmbedding],
        batch_size: int = 32
    ) -> List[L2MemoryEmbedding]:
        """生成 Embedding 向量"""
        self._load_embedding_model()
        
        print(f"[生成] 为 {len(l2_memories)} 条 L2 记忆生成 Embedding...")
        
        texts = [mem.combined_text for mem in l2_memories]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        for mem, emb in zip(l2_memories, embeddings):
            mem.embedding = emb
        
        print(f"[完成] Embedding 生成完成，维度: {embeddings[0].shape}")
        return l2_memories
    
    def search_by_embedding(
        self,
        query: str,
        l2_memories: List[L2MemoryEmbedding],
        top_k: int = 20
    ) -> List[Tuple[int, float, L2MemoryEmbedding]]:
        """使用 Embedding 相似度搜索 (粗排)"""
        self._load_embedding_model()
        
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        scores = []
        for mem in l2_memories:
            if mem.embedding is None:
                continue
            
            similarity = np.dot(query_embedding, mem.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(mem.embedding) + 1e-8
            )
            scores.append((mem.l2_index, similarity, mem))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, L2MemoryEmbedding]],
        top_k: int = 5
    ) -> List[Tuple[int, float, L2MemoryEmbedding]]:
        """使用 Rerank 模型重排序 (细排)"""
        self._load_rerank_model()
        
        if not candidates:
            return []
        
        print(f"[Rerank] 对 {len(candidates)} 个候选进行重排序...")
        
        pairs = [(query, cand[2].combined_text) for cand in candidates]
        rerank_scores = self.rerank_model.predict(pairs)
        
        reranked = []
        for (l2_idx, _, mem), score in zip(candidates, rerank_scores):
            reranked.append((l2_idx, float(score), mem))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        print("[完成] Rerank 完成")
        return reranked[:top_k]
    
    def search(
        self,
        query: str,
        l2_memories: List[L2MemoryEmbedding],
        top_k_embedding: int = 20,
        top_k_rerank: int = 5,
        use_rerank: bool = True
    ) -> List[Tuple[int, float, L2MemoryEmbedding]]:
        """完整的 L2 文本检索流程"""
        # Step 1: Embedding 粗排
        candidates = self.search_by_embedding(query, l2_memories, top_k=top_k_embedding)
        print(f"    [粗排] Embedding 检索返回 {len(candidates)} 个候选")
        
        if not use_rerank or len(candidates) <= top_k_rerank:
            return candidates[:top_k_rerank]
        
        # Step 2: Rerank 细排
        results = self.rerank(query, candidates, top_k=top_k_rerank)
        return results
    
    def save_embeddings(self, l2_memories: List[L2MemoryEmbedding], output_path: str):
        """保存 Embedding 到文件"""
        l2_indices = [mem.l2_index for mem in l2_memories]
        embeddings = np.array([mem.embedding for mem in l2_memories if mem.embedding is not None])
        texts = [mem.combined_text for mem in l2_memories]
        
        np.savez(
            output_path,
            l2_indices=l2_indices,
            embeddings=embeddings,
            texts=texts
        )
        print(f"[保存] Embedding 已保存到: {output_path}")
    
    def load_embeddings(
        self,
        embedding_path: str,
        l2_memories: List[L2MemoryEmbedding]
    ) -> List[L2MemoryEmbedding]:
        """从文件加载 Embedding"""
        data = np.load(embedding_path, allow_pickle=True)
        l2_indices = data["l2_indices"]
        embeddings = data["embeddings"]
        
        idx_to_emb = {int(idx): emb for idx, emb in zip(l2_indices, embeddings)}
        
        for mem in l2_memories:
            if mem.l2_index in idx_to_emb:
                mem.embedding = idx_to_emb[mem.l2_index]
        
        print(f"[加载] 从文件加载了 {len(l2_indices)} 个 Embedding")
        return l2_memories


# ============================================================================
#                           L1 视觉重排器
# ============================================================================

class L1VisualReranker:
    """L1 场景视觉重排器 (使用 MLLM 判断相关性)"""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8002/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen3-VL-2B-Instruct"
    ):
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )
        self.model_name = model_name
        print(f"[L1视觉重排器] 已初始化，模型: {model_name}")
    
    def load_l1_memories(self, l1_json_path: str) -> Dict[int, L1SceneInfo]:
        """加载 L1 记忆，返回 scene_index -> L1SceneInfo 的映射"""
        print(f"[加载] L1 记忆文件: {l1_json_path}")
        
        l1_map = {}
        with open(l1_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    scene_info = L1SceneInfo(
                        scene_index=data.get("scene_index", 0),
                        video_id=data.get("video_id", ""),
                        start_sec=data.get("start_sec", 0.0),
                        end_sec=data.get("end_sec", 0.0),
                        keyframe_paths=data.get("keyframe_paths", []),
                        keyframe_times=data.get("keyframe_times", []),
                        event_fact=data.get("event_fact", ""),
                        caption=data.get("caption", "")
                    )
                    l1_map[scene_info.scene_index] = scene_info
                except json.JSONDecodeError:
                    continue
        
        print(f"[完成] 加载了 {len(l1_map)} 个 L1 场景")
        return l1_map
    
    def get_scene_images(
        self,
        scene_info: L1SceneInfo,
        memory_dir: str,
        max_images: int = 12
    ) -> List[str]:
        """获取场景的图片路径列表"""
        image_paths = []
        memory_dir = osp.abspath(memory_dir)
        
        for rel_path in scene_info.keyframe_paths:
            if rel_path.startswith('/'):
                rel_path = rel_path.lstrip('/')
            full_path = osp.join(memory_dir, rel_path)
            full_path = osp.abspath(full_path)
            if osp.exists(full_path):
                image_paths.append(full_path)
        
        # 限制图片数量
        if len(image_paths) > max_images:
            indices = np.linspace(0, len(image_paths) - 1, max_images, dtype=int)
            image_paths = [image_paths[i] for i in indices]
        
        return image_paths
    
    def judge_scene_relevance(
        self,
        image_paths: List[str],
        question_text: str,
        context_text: str = ""
    ) -> Tuple[bool, float]:
        """判断场景图片是否与问题相关"""
        if not image_paths:
            return False, 0.0
        
        content = []
        
        # 添加图片
        for img_path in image_paths:
            img_path = osp.abspath(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"}
            })
        
        # 添加上下文
        if context_text:
            content.append({
                "type": "text",
                "text": f"Context: {context_text}\n"
            })
        
        # 构建 prompt
        prompt = (
            "Based on the images provided above, determine if these images are relevant "
            "to answering the following question.\n"
            "The images should be KEPT if they contain visual information that could help "
            "answer the question.\n"
            "The images should be DISCARDED if they are NOT relevant to the question.\n\n"
            f"Question: {question_text}\n\n"
            "Should these images be kept to help answer this question?\n"
            "Respond with only 'Yes' or 'No'."
        )
        
        content.append({"type": "text", "text": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=10,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )
            
            raw_answer = response.choices[0].message.content.strip().lower()
            should_keep = "yes" in raw_answer
            confidence = extract_yes_confidence(response)
            
            return should_keep, confidence
            
        except Exception as e:
            print(f"[API错误] 判断场景相关性失败: {e}")
            return True, 0.5
    
    def rerank_scenes(
        self,
        l1_scenes: List[L1SceneInfo],
        question_text: str,
        memory_dir: str,
        top_k: int = 3,
        max_images_per_scene: int = 12,
        verbose: bool = True
    ) -> List[Tuple[L1SceneInfo, float]]:
        """对 L1 场景进行视觉重排"""
        if not l1_scenes:
            return []
        
        if verbose:
            print(f"\n[L1视觉重排] 对 {len(l1_scenes)} 个场景进行重排...")
        
        scene_scores = []
        
        for i, scene in enumerate(l1_scenes):
            if verbose:
                print(f"    [{i+1}/{len(l1_scenes)}] 处理场景 {scene.scene_index}...", end=" ")
            
            image_paths = self.get_scene_images(scene, memory_dir, max_images_per_scene)
            
            if not image_paths:
                if verbose:
                    print("无图片，跳过")
                continue
            
            _, confidence = self.judge_scene_relevance(
                image_paths=image_paths,
                question_text=question_text,
                context_text=scene.event_fact
            )
            
            scene_scores.append((scene, confidence))
            
            if verbose:
                print(f"置信度: {confidence:.3f}")
        
        scene_scores.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"[完成] 视觉重排完成，保留 top-{top_k} 场景")
        
        return scene_scores[:top_k]


# ============================================================================
#                           VQA 问答器
# ============================================================================

class VQAAnswerer:
    """VQA 问答器"""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8002/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen3-VL-2B-Instruct"
    ):
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )
        self.model_name = model_name
    
    def answer_with_video(
        self,
        video_path: str,
        question_text: str,
        options: List[str]
    ) -> Dict[str, Any]:
        """使用视频直接回答问题"""
        video_path = osp.abspath(video_path)
        
        options_str = "\n".join(options)
        prompt = (
            "Select the best answer to the following multiple-choice question based on the video.\n"
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {question_text}\n"
            f"Possible answer choices:\n{options_str}\n"
            "The best answer is:"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": f"file://{video_path}"}
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=2,
                temperature=0,
                logprobs=True,
                top_logprobs=10,
            )
            
            confidence_info = extract_option_confidences(response)
            raw_answer = response.choices[0].message.content.strip()
            
            return {
                'raw_answer': raw_answer,
                'normalized_answer': confidence_info['answer'],
                'confidence': confidence_info['confidence'],
                'option_scores': confidence_info['option_scores']
            }
            
        except Exception as e:
            print(f"[API错误] 视频问答失败: {e}")
            return {
                'raw_answer': '',
                'normalized_answer': '',
                'confidence': 0.0,
                'option_scores': {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0},
                'error': str(e)
            }
    
    def answer_with_scenes(
        self,
        question_text: str,
        options: List[str],
        scene_image_paths: List[List[str]],
        video_path: str = None,
        include_video: bool = False
    ) -> Dict[str, Any]:
        """使用场景图片进行 VQA 问答"""
        content = []
        
        # 可选添加视频
        if include_video and video_path:
            video_path = osp.abspath(video_path)
            if osp.exists(video_path):
                content.append({
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path}"}
                })
        
        # 添加所有场景的图片
        total_images = 0
        for scene_paths in scene_image_paths:
            for img_path in scene_paths:
                img_path = osp.abspath(img_path)
                if osp.exists(img_path):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{img_path}"}
                    })
                    total_images += 1
        
        # 构建 Prompt
        options_str = "\n".join(options)
        prompt = (
            "Select the best answer to the following multiple-choice question "
            "based on the provided images.\n"
            "These images are key frames extracted from a video that are relevant "
            "to the question.\n"
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {question_text}\n"
            f"Possible answer choices:\n{options_str}\n"
            "The best answer is:"
        )
        
        content.append({"type": "text", "text": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=2,
                temperature=0,
                logprobs=True,
                top_logprobs=10,
            )
            
            raw_answer = response.choices[0].message.content.strip()
            confidence_info = extract_option_confidences(response)
            
            return {
                'raw_answer': raw_answer,
                'normalized_answer': confidence_info['answer'],
                'confidence': confidence_info['confidence'],
                'option_scores': confidence_info['option_scores'],
                'total_images_used': total_images,
                'scene_count': len(scene_image_paths)
            }
            
        except Exception as e:
            print(f"[API错误] 场景问答失败: {e}")
            return {
                'raw_answer': '',
                'normalized_answer': '',
                'confidence': 0.0,
                'option_scores': {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0},
                'total_images_used': total_images,
                'scene_count': len(scene_image_paths),
                'error': str(e)
            }


# ============================================================================
#                           VQA 记忆检索系统
# ============================================================================

class VideoVQAWithMemoryRetrieval:
    """视频 VQA 问答系统 (带记忆检索回退)"""
    
    def __init__(self, config: Dict):
        """
        初始化系统
        
        Args:
            config: vqa_memory_retrieval 配置字典
        """
        self.config = config
        
        # 路径配置
        self.video_dir = config.get("video_dir", "")
        self.memory_dir = osp.abspath(config.get("memory_dir", ""))
        
        # API 配置
        api_base_url = config.get("api_base_url", "http://localhost:8002/v1")
        api_key = config.get("api_key", "EMPTY")
        model_name = config.get("model_name", "Qwen3-VL-2B-Instruct")
        
        # 检索参数
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.top_k_l2_embedding = config.get("top_k_l2_embedding", 20)
        self.top_k_l2_rerank = config.get("top_k_l2_rerank", 5)
        self.top_k_l1_visual = config.get("top_k_l1_visual", 3)
        self.max_images_per_scene = config.get("max_images_per_scene", 12)
        
        # 功能开关
        self.use_rerank = config.get("use_rerank", True)
        self.use_visual_rerank = config.get("use_visual_rerank", True)
        self.include_video_in_retrieval = config.get("include_video_in_retrieval", False)
        
        # Embedding 配置
        embedding_model_name = config.get("embedding_model_name", "BAAI/bge-large-en-v1.5")
        rerank_model_name = config.get("rerank_model_name", "BAAI/bge-reranker-v2-m3")
        embedding_device = config.get("embedding_device", "cuda")
        
        # 初始化组件
        self.l2_retriever = L2TextRetriever(
            embedding_model_name=embedding_model_name,
            rerank_model_name=rerank_model_name,
            device=embedding_device
        )
        
        self.l1_reranker = L1VisualReranker(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model_name
        )
        
        self.vqa_answerer = VQAAnswerer(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model_name
        )
        
        # 缓存
        self._l2_cache: Dict[str, List[L2MemoryEmbedding]] = {}
        self._l1_cache: Dict[str, Dict[int, L1SceneInfo]] = {}
        self._video_cache: Dict[str, str] = {}
        
        print(f"[VideoVQAWithMemoryRetrieval] 初始化完成")
        print(f"    置信度阈值: {self.confidence_threshold}")
        print(f"    L2 Embedding top-k: {self.top_k_l2_embedding}")
        print(f"    L2 Rerank top-k: {self.top_k_l2_rerank}")
        print(f"    L1 视觉重排 top-k: {self.top_k_l1_visual}")
    
    def _scan_videos(self):
        """扫描视频目录，建立视频路径缓存"""
        if self._video_cache:
            return
        
        print(f"\n[扫描] 视频目录: {self.video_dir}")
        for root, dirs, files in os.walk(self.video_dir):
            for f in files:
                if f.lower().endswith('.mp4'):
                    video_id = osp.splitext(f)[0]
                    self._video_cache[video_id] = osp.join(root, f)
        print(f"[完成] 找到 {len(self._video_cache)} 个视频文件")
    
    def _load_video_memories(self, video_id: str):
        """加载指定视频的记忆"""
        video_dir = osp.join(self.memory_dir, video_id)
        
        # 加载 L2 记忆
        if video_id not in self._l2_cache:
            l2_path = osp.join(video_dir, "semantic_memories_L2.json")
            embedding_cache_path = osp.join(video_dir, "l2_embeddings.npz")
            
            if osp.exists(l2_path):
                l2_memories = self.l2_retriever.load_l2_memories(l2_path)
                
                # 尝试加载或生成 Embedding
                if osp.exists(embedding_cache_path):
                    l2_memories = self.l2_retriever.load_embeddings(embedding_cache_path, l2_memories)
                else:
                    l2_memories = self.l2_retriever.generate_embeddings(l2_memories)
                    self.l2_retriever.save_embeddings(l2_memories, embedding_cache_path)
                
                self._l2_cache[video_id] = l2_memories
            else:
                print(f"[警告] L2 记忆文件不存在: {l2_path}")
                self._l2_cache[video_id] = []
        
        # 加载 L1 记忆
        if video_id not in self._l1_cache:
            l1_path = osp.join(video_dir, "episodic_memories_L1.json")
            
            if osp.exists(l1_path):
                self._l1_cache[video_id] = self.l1_reranker.load_l1_memories(l1_path)
            else:
                print(f"[警告] L1 记忆文件不存在: {l1_path}")
                self._l1_cache[video_id] = {}
    
    def _retrieve_and_answer(
        self,
        video_id: str,
        question_text: str,
        options: List[str],
        video_path: str = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """执行记忆检索并回答问题"""
        result = {
            'retrieval_triggered': True,
            'l2_retrieval': [],
            'l1_visual_rerank': [],
            'final_scenes': [],
            'answer': None
        }
        
        # 加载记忆
        self._load_video_memories(video_id)
        
        l2_memories = self._l2_cache.get(video_id, [])
        l1_map = self._l1_cache.get(video_id, {})
        
        if not l2_memories:
            print(f"[警告] 视频 {video_id} 没有 L2 记忆，无法执行检索")
            return result
        
        # Step 1: L2 文本检索
        if verbose:
            print(f"\n[Step 1] L2 文本检索 (query: {question_text[:50]}...)")
        
        l2_results = self.l2_retriever.search(
            query=question_text,
            l2_memories=l2_memories,
            top_k_embedding=self.top_k_l2_embedding,
            top_k_rerank=self.top_k_l2_rerank,
            use_rerank=self.use_rerank
        )
        
        result['l2_retrieval'] = [
            {'l2_index': idx, 'score': float(score)}
            for idx, score, mem in l2_results
        ]
        
        if verbose:
            print(f"    返回 {len(l2_results)} 个 L2 节点")
            for rank, (idx, score, mem) in enumerate(l2_results, 1):
                summary = mem.event_summary[:50] if mem.event_summary else "N/A"
                print(f"        [{rank}] L2-{idx} (score={score:.4f}): {summary}...")
        
        # Step 2: 收集关联的 L1 场景
        all_l1_scenes = []
        for _, _, l2_mem in l2_results:
            for scene_idx in l2_mem.l1_scene_indices:
                if scene_idx in l1_map:
                    all_l1_scenes.append(l1_map[scene_idx])
        
        # 去重
        seen_indices = set()
        unique_l1_scenes = []
        for scene in all_l1_scenes:
            if scene.scene_index not in seen_indices:
                seen_indices.add(scene.scene_index)
                unique_l1_scenes.append(scene)
        
        if verbose:
            print(f"\n[Step 2] 收集到 {len(unique_l1_scenes)} 个关联的 L1 场景")
        
        # Step 3: L1 视觉重排
        if self.use_visual_rerank and unique_l1_scenes:
            if verbose:
                print(f"\n[Step 3] L1 视觉重排")
            
            reranked_scenes = self.l1_reranker.rerank_scenes(
                l1_scenes=unique_l1_scenes,
                question_text=question_text,
                memory_dir=self.memory_dir,
                top_k=self.top_k_l1_visual,
                max_images_per_scene=self.max_images_per_scene,
                verbose=verbose
            )
            
            result['l1_visual_rerank'] = [
                {'scene_index': scene.scene_index, 'confidence': float(conf)}
                for scene, conf in reranked_scenes
            ]
            
            final_scenes = [scene for scene, _ in reranked_scenes]
        else:
            final_scenes = unique_l1_scenes[:self.top_k_l1_visual]
        
        result['final_scenes'] = [scene.scene_index for scene in final_scenes]
        
        if verbose:
            print(f"\n[Step 4] 使用 {len(final_scenes)} 个场景进行 VQA 问答")
        
        # Step 4: VQA 问答
        scene_image_paths = []
        for scene in final_scenes:
            image_paths = self.l1_reranker.get_scene_images(
                scene, self.memory_dir, self.max_images_per_scene
            )
            if image_paths:
                scene_image_paths.append(image_paths)
        
        if not scene_image_paths:
            print("[警告] 没有可用的场景图片")
            return result
        
        answer_result = self.vqa_answerer.answer_with_scenes(
            question_text=question_text,
            options=options,
            scene_image_paths=scene_image_paths,
            video_path=video_path,
            include_video=self.include_video_in_retrieval
        )
        
        result['answer'] = answer_result
        
        return result
    
    def process_question(
        self,
        video_id: str,
        question_text: str,
        options: List[str],
        ground_truth: str = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        处理单个问题
        
        流程:
        1. 首先使用视频直接回答
        2. 如果置信度低于阈值，触发记忆检索并重新回答
        """
        self._scan_videos()
        
        result = {
            'video_id': video_id,
            'question': question_text,
            'options': options,
            'ground_truth': ground_truth,
            'ground_truth_normalized': normalize_answer(ground_truth) if ground_truth else "",
            
            # 第一次回答 (视频)
            'first_answer': None,
            
            # 记忆检索回退
            'retrieval_triggered': False,
            'retrieval_reason': None,
            'retrieval_result': None,
            
            # 第二次回答 (记忆检索)
            'second_answer': None,
            
            # 最终结果
            'final_answer': None,
            'final_normalized_answer': None,
            'final_confidence': None,
            'final_is_correct': None,
            
            'error': None
        }
        
        # 查找视频
        video_path = self._video_cache.get(video_id)
        if not video_path:
            result['error'] = f"未找到视频文件: {video_id}"
            print(f"  ❌ 错误: {result['error']}")
            return result
        
        # ========== 第一次回答：使用视频 ==========
        if verbose:
            print(f"\n[第一次回答] 使用视频直接回答...")
        
        first_result = self.vqa_answerer.answer_with_video(
            video_path=video_path,
            question_text=question_text,
            options=options
        )
        
        result['first_answer'] = first_result
        
        first_normalized = first_result.get('normalized_answer', '')
        first_confidence = first_result.get('confidence', 0.0)
        first_is_correct = first_normalized == result['ground_truth_normalized'] if ground_truth else None
        
        if verbose:
            status = "✅" if first_is_correct else "❌" if first_is_correct is not None else "?"
            scores_str = ", ".join([f"{k}:{v:.3f}" for k, v in first_result.get('option_scores', {}).items()])
            print(f"  📹 视频回答: {status} {first_normalized} (置信度: {first_confidence:.3f})")
            print(f"     各选项得分: [{scores_str}]")
        
        # 默认最终答案为第一次回答
        result['final_answer'] = first_result.get('raw_answer', '')
        result['final_normalized_answer'] = first_normalized
        result['final_confidence'] = first_confidence
        result['final_is_correct'] = first_is_correct
        
        # ========== 检查是否需要记忆检索回退 ==========
        if first_confidence < self.confidence_threshold:
            result['retrieval_triggered'] = True
            result['retrieval_reason'] = f"置信度 {first_confidence:.3f} < 阈值 {self.confidence_threshold}"
            
            if verbose:
                print(f"\n  ⚠️  触发记忆检索: {result['retrieval_reason']}")
            
            # 执行记忆检索
            retrieval_result = self._retrieve_and_answer(
                video_id=video_id,
                question_text=question_text,
                options=options,
                video_path=video_path,
                verbose=verbose
            )
            
            result['retrieval_result'] = {
                'l2_retrieval': retrieval_result.get('l2_retrieval', []),
                'l1_visual_rerank': retrieval_result.get('l1_visual_rerank', []),
                'final_scenes': retrieval_result.get('final_scenes', [])
            }
            
            # 第二次回答
            if retrieval_result.get('answer'):
                second_answer = retrieval_result['answer']
                result['second_answer'] = second_answer
                
                second_normalized = second_answer.get('normalized_answer', '')
                second_confidence = second_answer.get('confidence', 0.0)
                second_is_correct = second_normalized == result['ground_truth_normalized'] if ground_truth else None
                
                if verbose:
                    status = "✅" if second_is_correct else "❌" if second_is_correct is not None else "?"
                    scores_str = ", ".join([f"{k}:{v:.3f}" for k, v in second_answer.get('option_scores', {}).items()])
                    print(f"\n  🖼️  记忆检索回答: {status} {second_normalized} (置信度: {second_confidence:.3f})")
                    print(f"     各选项得分: [{scores_str}]")
                    print(f"     使用图片数: {second_answer.get('total_images_used', 0)}")
                
                # 更新最终答案
                result['final_answer'] = second_answer.get('raw_answer', '')
                result['final_normalized_answer'] = second_normalized
                result['final_confidence'] = second_confidence
                result['final_is_correct'] = second_is_correct
                
                # 统计改进/退步
                if verbose and ground_truth:
                    if second_is_correct and not first_is_correct:
                        print(f"  📈 改进: 从错误变为正确!")
                    elif first_is_correct and not second_is_correct:
                        print(f"  📉 退步: 从正确变为错误!")
            else:
                if verbose:
                    print(f"  ⚠️  记忆检索无结果，使用第一次回答")
        
        # 打印最终结果
        if verbose and ground_truth:
            final_status = "✅ 最终正确" if result['final_is_correct'] else "❌ 最终错误"
            print(f"\n  🎯 {final_status} | 答案: {result['final_normalized_answer']} | 正确答案: {result['ground_truth_normalized']}")
        
        return result
    
    def process_batch(
        self,
        questions: List[Dict],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """批量处理问题"""
        all_results = []
        
        # 统计数据
        correct_by_duration = {'short': 0, 'medium': 0, 'long': 0, 'unknown': 0}
        total_by_duration = {'short': 0, 'medium': 0, 'long': 0, 'unknown': 0}
        
        # 回退统计
        retrieval_stats = {
            'triggered': 0,
            'success': 0,
            'improved': 0,
            'degraded': 0,
        }
        
        print(f"\n开始处理 {len(questions)} 个问题...")
        print("=" * 80)
        
        for i, q_data in enumerate(questions):
            question_id = q_data.get('question_id', f'unknown-{i}')
            video_id = q_data.get('videoID') or q_data.get('video_id') or q_data.get('videoId')
            duration = q_data.get('duration', 'unknown').lower()
            question_text = q_data.get('question', '')
            options = q_data.get('options', [])
            ground_truth = q_data.get('answer', '')
            
            if duration not in total_by_duration:
                duration = 'unknown'
            total_by_duration[duration] += 1
            
            print(f"\n[{i+1}/{len(questions)}] 问题ID: {question_id}, 视频ID: {video_id}")
            print(f"问题: {question_text[:80]}..." if len(question_text) > 80 else f"问题: {question_text}")
            
            # 处理问题
            result = self.process_question(
                video_id=video_id,
                question_text=question_text,
                options=options,
                ground_truth=ground_truth,
                verbose=verbose
            )
            
            # 添加元信息
            result['question_id'] = question_id
            result['duration'] = duration
            result['domain'] = q_data.get('domain')
            result['sub_category'] = q_data.get('sub_category')
            result['task_type'] = q_data.get('task_type')
            
            all_results.append(result)
            
            # 更新统计
            if result['final_is_correct']:
                correct_by_duration[duration] += 1
            
            if result['retrieval_triggered']:
                retrieval_stats['triggered'] += 1
                if result['final_is_correct']:
                    retrieval_stats['success'] += 1
                
                first_correct = result['first_answer'].get('normalized_answer') == result['ground_truth_normalized'] if result['first_answer'] else False
                second_correct = result['final_is_correct']
                
                if second_correct and not first_correct:
                    retrieval_stats['improved'] += 1
                elif first_correct and not second_correct:
                    retrieval_stats['degraded'] += 1
        
        # 构建统计结果
        total_questions = sum(total_by_duration.values())
        total_correct = sum(correct_by_duration.values())
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        statistics = {
            'total_questions': total_questions,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy,
            'accuracy_by_duration': {
                duration: {
                    'total': total_by_duration[duration],
                    'correct': correct_by_duration[duration],
                    'accuracy': (correct_by_duration[duration] / total_by_duration[duration] * 100)
                               if total_by_duration[duration] > 0 else 0
                }
                for duration in ['short', 'medium', 'long', 'unknown']
            },
            'retrieval_stats': retrieval_stats
        }
        
        return {
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'top_k_l2_embedding': self.top_k_l2_embedding,
                'top_k_l2_rerank': self.top_k_l2_rerank,
                'top_k_l1_visual': self.top_k_l1_visual,
                'use_rerank': self.use_rerank,
                'use_visual_rerank': self.use_visual_rerank,
                'include_video_in_retrieval': self.include_video_in_retrieval,
            },
            'statistics': statistics,
            'results': all_results
        }


# ============================================================================
#                           主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="视频VQA问答系统 (带记忆检索回退)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/memory_generator_config.json",
        help="配置文件路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入问题JSON文件路径 (覆盖配置文件)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果JSON文件路径 (覆盖配置文件)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细信息"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    full_config = load_config(args.config)
    config = full_config.get("vqa_memory_retrieval", {})
    
    if not config:
        print("错误: 配置文件中没有 vqa_memory_retrieval 配置")
        return
    
    # 覆盖路径配置
    input_path = args.input or config.get("input_json_path", "")
    output_path = args.output or config.get("output_json_path", "")
    
    # 加载问题
    print(f"\n加载问题数据: {input_path}")
    if not osp.exists(input_path):
        print(f"错误: 问题文件不存在: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"已加载 {len(questions)} 个问题")
    
    # 初始化系统
    system = VideoVQAWithMemoryRetrieval(config)
    
    # 批量处理
    output_data = system.process_batch(questions, verbose=args.verbose)
    
    # 打印统计结果
    stats = output_data['statistics']
    print("\n" + "=" * 80)
    print("最终统计结果")
    print("=" * 80)
    
    print(f"\n总问题数: {stats['total_questions']}")
    print(f"正确回答数: {stats['total_correct']}")
    print(f"总准确率: {stats['overall_accuracy']:.2f}%")
    
    print("\n--- 按时长分类准确率 ---")
    for duration in ['short', 'medium', 'long', 'unknown']:
        d_stats = stats['accuracy_by_duration'][duration]
        if d_stats['total'] > 0:
            print(f"  {duration.capitalize():8s}: {d_stats['accuracy']:6.2f}% ({d_stats['correct']}/{d_stats['total']})")
    
    print("\n--- 记忆检索回退统计 ---")
    r_stats = stats['retrieval_stats']
    print(f"  触发回退次数: {r_stats['triggered']}")
    if r_stats['triggered'] > 0:
        print(f"  回退后正确: {r_stats['success']} ({r_stats['success']/r_stats['triggered']*100:.1f}%)")
        print(f"  从错变对(改进): {r_stats['improved']}")
        print(f"  从对变错(退步): {r_stats['degraded']}")
    
    # 保存结果
    print(f"\n保存结果到: {output_path}")
    
    def to_serializable(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=to_serializable)
        print("保存成功!")
    except Exception as e:
        print(f"保存失败: {e}")
    
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
