"""
记忆检索与VQA问答系统

流程:
1. L2 文本检索 (粗排 + 细排): 使用 Embedding + Rerank 选出 top_k 个 L2 节点
2. L1 视觉重排: 对选中的 L2 节点对应的 L1 场景，用 MLLM 判断 Yes 置信度进行重排
3. VQA 问答: 用最终选定的场景图片进行问答

依赖:
- sentence_transformers: 用于 Embedding 和 Rerank
- openai: 用于 vLLM API 调用
"""

import json
import os
import os.path as osp
import math
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


@dataclass
class RetrievalResult:
    """检索结果"""
    l2_index: int
    l2_score: float  # L2 检索分数
    l1_scenes: List[L1SceneInfo]  # 关联的 L1 场景
    visual_score: float = 0.0  # 视觉重排分数
    final_score: float = 0.0  # 最终分数


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
        
        # 提取主要字段
        if "event_summary" in l2_memory:
            text_fields["event_summary"] = str(l2_memory["event_summary"])
        
        if "working_memory" in l2_memory:
            text_fields["working_memory"] = str(l2_memory["working_memory"])
        
        # 从 l1_details 中提取信息
        if "l1_details" in l2_memory:
            event_facts = []
            visual_evidences = []
            for detail in l2_memory["l1_details"]:
                if "event_fact" in detail and detail["event_fact"]:
                    event_facts.append(str(detail["event_fact"]))
                if "visual_evidence" in detail and detail["visual_evidence"]:
                    visual_evidences.append(str(detail["visual_evidence"]))
            
            if event_facts:
                text_fields["event_facts"] = " | ".join(event_facts)
            if visual_evidences:
                text_fields["visual_evidences"] = " | ".join(visual_evidences)
        
        return text_fields
    
    def combine_text_fields(self, text_fields: Dict[str, str]) -> str:
        """合并文本字段"""
        parts = []
        field_order = [
            ("working_memory", "Memory"),
            ("event_summary", "Summary"),
            ("event_facts", "Events"),
            ######   !!!!! 这行注释掉，就不再读取这个isual_evidences进行embedding的检索啦!!!!!   ########
            # ("visual_evidences", "Visual"),
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
            
            # 获取关联的 L1 场景索引
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
            
            # 余弦相似度
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
        print(f"\n[L2检索] Query: {query[:80]}...")
        
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
#                           L1 视觉重排器 (MLLM)
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
        
        # 确保 memory_dir 是绝对路径
        memory_dir = osp.abspath(memory_dir)
        
        for rel_path in scene_info.keyframe_paths:
            # 如果 rel_path 是绝对路径（以 / 开头），则去掉开头的 /
            if rel_path.startswith('/'):
                rel_path = rel_path.lstrip('/')
            full_path = osp.join(memory_dir, rel_path)
            full_path = osp.abspath(full_path)  # 确保是绝对路径
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
        """
        判断场景图片是否与问题相关
        
        Returns:
            (should_keep, confidence): 是否保留, Yes 置信度
        """
        if not image_paths:
            return False, 0.0
        
        content = []
        
        # 添加图片
        for img_path in image_paths:
            # 确保图片路径是绝对路径
            img_path = osp.abspath(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"}
            })
        
        # 添加上下文（如果有）
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
            confidence = self._extract_yes_confidence(response)
            
            return should_keep, confidence
            
        except Exception as e:
            print(f"[API错误] 判断场景相关性失败: {e}")
            return True, 0.5  # 默认保留，中等置信度
    
    def _extract_yes_confidence(self, response) -> float:
        """从 API 响应中提取 Yes 的置信度"""
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
            print(f"[警告] 提取置信度失败: {e}")
            return 0.0
    
    def rerank_scenes(
        self,
        l1_scenes: List[L1SceneInfo],
        question_text: str,
        memory_dir: str,
        top_k: int = 3,
        max_images_per_scene: int = 12,
        verbose: bool = True
    ) -> List[Tuple[L1SceneInfo, float]]:
        """
        对 L1 场景进行视觉重排
        
        Args:
            l1_scenes: L1 场景列表
            question_text: 问题文本
            memory_dir: 记忆文件根目录
            top_k: 返回前 k 个场景
            max_images_per_scene: 每个场景最多使用的图片数
            verbose: 是否打印详细信息
            
        Returns:
            按置信度排序的 (L1SceneInfo, confidence) 列表
        """
        if not l1_scenes:
            return []
        
        if verbose:
            print(f"\n[L1视觉重排] 对 {len(l1_scenes)} 个场景进行重排...")
        
        scene_scores = []
        
        for i, scene in enumerate(l1_scenes):
            if verbose:
                print(f"    [{i+1}/{len(l1_scenes)}] 处理场景 {scene.scene_index}...", end=" ")
            
            # 获取场景图片
            image_paths = self.get_scene_images(scene, memory_dir, max_images_per_scene)
            
            if not image_paths:
                if verbose:
                    print("无图片，跳过")
                continue
            
            # 判断相关性
            _, confidence = self.judge_scene_relevance(
                image_paths=image_paths,
                question_text=question_text,
                context_text=scene.event_fact
            )
            
            scene_scores.append((scene, confidence))
            
            if verbose:
                print(f"置信度: {confidence:.3f}")
        
        # 按置信度降序排序
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
        model_name: str = "Qwen2.5-VL-7B-Instruct"
    ):
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )
        self.model_name = model_name
    
    def answer_with_scenes(
        self,
        question_text: str,
        options: List[str],
        scene_image_paths: List[List[str]],
        video_path: str = None,
        include_video: bool = False
    ) -> Dict[str, Any]:
        """
        使用场景图片进行 VQA 问答
        
        Args:
            question_text: 问题文本
            options: 选项列表 ["A. xxx", "B. xxx", ...]
            scene_image_paths: 场景图片路径列表的列表
            video_path: 视频路径（可选）
            include_video: 是否同时输入视频
            
        Returns:
            回答结果字典
        """
        content = []
        
        # 1. 可选添加视频
        if include_video and video_path:
            video_path = osp.abspath(video_path)  # 确保是绝对路径
            if osp.exists(video_path):
                content.append({
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path}"}
                })
        
        # 2. 添加所有场景的图片
        total_images = 0
        for scene_paths in scene_image_paths:
            for img_path in scene_paths:
                # 确保图片路径是绝对路径
                img_path = osp.abspath(img_path)
                if osp.exists(img_path):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{img_path}"}
                    })
                    total_images += 1
        
        # 3. 构建 Prompt
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
        
        # 4. 调用 API
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
            confidence_info = self._extract_option_confidences(response)
            
            return {
                'raw_answer': raw_answer,
                'normalized_answer': confidence_info['answer'],
                'confidence': confidence_info['confidence'],
                'option_scores': confidence_info['option_scores'],
                'total_images_used': total_images,
                'scene_count': len(scene_image_paths)
            }
            
        except Exception as e:
            print(f"[API错误] VQA 问答失败: {e}")
            return {
                'raw_answer': '',
                'normalized_answer': '',
                'confidence': 0.0,
                'option_scores': {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0},
                'total_images_used': total_images,
                'scene_count': len(scene_image_paths),
                'error': str(e)
            }
    
    def _extract_option_confidences(self, response) -> Dict[str, Any]:
        """从 API 响应中提取选项置信度"""
        result = {
            'answer': '',
            'confidence': 0.0,
            'option_scores': {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        }
        
        try:
            raw_text = response.choices[0].message.content.strip()
            
            # 提取答案字母
            for char in raw_text:
                if char.upper() in 'ABCD':
                    result['answer'] = char.upper()
                    break
            
            # 检查 logprobs
            if not hasattr(response.choices[0], 'logprobs') or response.choices[0].logprobs is None:
                return result
            
            logprobs_content = response.choices[0].logprobs.content
            if not logprobs_content:
                return result
            
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
            print(f"[警告] 提取置信度失败: {e}")
        
        return result


# ============================================================================
#                           记忆检索 VQA 系统
# ============================================================================

class MemoryRetrievalVQA:
    """记忆检索与 VQA 问答系统"""
    
    def __init__(
        self,
        memory_dir: str,
        api_base_url: str = "http://localhost:8002/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen2.5-VL-7B-Instruct",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        rerank_model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda"
    ):
        """
        初始化记忆检索 VQA 系统
        
        Args:
            memory_dir: 记忆文件根目录（包含各视频 ID 子目录）
            api_base_url: vLLM API 地址
            api_key: API Key
            model_name: MLLM 模型名称
            embedding_model_name: Embedding 模型名称
            rerank_model_name: Rerank 模型名称
            device: 运行设备
        """
        # 确保 memory_dir 是绝对路径
        self.memory_dir = osp.abspath(memory_dir)
        print(f"[MemoryRetrievalVQA] memory_dir 使用绝对路径: {self.memory_dir}")
        
        # L2 文本检索器
        self.l2_retriever = L2TextRetriever(
            embedding_model_name=embedding_model_name,
            rerank_model_name=rerank_model_name,
            device=device
        )
        
        # L1 视觉重排器
        self.l1_reranker = L1VisualReranker(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model_name
        )
        
        # VQA 问答器
        self.vqa_answerer = VQAAnswerer(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model_name
        )
        
        # 缓存
        self._l2_cache: Dict[str, List[L2MemoryEmbedding]] = {}
        self._l1_cache: Dict[str, Dict[int, L1SceneInfo]] = {}
    
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
    
    def retrieve_and_answer(
        self,
        video_id: str,
        question_text: str,
        options: List[str],
        top_k_l2: int = 5,
        top_k_l1: int = 3,
        top_k_embedding: int = 20,
        use_rerank: bool = True,
        use_visual_rerank: bool = True,
        max_images_per_scene: int = 12,
        video_path: str = None,
        include_video: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        检索记忆并回答问题
        
        流程:
        1. L2 文本检索 (粗排 + 细排) -> top_k_l2 个 L2 节点
        2. L1 视觉重排 -> top_k_l1 个 L1 场景
        3. VQA 问答
        
        Args:
            video_id: 视频 ID
            question_text: 问题文本
            options: 选项列表
            top_k_l2: L2 检索返回数量
            top_k_l1: L1 视觉重排返回数量
            top_k_embedding: Embedding 粗排数量
            use_rerank: 是否使用 Rerank 细排
            use_visual_rerank: 是否使用 L1 视觉重排
            max_images_per_scene: 每个场景最多使用的图片数
            video_path: 视频路径（可选）
            include_video: VQA 时是否同时输入视频
            verbose: 是否打印详细信息
            
        Returns:
            回答结果字典
        """
        result = {
            'video_id': video_id,
            'question': question_text,
            'options': options,
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
            print(f"[错误] 视频 {video_id} 没有 L2 记忆")
            return result
        
        # ====== Step 1: L2 文本检索 ======
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Step 1] L2 文本检索")
            print(f"{'='*60}")
        
        l2_results = self.l2_retriever.search(
            query=question_text,
            l2_memories=l2_memories,
            top_k_embedding=top_k_embedding,
            top_k_rerank=top_k_l2,
            use_rerank=use_rerank
        )
        
        result['l2_retrieval'] = [
            {'l2_index': idx, 'score': score, 'text': mem.combined_text[:200]}
            for idx, score, mem in l2_results
        ]
        
        if verbose:
            print(f"\n[L2检索结果] 返回 {len(l2_results)} 个 L2 节点:")
            for rank, (idx, score, mem) in enumerate(l2_results, 1):
                print(f"    [{rank}] L2-{idx} (score={score:.4f}): {mem.event_summary[:50] if mem.event_summary else 'N/A'}...")
        
        # ====== Step 2: 收集关联的 L1 场景 ======
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
            print(f"\n[关联L1场景] 共 {len(unique_l1_scenes)} 个唯一场景")
        
        # ====== Step 3: L1 视觉重排 ======
        if use_visual_rerank and unique_l1_scenes:
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Step 2] L1 视觉重排")
                print(f"{'='*60}")
            
            video_memory_dir = osp.join(self.memory_dir, video_id)
            
            reranked_scenes = self.l1_reranker.rerank_scenes(
                l1_scenes=unique_l1_scenes,
                question_text=question_text,
                memory_dir=self.memory_dir,
                top_k=top_k_l1,
                max_images_per_scene=max_images_per_scene,
                verbose=verbose
            )
            
            result['l1_visual_rerank'] = [
                {'scene_index': scene.scene_index, 'confidence': conf}
                for scene, conf in reranked_scenes
            ]
            
            final_scenes = [scene for scene, _ in reranked_scenes]
        else:
            # 不使用视觉重排，直接取前 top_k_l1 个场景
            final_scenes = unique_l1_scenes[:top_k_l1]
        
        result['final_scenes'] = [scene.scene_index for scene in final_scenes]
        
        if verbose:
            print(f"\n[最终场景] 使用 {len(final_scenes)} 个场景进行 VQA")
            for scene in final_scenes:
                print(f"    场景 {scene.scene_index}: {scene.event_fact[:50] if scene.event_fact else 'N/A'}...")
        
        # ====== Step 4: VQA 问答 ======
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Step 3] VQA 问答")
            print(f"{'='*60}")
        
        # 收集所有场景的图片路径
        scene_image_paths = []
        for scene in final_scenes:
            image_paths = self.l1_reranker.get_scene_images(
                scene, self.memory_dir, max_images_per_scene
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
            include_video=include_video
        )
        
        result['answer'] = answer_result
        
        if verbose:
            print(f"\n[VQA结果]")
            print(f"    答案: {answer_result.get('normalized_answer', 'N/A')}")
            print(f"    置信度: {answer_result.get('confidence', 0):.4f}")
            print(f"    使用图片数: {answer_result.get('total_images_used', 0)}")
        
        return result


# ============================================================================
#                           命令行入口
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="记忆检索 VQA 系统")
    parser.add_argument("--memory-dir", type=str, required=True, help="记忆文件根目录")
    parser.add_argument("--video-id", type=str, required=True, help="视频 ID")
    parser.add_argument("--question", type=str, required=True, help="问题文本")
    parser.add_argument("--options", type=str, nargs="+", required=True, help="选项列表")
    parser.add_argument("--api-url", type=str, default="http://localhost:8002/v1", help="vLLM API URL")
    parser.add_argument("--model-name", type=str, default="Qwen3-VL-2B-Instruct", help="模型名称")
    parser.add_argument("--top-k-l2", type=int, default=2, help="L2 检索返回数量")
    parser.add_argument("--top-k-l1", type=int, default=1, help="L1 视觉重排返回数量")
    parser.add_argument("--no-visual-rerank", action="store_true", help="禁用 L1 视觉重排")
    parser.add_argument("--video-path", type=str, default=None, help="视频路径（可选）")
    parser.add_argument("--include-video", action="store_true", help="VQA 时同时输入视频")
    
    args = parser.parse_args()
    
    # 创建系统
    system = MemoryRetrievalVQA(
        memory_dir=args.memory_dir,
        api_base_url=args.api_url,
        model_name=args.model_name
    )
    
    # 执行检索和问答
    result = system.retrieve_and_answer(
        video_id=args.video_id,
        question_text=args.question,
        options=args.options,
        top_k_l2=args.top_k_l2,
        top_k_l1=args.top_k_l1,
        use_visual_rerank=not args.no_visual_rerank,
        video_path=args.video_path,
        include_video=args.include_video,
        verbose=True
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    
    # 把 numpy 类型转成 Python 原生类型，避免 json 报错
    def to_serializable(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    print(json.dumps(result, ensure_ascii=False, indent=2, default=to_serializable))


if __name__ == "__main__":
    main()
