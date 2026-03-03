"""
Phase 1: 轨迹收集器

功能:
1. 加载/生成 L1 记忆
2. 窗口化处理 L1 节点
3. 采样 G 条路径
4. 用 Teacher Model 打分
5. 执行最优路径，生成 L2 记忆
6. 记录轨迹数据
"""

import os
import os.path as osp
import json
import random
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI

from .data_structures import WindowTrajectory, VideoTrajectory


# 动作类型
ACTIONS = ["CREATE_NEW", "MERGE", "DISCARD", "UPDATE"]


class TrajectoryCollector:
    """轨迹收集器"""
    
    def __init__(
        self,
        # 微调模型配置 (用于采样路径)
        finetune_model_path: str = None,
        finetune_device: str = "cuda:0",
        
        # Teacher API 配置 (用于打分)
        teacher_api_url: str = "http://localhost:8003/v1",
        teacher_api_key: str = "EMPTY",
        teacher_model_name: str = "Qwen3-VL-32B-Instruct",
        
        # Base API 配置 (用于 L1 生成)
        base_api_url: str = "http://localhost:8002/v1",
        base_api_key: str = "EMPTY",
        base_model_name: str = "Qwen2.5-VL-7B-Instruct",
        
        # 窗口配置
        window_size: int = 5,
        window_stride: int = 3,
        num_sampled_paths: int = 4,
        sampling_temperature: float = 0.7,
        
        # 输出目录
        output_dir: str = "./output/trajectories",
        memory_dir: str = "./output/memories",
    ):
        self.finetune_model_path = finetune_model_path
        self.finetune_device = finetune_device
        
        self.teacher_api_url = teacher_api_url
        self.teacher_api_key = teacher_api_key
        self.teacher_model_name = teacher_model_name
        
        self.base_api_url = base_api_url
        self.base_api_key = base_api_key
        self.base_model_name = base_model_name
        
        self.window_size = window_size
        self.window_stride = window_stride
        self.num_sampled_paths = num_sampled_paths
        self.sampling_temperature = sampling_temperature
        
        # 确保使用绝对路径
        self.output_dir = osp.abspath(output_dir)
        self.memory_dir = osp.abspath(memory_dir)
        
        print(f"[TrajectoryCollector] output_dir: {self.output_dir}")
        print(f"[TrajectoryCollector] memory_dir: {self.memory_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 延迟加载模型
        self.finetune_model = None
        self.finetune_processor = None
        self._finetune_loaded = False
        
        # API 客户端
        self.teacher_client = OpenAI(
            base_url=teacher_api_url,
            api_key=teacher_api_key
        )
        self.base_client = OpenAI(
            base_url=base_api_url,
            api_key=base_api_key
        )
    
    def _load_finetune_model(self):
        """延迟加载微调模型"""
        if self._finetune_loaded:
            return
        
        if self.finetune_model_path is None:
            print("[警告] 未指定微调模型路径，将使用随机采样")
            self._finetune_loaded = True
            return
        
        print(f"[加载] 微调模型: {self.finetune_model_path}")
        
        try:
            from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
            import torch
            
            self.finetune_processor = AutoProcessor.from_pretrained(
                self.finetune_model_path
            )
            self.finetune_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.finetune_model_path,
                dtype=torch.bfloat16,
                device_map=self.finetune_device
            )
            self.finetune_model.eval()
            self._finetune_loaded = True
            print("[完成] 微调模型加载成功")
        except Exception as e:
            print(f"[错误] 加载微调模型失败: {e}")
            import traceback
            traceback.print_exc()
            self._finetune_loaded = True
    
    def _load_l1_memories(self, video_id: str) -> List[Dict]:
        """加载已有的 L1 记忆"""
        video_memory_dir = osp.join(self.memory_dir, video_id)
        l1_path = osp.join(video_memory_dir, "episodic_memories_L1.json")
        
        if not osp.exists(l1_path):
            return []
        
        memories = []
        with open(l1_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return memories
    
    def _generate_l1_memories(
        self,
        video_path: str,
        video_id: str,
        frames_per_scene: int = 8
    ) -> List[Dict]:
        """生成 L1 记忆（调用 memory_generator_finetuning 的逻辑）"""
        import tempfile
        
        # 动态导入，避免循环导入
        import sys
        parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from memory_generator_finetuning import create_memory_generator
        
        # 尝试加载已有配置
        config_path = osp.join(parent_dir, "configs/memory_generator_config.json")
        
        if osp.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 使用默认配置
            config = {
                "pyscenesdetect_threshold": 20.0,
                "adaptive_k": 2.0
            }
        
        # 确保 base_api 配置正确
        config["base_api"] = {
            "enabled": True,
            "base_url": self.base_api_url,
            "api_key": self.base_api_key,
            "model_name": self.base_model_name,
            "max_concurrent": 4
        }
        
        # 写入临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            temp_config_path = f.name
        
        try:
            # 创建生成器
            generator = create_memory_generator(
                finetune_model_path=self.finetune_model_path or "Qwen/Qwen3-VL-2B-Instruct",
                config_path=temp_config_path,
                output_dir=self.memory_dir,
                finetune_device=self.finetune_device
            )
            
            # 只生成 L1（会自动跳过已存在的）
            generator.generate_l1_memories(video_path, skip_if_exists=False)
            
        finally:
            # 清理临时配置文件
            if osp.exists(temp_config_path):
                os.remove(temp_config_path)
        
        # 从文件加载 L1 记忆
        return self._load_l1_memories(video_id)
    
    def _sample_paths_random(self, window_size: int) -> List[List[str]]:
        """随机采样路径（当没有模型时使用）"""
        paths = []
        for _ in range(self.num_sampled_paths):
            path = [random.choice(ACTIONS) for _ in range(window_size)]
            paths.append(path)
        return paths
    
    def _sample_paths_from_model(
        self,
        window_l1_memories: List[Dict],
        prev_l2_text: str,
        current_l2_text: str
    ) -> Tuple[List[List[str]], List[float]]:
        """
        从微调模型采样路径，同时返回 log probabilities
        使用并行采样策略，一次 forward 生成多条路径
        
        Returns:
            Tuple[List[List[str]], List[float]]: (路径列表, 对应的log_prob列表)
        """
        self._load_finetune_model()
        
        if self.finetune_model is None:
            paths = self._sample_paths_random(len(window_l1_memories))
            # 随机采样没有 log_prob，使用 0.0 作为占位
            return paths, [0.0] * len(paths)
        
        import torch
        from PIL import Image
        
        num_samples = self.num_sampled_paths  # 并行采样数量
        
        # 构建 prompt
        scenes_text = ""
        images = []
        
        for i, mem in enumerate(window_l1_memories):
            scenes_text += f"Scene {mem.get('scene_index', i)}: {mem.get('event_fact', 'Unknown')}\n"
            
            # 加载第一张关键帧
            keyframe_paths = mem.get('keyframe_paths', [])
            if keyframe_paths:
                # 确保使用绝对路径
                img_path = osp.join(self.memory_dir, keyframe_paths[0])
                img_path = osp.abspath(img_path)  # 转换为绝对路径
                if osp.exists(img_path):
                    images.append(Image.open(img_path).convert('RGB'))
        
        prompt = (
            "You are a video memory agent. Based on the current context, decide the action for each scene.\n\n"
            f"Previous L2 memory: {prev_l2_text or 'None'}\n"
            f"Current L2 memory: {current_l2_text or 'None'}\n"
            f"New scenes to process:\n{scenes_text}\n\n"
            "For each scene, choose one action: CREATE_NEW, MERGE, DISCARD, or UPDATE.\n"
            "Output the actions as a JSON list, e.g., [\"MERGE\", \"CREATE_NEW\", \"MERGE\"]"
        )
        
        try:
            # 构建输入
            if images:
                content = [{"type": "image", "image": img} for img in images]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            # 使用 processor - 采用官方的 apply_chat_template 方式
            inputs = self.finetune_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.finetune_model.device)
            
            # 记录输入长度，用于后续截取生成部分
            input_len = inputs['input_ids'].shape[1]
            
            # 使用 num_return_sequences 进行并行采样
            # 注意：对于多模态模型，不能直接 expand inputs，而是让模型自己处理
            with torch.no_grad():
                outputs = self.finetune_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=self.sampling_temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=num_samples,  # 并行生成多个序列
                    return_dict_in_generate=True,
                    output_scores=True  # 返回每一步的 scores
                )
            
            # 处理生成结果
            # outputs.sequences: [num_samples, seq_len]
            # outputs.scores: tuple of [num_samples, vocab_size], 长度为生成的 token 数
            
            paths = []
            log_probs = []
            
            # 定义 log_prob 的上下界，防止 -inf 和数值溢出
            # 单个 token 的 log_prob 范围：正常在 [-15, 0]，极端情况不应低于 -20
            MIN_TOKEN_LOG_PROB = -20.0   # 单个 token 的下界
            # 整个序列的 log_prob 范围：取决于序列长度，但设置一个合理的总下界
            MIN_TOTAL_LOG_PROB = -50.0   # 总 log_prob 的下界（更合理的范围）
            MAX_LOG_PROB = 0.0           # 上界，log_prob 不应该大于 0
            
            for sample_idx in range(num_samples):
                try:
                    # 获取该样本的生成 token ids
                    generated_ids = outputs.sequences[sample_idx][input_len:]
                    
                    # 计算该序列的 log probability
                    total_log_prob = 0.0
                    valid_steps = 0
                    
                    for step_idx, token_id in enumerate(generated_ids):
                        if step_idx < len(outputs.scores):
                            score = outputs.scores[step_idx]  # [num_samples, vocab_size]
                            # 对该样本的 scores 做 log_softmax
                            log_probs_step = torch.nn.functional.log_softmax(score[sample_idx], dim=-1)
                            token_log_prob = log_probs_step[token_id].item()
                            
                            # 截断单个 token 的极端值
                            if token_log_prob < MIN_TOKEN_LOG_PROB:
                                token_log_prob = MIN_TOKEN_LOG_PROB
                            elif token_log_prob > MAX_LOG_PROB:
                                token_log_prob = MAX_LOG_PROB
                            
                            # 检查是否为有效数值（NaN 检查）
                            if not (token_log_prob != token_log_prob):
                                total_log_prob += token_log_prob
                                valid_steps += 1
                    
                    # 如果没有有效步骤，使用默认值
                    if valid_steps == 0:
                        total_log_prob = MIN_TOTAL_LOG_PROB
                    else:
                        # 最终截断总 log_prob 到合理范围
                        total_log_prob = max(total_log_prob, MIN_TOTAL_LOG_PROB)
                    
                    # 解码
                    response = self.finetune_processor.decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                    
                    # 解析动作列表
                    path = self._parse_action_list(response, len(window_l1_memories))
                    paths.append(path)
                    log_probs.append(total_log_prob)
                    
                except Exception as e:
                    print(f"[警告] 解析样本 {sample_idx} 失败: {e}")
                    paths.append(self._sample_paths_random(len(window_l1_memories))[0])
                    log_probs.append(MIN_TOTAL_LOG_PROB)  # 使用总下界
            
            return paths, log_probs
            
        except Exception as e:
            print(f"[警告] 并行采样失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到随机采样
            paths = self._sample_paths_random(len(window_l1_memories))
            return paths, [-50.0] * len(paths)  # 使用总下界
    
    def _parse_action_list(self, response: str, expected_length: int) -> List[str]:
        """解析动作列表"""
        import re
        
        # 尝试提取 JSON 列表
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                actions = json.loads(match.group())
                # 验证动作有效性
                valid_actions = []
                for action in actions:
                    action_upper = str(action).upper().replace('"', '').replace("'", "")
                    if action_upper in ACTIONS:
                        valid_actions.append(action_upper)
                    else:
                        valid_actions.append("MERGE")  # 默认
                
                # 补齐或截断
                while len(valid_actions) < expected_length:
                    valid_actions.append("MERGE")
                return valid_actions[:expected_length]
                
            except json.JSONDecodeError:
                pass
        
        # 解析失败，返回默认
        return ["MERGE"] * expected_length
    
    def _score_paths_with_teacher(
        self,
        window_l1_memories: List[Dict],
        paths: List[List[str]],
        prev_l2_text: str,
        current_l2_text: str
    ) -> List[float]:
        """用 Teacher Model 给路径打分 - 基于 logprobs"""
        scores = []
        
        # 构建场景描述
        scenes_text = ""
        for i, mem in enumerate(window_l1_memories):
            scenes_text += f"Scene {mem.get('scene_index', i)}: {mem.get('event_fact', 'Unknown')}\n"
        
        for path in paths:
            # 修改 prompt 为判断型问题，限定输出
            prompt = (
                "You are evaluating a video memory organization decision.\n\n"
                f"Context:\n"
                f"- Previous L2 memory: {prev_l2_text or 'None'}\n"
                f"- Current L2 memory: {current_l2_text or 'None'}\n\n"
                f"Scenes to organize:\n{scenes_text}\n"
                f"Proposed action sequence: {path}\n\n"
                "Actions explanation:\n"
                "- CREATE_NEW: Start a new semantic memory (L2)\n"
                "- MERGE: Merge this scene into the current L2\n"
                "- DISCARD: Skip this scene (not important)\n"
                "- UPDATE: Update the current L2 with new information\n\n"
                "Evaluate based on:\n"
                "1. Temporal coherence (scenes in same L2 should be continuous)\n"
                "2. Semantic consistency (scenes in same L2 should be related)\n"
                "3. Information completeness (important info should not be discarded)\n\n"
                "Is this a good decision? Answer with ONLY one word: GOOD or BAD.\n"
                "Answer:"
            )
            
            try:
                # 使用 logprobs 参数获取 token 概率
                response = self.teacher_client.chat.completions.create(
                    model=self.teacher_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,  # 只需要生成第一个 token
                    temperature=0,
                    logprobs=True,  # 启用 logprobs
                    top_logprobs=10  # 获取 top 10 个 token 的概率
                )
                
                # 获取第一个 token 的 logprobs
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                    
                    # 只检测 GOOD/BAD (及其大小写变体)
                    positive_tokens = ["GOOD", "Good", "good"]
                    negative_tokens = ["BAD", "Bad", "bad"]
                    
                    # 收集正面和负面 token 的 logprobs
                    pos_logprobs = []
                    neg_logprobs = []
                    
                    for token_info in token_logprobs:
                        token_text = token_info.token.strip()
                        if token_text in positive_tokens:
                            pos_logprobs.append(token_info.logprob)
                        elif token_text in negative_tokens:
                            neg_logprobs.append(token_info.logprob)
                    
                    # 计算概率得分
                    if pos_logprobs or neg_logprobs:
                        # 使用 logsumexp 技巧计算总概率
                        import math
                        
                        if pos_logprobs:
                            max_pos = max(pos_logprobs)
                            pos_prob = math.exp(max_pos + math.log(sum(math.exp(lp - max_pos) for lp in pos_logprobs)))
                        else:
                            pos_prob = 0.0
                        
                        if neg_logprobs:
                            max_neg = max(neg_logprobs)
                            neg_prob = math.exp(max_neg + math.log(sum(math.exp(lp - max_neg) for lp in neg_logprobs)))
                        else:
                            neg_prob = 0.0
                        
                        # 归一化并转换为 0-10 分数
                        total_prob = pos_prob + neg_prob
                        if total_prob > 0:
                            score = (pos_prob / total_prob) * 10.0
                        else:
                            # 如果没有匹配到目标 token，使用最高概率 token 的 logprob
                            max_token = max(token_logprobs, key=lambda x: x.logprob)
                            # 将 logprob 转换为概率，然后映射到 0-10
                            prob = math.exp(max_token.logprob)
                            score = prob * 10.0
                    else:
                        score = 5.0  # 默认中等分数
                else:
                    score = 5.0
                    
            except Exception as e:
                print(f"[警告] Teacher 打分失败: {e}")
                score = 5.0
            
            scores.append(score)
        
        return scores
    
    def _execute_path(
        self,
        window_l1_memories: List[Dict],
        path: List[str],
        current_l2_list: List[Dict],
        active_l2_idx: int
    ) -> Tuple[List[Dict], int]:
        """执行一条路径，更新 L2 状态"""
        for i, (l1_mem, action) in enumerate(zip(window_l1_memories, path)):
            if action == "CREATE_NEW":
                # 创建新的 L2
                new_l2 = {
                    "l2_index": len(current_l2_list),
                    "l1_scene_indices": [l1_mem.get("scene_index", 0)],
                    "l1_details": [l1_mem],
                    "start_sec": l1_mem.get("start_sec", 0),
                    "end_sec": l1_mem.get("end_sec", 0),
                    "working_memory": l1_mem.get("event_fact", ""),
                    "event_summary": l1_mem.get("event_fact", "")
                }
                current_l2_list.append(new_l2)
                active_l2_idx = len(current_l2_list) - 1
                
            elif action == "MERGE":
                if active_l2_idx >= 0 and active_l2_idx < len(current_l2_list):
                    # 合并到当前 L2
                    l2 = current_l2_list[active_l2_idx]
                    l2["l1_scene_indices"].append(l1_mem.get("scene_index", 0))
                    l2["l1_details"].append(l1_mem)
                    l2["end_sec"] = l1_mem.get("end_sec", l2["end_sec"])
                    # 更新 working_memory
                    l2["working_memory"] += " " + l1_mem.get("event_fact", "")
                else:
                    # 没有活跃 L2，创建新的
                    new_l2 = {
                        "l2_index": len(current_l2_list),
                        "l1_scene_indices": [l1_mem.get("scene_index", 0)],
                        "l1_details": [l1_mem],
                        "start_sec": l1_mem.get("start_sec", 0),
                        "end_sec": l1_mem.get("end_sec", 0),
                        "working_memory": l1_mem.get("event_fact", ""),
                        "event_summary": l1_mem.get("event_fact", "")
                    }
                    current_l2_list.append(new_l2)
                    active_l2_idx = len(current_l2_list) - 1
                    
            elif action == "UPDATE":
                if active_l2_idx >= 0 and active_l2_idx < len(current_l2_list):
                    # 更新当前 L2 的信息
                    l2 = current_l2_list[active_l2_idx]
                    l2["l1_scene_indices"].append(l1_mem.get("scene_index", 0))
                    l2["l1_details"].append(l1_mem)
                    l2["end_sec"] = l1_mem.get("end_sec", l2["end_sec"])
                    # 覆盖 working_memory
                    l2["working_memory"] = l1_mem.get("event_fact", "")
                    
            elif action == "DISCARD":
                # 跳过这个场景
                pass
        
        return current_l2_list, active_l2_idx
    
    def collect_trajectory(
        self,
        video_path: str,
        video_id: str = None,
        force_regenerate_l1: bool = False
    ) -> VideoTrajectory:
        """
        收集单个视频的轨迹
        
        Args:
            video_path: 视频文件路径
            video_id: 视频 ID（如果不指定，从文件名提取）
            force_regenerate_l1: 是否强制重新生成 L1
            
        Returns:
            VideoTrajectory 对象
        """
        if video_id is None:
            video_id = osp.splitext(osp.basename(video_path))[0]
        
        print(f"\n{'='*60}")
        print(f"[收集轨迹] Video: {video_id}")
        print(f"{'='*60}")
        
        # 创建视频记忆目录
        video_memory_dir = osp.join(self.memory_dir, video_id)
        os.makedirs(video_memory_dir, exist_ok=True)
        
        # Step 1: 加载或生成 L1 记忆
        l1_path = osp.join(video_memory_dir, "episodic_memories_L1.json")
        
        if osp.exists(l1_path) and not force_regenerate_l1:
            print(f"[加载] L1 记忆已存在，直接加载")
            l1_memories = self._load_l1_memories(video_id)
        else:
            print(f"[生成] L1 记忆不存在，开始生成")
            l1_memories = self._generate_l1_memories(video_path, video_id)
        
        if not l1_memories:
            print("[错误] 没有 L1 记忆")
            return VideoTrajectory(video_id=video_id, video_path=video_path)
        
        print(f"[信息] 共 {len(l1_memories)} 个 L1 场景")
        
        # Step 2: 初始化
        trajectory = VideoTrajectory(
            video_id=video_id,
            video_path=video_path,
            l1_memory_path=l1_path
        )
        
        current_l2_list = []
        active_l2_idx = -1
        window_idx = 0
        
        # Step 3: 处理第一个L1节点（特殊处理）
        if len(l1_memories) > 0:
            first_l1 = l1_memories[0]
            print(f"\n[窗口 0] 第一个 L1 节点特殊处理 - 直接 CREATE_NEW")
            
            # 直接创建第一个 L2
            first_path = ["CREATE_NEW"]
            current_l2_list, active_l2_idx = self._execute_path(
                [first_l1],
                first_path,
                current_l2_list,
                active_l2_idx
            )
            
            # 记录第一个窗口的轨迹
            window_traj = WindowTrajectory(
                video_id=video_id,
                window_idx=0,
                l1_start_idx=0,
                l1_end_idx=0,
                window_l1_scene_indices=[first_l1.get("scene_index", 0)],
                window_l1_keyframe_paths=[first_l1.get("keyframe_paths", [])],
                window_l1_texts=[first_l1.get("event_fact", "")],
                prev_l2_text="",
                current_l2_text="",
                sampled_paths=[first_path],
                r_teacher_scores=[10.0],  # 特殊处理，给最高分
                chosen_path_idx=0
            )
            trajectory.windows.append(window_traj)
            
            print(f"    [执行] 创建第一个 L2 记忆")
            window_idx = 1
        
        # Step 4: 从第二个节点开始进行滑动窗口处理
        i = 1  # 从第二个节点开始
        while i < len(l1_memories):
            # 获取窗口内的 L1 记忆
            window_end = min(i + self.window_size, len(l1_memories))
            window_l1_memories = l1_memories[i:window_end]
            actual_window_size = len(window_l1_memories)
            
            print(f"\n[窗口 {window_idx}] L1 索引 {i} - {window_end - 1}，共 {actual_window_size} 个场景")
            
            # 获取上下文
            prev_l2_text = ""
            current_l2_text = ""
            
            if active_l2_idx > 0 and active_l2_idx - 1 < len(current_l2_list):
                prev_l2_text = current_l2_list[active_l2_idx - 1].get("working_memory", "")
            
            if active_l2_idx >= 0 and active_l2_idx < len(current_l2_list):
                current_l2_text = current_l2_list[active_l2_idx].get("working_memory", "")
            
            # Step 4.1: 采样 G 条路径
            print(f"    [采样] 采样 {self.num_sampled_paths} 条路径...")
            sampled_paths, old_log_probs = self._sample_paths_from_model(
                window_l1_memories,
                prev_l2_text,
                current_l2_text
            )
            
            for p_idx, path in enumerate(sampled_paths):
                log_prob_str = f", log_prob={old_log_probs[p_idx]:.4f}" if old_log_probs else ""
                print(f"        路径 {p_idx}: {path}{log_prob_str}")
            
            # Step 4.2: Teacher 打分
            print(f"    [打分] Teacher Model 打分...")
            r_teacher_scores = self._score_paths_with_teacher(
                window_l1_memories,
                sampled_paths,
                prev_l2_text,
                current_l2_text
            )
            
            for p_idx, score in enumerate(r_teacher_scores):
                print(f"        路径 {p_idx} 分数: {score:.2f}")
            
            # Step 4.3: 选择最优路径
            best_idx = int(np.argmax(r_teacher_scores))
            best_path = sampled_paths[best_idx]
            print(f"    [选择] 选中路径 {best_idx}: {best_path}")
            
            # Step 4.4: 执行路径
            current_l2_list, active_l2_idx = self._execute_path(
                window_l1_memories,
                best_path,
                current_l2_list,
                active_l2_idx
            )
            
            # Step 4.5: 记录轨迹
            window_traj = WindowTrajectory(
                video_id=video_id,
                window_idx=window_idx,
                l1_start_idx=i,
                l1_end_idx=window_end - 1,
                window_l1_scene_indices=[m.get("scene_index", 0) for m in window_l1_memories],
                window_l1_keyframe_paths=[m.get("keyframe_paths", []) for m in window_l1_memories],
                window_l1_texts=[m.get("event_fact", "") for m in window_l1_memories],
                prev_l2_text=prev_l2_text,
                current_l2_text=current_l2_text,
                sampled_paths=sampled_paths,
                old_log_probs=old_log_probs,  # 保存采样时的 log probabilities（用于重要性采样）
                r_teacher_scores=r_teacher_scores,
                chosen_path_idx=best_idx
            )
            trajectory.windows.append(window_traj)
            
            # 更新索引
            window_idx += 1
            i += self.window_stride
        
        # Step 4: 保存 L2 记忆
        l2_path = osp.join(video_memory_dir, "semantic_memories_L2.json")
        with open(l2_path, 'w', encoding='utf-8') as f:
            for l2 in current_l2_list:
                f.write(json.dumps(l2, ensure_ascii=False) + '\n')
        
        trajectory.l2_memory_path = l2_path
        
        print(f"\n[完成] 生成了 {len(current_l2_list)} 个 L2 记忆")
        print(f"[完成] 记录了 {len(trajectory.windows)} 个窗口轨迹")
        
        # Step 5: 保存轨迹
        traj_path = osp.join(self.output_dir, f"{video_id}_trajectory.json")
        trajectory.save(traj_path)
        print(f"[保存] 轨迹已保存到: {traj_path}")
        
        return trajectory

    def collect_trajectories_batch(
        self,
        video_list: List[Tuple[str, str]],
        force_regenerate_l1: bool = False
    ) -> List[VideoTrajectory]:
        """
        批量收集轨迹
        
        Args:
            video_list: [(video_path, video_id), ...] 列表
            force_regenerate_l1: 是否强制重新生成 L1
            
        Returns:
            VideoTrajectory 列表
        """
        trajectories = []
        
        for i, (video_path, video_id) in enumerate(video_list):
            print(f"\n[进度] {i+1}/{len(video_list)}")
            
            try:
                traj = self.collect_trajectory(
                    video_path=video_path,
                    video_id=video_id,
                    force_regenerate_l1=force_regenerate_l1
                )
                trajectories.append(traj)
            except Exception as e:
                print(f"[错误] 处理视频 {video_id} 失败: {e}")
                continue
        
        return trajectories
