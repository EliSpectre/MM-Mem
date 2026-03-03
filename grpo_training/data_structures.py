"""
GRPO 训练数据结构定义
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class WindowTrajectory:
    """单个窗口的轨迹数据"""
    video_id: str
    window_idx: int
    l1_start_idx: int                       # 窗口起始 L1 索引
    l1_end_idx: int                         # 窗口结束 L1 索引
    
    # 窗口输入
    window_l1_scene_indices: List[int] = field(default_factory=list)  # L1 scene indices
    window_l1_keyframe_paths: List[List[str]] = field(default_factory=list)  # 每个L1的关键帧路径
    window_l1_texts: List[str] = field(default_factory=list)  # 窗口内 L1 的 event_fact
    
    # 上下文
    prev_l2_text: str = ""                  # 前一个 L2 的 working_memory
    current_l2_text: str = ""               # 当前活跃 L2 的 working_memory
    
    # 采样的 G 条路径
    sampled_paths: List[List[str]] = field(default_factory=list)  # G 条路径，每条是 N 个 action
    
    # Old Log Probabilities（采样时的 log prob，用于重要性采样）
    old_log_probs: List[float] = field(default_factory=list)  # G 个路径的 log probability
    
    # Reward
    r_teacher_scores: List[float] = field(default_factory=list)  # G 个 Teacher 打分
    r_vqa: float = 0.0                      # VQA 奖励（Phase 2 填充）
    r_total_scores: List[float] = field(default_factory=list)  # G 个总分（Phase 2 计算）
    
    # 选中的路径
    chosen_path_idx: int = 0                # 执行的路径索引
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WindowTrajectory':
        return cls(**data)


@dataclass
class QAResult:
    """单个问答的结果"""
    qa_id: str
    video_id: str
    question: str
    choices: List[str]
    correct_idx: int
    model_answer: str = ""
    model_answer_idx: int = -1
    is_correct: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QAResult':
        return cls(**data)


@dataclass
class VideoTrajectory:
    """单个视频的完整轨迹"""
    video_id: str
    video_path: str
    windows: List[WindowTrajectory] = field(default_factory=list)
    
    # L1/L2 记忆路径
    l1_memory_path: str = ""
    l2_memory_path: str = ""
    
    # VQA 结果（Phase 2 填充）
    qa_results: List[QAResult] = field(default_factory=list)
    r_vqa: float = 0.0                      # 该视频的 VQA 准确率
    
    def to_dict(self) -> Dict:
        data = {
            'video_id': self.video_id,
            'video_path': self.video_path,
            'l1_memory_path': self.l1_memory_path,
            'l2_memory_path': self.l2_memory_path,
            'r_vqa': self.r_vqa,
            'windows': [w.to_dict() for w in self.windows],
            'qa_results': [q.to_dict() for q in self.qa_results]
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoTrajectory':
        windows = [WindowTrajectory.from_dict(w) for w in data.get('windows', [])]
        qa_results = [QAResult.from_dict(q) for q in data.get('qa_results', [])]
        return cls(
            video_id=data['video_id'],
            video_path=data['video_path'],
            l1_memory_path=data.get('l1_memory_path', ''),
            l2_memory_path=data.get('l2_memory_path', ''),
            r_vqa=data.get('r_vqa', 0.0),
            windows=windows,
            qa_results=qa_results
        )
    
    def save(self, path: str):
        """保存到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'VideoTrajectory':
        """从 JSON 文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class GRPOSample:
    """Swift GRPO 训练样本格式"""
    query: str                              # 输入 prompt
    responses: List[str]                    # G 条候选响应
    rewards: List[float]                    # 对应的 reward
    
    # Old Log Probabilities（用于重要性采样）
    old_log_probs: List[float] = field(default_factory=list)  # G 个路径的 log probability
    
    # 额外信息（用于追溯）
    video_id: str = ""
    window_idx: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GRPOSample':
        return cls(**data)


def convert_trajectories_to_grpo_samples(
    trajectories: List[VideoTrajectory],
    prompt_template: str = None
) -> List[GRPOSample]:
    """将轨迹数据转换为 Swift GRPO 训练样本"""
    
    if prompt_template is None:
        prompt_template = (
            "You are a video memory agent. Based on the current context, decide the action for each scene.\n\n"
            "Previous L2 memory: {prev_l2_text}\n"
            "Current L2 memory: {current_l2_text}\n"
            "New scenes to process:\n{scenes_text}\n\n"
            "For each scene, choose one action: CREATE_NEW, MERGE, DISCARD, or UPDATE.\n"
            "Output the actions as a list, e.g., [MERGE, CREATE_NEW, MERGE, DISCARD, MERGE]"
        )
    
    samples = []
    
    for traj in trajectories:
        for window in traj.windows:
            # 构建场景文本
            scenes_text = ""
            for i, (scene_idx, text) in enumerate(zip(window.window_l1_scene_indices, window.window_l1_texts)):
                scenes_text += f"Scene {scene_idx}: {text}\n"
            
            # 构建 prompt
            query = prompt_template.format(
                prev_l2_text=window.prev_l2_text or "None",
                current_l2_text=window.current_l2_text or "None",
                scenes_text=scenes_text.strip()
            )
            
            # 构建 responses
            responses = [str(path) for path in window.sampled_paths]
            
            # 使用 r_total_scores（如果有）或 r_teacher_scores
            if window.r_total_scores:
                rewards = window.r_total_scores
            else:
                rewards = window.r_teacher_scores
            
            # 获取 old_log_probs（用于重要性采样）
            old_log_probs = window.old_log_probs if window.old_log_probs else []
            
            sample = GRPOSample(
                query=query,
                responses=responses,
                rewards=rewards,
                old_log_probs=old_log_probs,
                video_id=traj.video_id,
                window_idx=window.window_idx
            )
            samples.append(sample)
    
    return samples
