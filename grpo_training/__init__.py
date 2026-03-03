"""
GRPO 强化学习微调模块

用于训练 Qwen3-VL-2B 模型做 L1→L2 聚合决策
"""

from .data_structures import WindowTrajectory, VideoTrajectory, QAResult
from .trajectory_collector import TrajectoryCollector
from .vqa_evaluator import VQAEvaluator
from .grpo_trainer import GRPOTrainer

__all__ = [
    'WindowTrajectory',
    'VideoTrajectory', 
    'QAResult',
    'TrajectoryCollector',
    'VQAEvaluator',
    'GRPOTrainer'
]
