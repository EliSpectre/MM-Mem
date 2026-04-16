"""
SIB-GRPO Trainer
- LoRA setup (using peft)
- Reference model management
- GRPO loss computation and update
- Training loop
"""

import os
import sys
import copy
import json
import math
import logging
import random
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from PIL import Image

from config import GRPOConfig
from dataset import load_training_data, find_video_path, ensure_l1_built
from rollout import rollout_trajectories, policy_compute_log_prob
from reward import compute_total_reward

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_retrieve"))
from models import build_messages
from memory_build import L2_DECISION_PROMPT

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    SIB-GRPO Trainer

    Training flow:
    1. Rollout G trajectories (policy model samples L2 decisions)
    2. Compute reward for each trajectory
    3. Normalize advantages within group
    4. Recompute current policy log_probs (with gradients)
    5. Compute GRPO loss + KL penalty
    6. Backpropagate to update LoRA parameters
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_model,
        policy_processor,
        base_llm,
        base_processor,
        supervisor_llm,
        supervisor_processor,
        embedding_model,
    ):
        self.config = config
        self.policy_model = policy_model
        self.policy_processor = policy_processor
        self.base_llm = base_llm
        self.base_processor = base_processor
        self.supervisor_llm = supervisor_llm
        self.supervisor_processor = supervisor_processor
        self.embedding_model = embedding_model

        # Setup optimizer
        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Save reference model parameters (for KL regularization)
        # Only save initial values of LoRA parameters
        self.ref_params = {}
        for name, param in self.policy_model.named_parameters():
            if param.requires_grad:
                self.ref_params[name] = param.data.clone()

        self.global_step = 0
        self.grad_accum_count = 0

    def compute_kl_penalty(self) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        Simplified version: uses L2 distance in parameter space as KL approximation.
        """
        kl = torch.tensor(0.0, device=self.config.device)
        for name, param in self.policy_model.named_parameters():
            if param.requires_grad and name in self.ref_params:
                kl += torch.sum((param - self.ref_params[name].to(param.device)) ** 2)
        return kl

    def train_step(
        self,
        video_path: str,
        video_id: str,
        l1_nodes: List[Dict],
        question: str,
        choices: List[str],
        correct_idx: int,
    ) -> Dict[str, float]:
        """
        One training step:
        1. Rollout G trajectories
        2. Compute rewards
        3. GRPO update
        """
        # --- 1. Rollout ---
        self.policy_model.eval()  # No gradients needed during rollout
        trajectories = rollout_trajectories(
            self.policy_model, self.policy_processor,
            self.base_llm, self.base_processor,
            l1_nodes, question, choices,
            self.config, self.embedding_model,
            video_path, video_id,
        )

        # --- 2. Compute Rewards ---
        rewards = []
        reward_details = []
        for traj in trajectories:
            rd = compute_total_reward(
                traj, correct_idx, choices, l1_nodes, self.config,
                self.supervisor_llm, self.supervisor_processor,
            )
            rewards.append(rd["total"])
            reward_details.append(rd)

        # --- 3. Normalize advantages within group ---
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1 and rewards_tensor.std() > 0:
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        else:
            advantages = rewards_tensor - rewards_tensor.mean()

        # --- 4. GRPO Update ---
        self.policy_model.train()
        total_loss = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        num_steps = 0

        for k, traj in enumerate(trajectories):
            advantage = advantages[k].item()

            for step in traj["steps"]:
                if step.get("is_default", False):
                    continue  # Skip the default first ADD_NEW
                if step["target_token_id"] < 0:
                    continue  # Invalid token

                # Reconstruct messages (using saved info)
                decision_text = step.get("decision_text", "")
                combined_paths = step.get("combined_paths", [])

                if not decision_text or not combined_paths:
                    continue

                # Load frames
                combined_images = []
                for p in combined_paths:
                    if os.path.exists(p):
                        combined_images.append(Image.open(p).convert("RGB"))

                if not combined_images:
                    continue

                messages = build_messages(decision_text, images=combined_images)

                # Recompute current policy log_prob (with gradients)
                new_log_prob = policy_compute_log_prob(
                    self.policy_model, self.policy_processor,
                    messages, combined_images,
                    target_token_id=step["target_token_id"],
                    device=self.config.device,
                )

                old_log_prob = step["log_prob"]

                # PPO-style clipped objective
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                step_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

                total_loss = total_loss + step_loss
                num_steps += 1

        if num_steps > 0:
            total_loss = total_loss / num_steps

        # KL penalty
        kl_penalty = self.compute_kl_penalty()
        total_loss = total_loss + self.config.kl_coeff * kl_penalty

        # Gradient accumulation
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()
        self.grad_accum_count += 1

        if self.grad_accum_count >= self.config.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.grad_accum_count = 0
            self.global_step += 1

        # Statistics
        mean_reward = rewards_tensor.mean().item()
        correct_rate = sum(1 for rd in reward_details if rd["correct"] > 0) / len(reward_details)

        return {
            "loss": total_loss.item(),
            "mean_reward": mean_reward,
            "correct_rate": correct_rate,
            "kl_penalty": kl_penalty.item(),
            "num_steps": num_steps,
            "advantages_std": advantages.std().item() if len(advantages) > 1 else 0.0,
        }

    def save_checkpoint(self, step: int):
        """Save LoRA checkpoint"""
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # Save only LoRA parameters
        lora_state = {}
        for name, param in self.policy_model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
        logger.info(f"Checkpoint saved: {save_dir}")

    def train(self, dataset: List[Dict]):
        """Full training loop"""
        logger.info(f"===== Starting SIB-GRPO Training =====")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Trajectories per sample: {self.config.num_generations}")

        os.makedirs(self.config.output_dir, exist_ok=True)

        sample_idx = 0
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")

            # Shuffle data each epoch
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            for i, idx in enumerate(indices):
                sample = dataset[idx]
                video_id = sample["video_id"]
                question = sample["question"]
                choices = sample["choices"]
                correct_idx = sample["correct_idx"]

                # Find video
                video_path = find_video_path(self.config.video_dir, video_id)
                if not video_path:
                    logger.warning(f"  Video not found: {video_id}, skipping")
                    continue

                # Ensure L1 is built
                l1_nodes = ensure_l1_built(
                    video_path, video_id, self.config.l1_cache_dir,
                    self.base_llm, self.base_processor, self.config,
                )
                if not l1_nodes:
                    logger.warning(f"  L1 nodes empty: {video_id}, skipping")
                    continue

                # Training step
                logger.info(f"  [{i+1}/{len(dataset)}] {video_id} | Q: {question[:50]}...")
                try:
                    stats = self.train_step(
                        video_path, video_id, l1_nodes,
                        question, choices, correct_idx,
                    )
                    sample_idx += 1

                    if sample_idx % self.config.log_steps == 0:
                        logger.info(
                            f"  Step {self.global_step} | "
                            f"loss={stats['loss']:.4f} | "
                            f"reward={stats['mean_reward']:.4f} | "
                            f"correct={stats['correct_rate']:.2%} | "
                            f"kl={stats['kl_penalty']:.6f}"
                        )

                    if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)

                except Exception as e:
                    logger.error(f"  Training step failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Final save
        self.save_checkpoint(self.global_step)
        logger.info("===== Training Complete =====")
