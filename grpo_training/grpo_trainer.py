"""
Phase 3: GRPO 训练器

使用 Swift 框架进行 GRPO 强化学习训练

功能:
1. 将轨迹数据转换为 Swift GRPO 格式
2. 配置 Swift 训练参数
3. 运行 GRPO 训练
"""

import os
import os.path as osp
import json
import tempfile
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .data_structures import (
    VideoTrajectory, 
    WindowTrajectory, 
    GRPOSample,
    convert_trajectories_to_grpo_samples
)


@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    # 模型配置
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "./output/grpo_checkpoints"
    
    # LoRA 配置
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # 训练配置
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    max_length: int = 2048
    
    # GRPO 特定配置
    beta: float = 0.1  # KL 散度权重 / 偏好强度
    num_generations: int = 4  # 每个 prompt 的生成数量
    
    # PPO-Clip 配置
    ppo_clip_epsilon: float = 0.2  # PPO 截断参数 ε
    use_importance_sampling: bool = True  # 是否使用重要性采样
    
    # KL 惩罚配置（可选，与 PPO-clip 二选一）
    use_kl_penalty: bool = False  # 是否使用 KL 惩罚
    kl_penalty_coef: float = 0.1  # KL 惩罚系数
    
    # 其他
    seed: int = 42
    save_steps: int = 100
    logging_steps: int = 10
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "beta": self.beta,
            "num_generations": self.num_generations,
            "ppo_clip_epsilon": self.ppo_clip_epsilon,
            "use_importance_sampling": self.use_importance_sampling,
            "use_kl_penalty": self.use_kl_penalty,
            "kl_penalty_coef": self.kl_penalty_coef,
            "seed": self.seed,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps
        }


class GRPOTrainer:
    """GRPO 训练器"""
    
    def __init__(self, config: GRPOConfig = None):
        self.config = config or GRPOConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def prepare_training_data(
        self,
        trajectories: List[VideoTrajectory],
        output_path: str = None
    ) -> str:
        """
        准备 Swift GRPO 训练数据
        
        Args:
            trajectories: 视频轨迹列表
            output_path: 输出 JSON 文件路径
            
        Returns:
            训练数据文件路径
        """
        print("\n[准备] 转换轨迹数据为 GRPO 训练格式...")
        
        # 转换为 GRPO 样本
        samples = convert_trajectories_to_grpo_samples(trajectories)
        
        print(f"[信息] 共生成 {len(samples)} 个 GRPO 训练样本")
        
        # 转换为 Swift 格式
        swift_data = []
        for sample in samples:
            # Swift GRPO 格式需要 query, responses, rewards
            swift_item = {
                "query": sample.query,
                "response": sample.responses,  # Swift 使用 response 而不是 responses
                "reward": sample.rewards,      # Swift 使用 reward 而不是 rewards
            }
            swift_data.append(swift_item)
        
        # 保存
        if output_path is None:
            output_path = osp.join(self.config.output_dir, "grpo_train_data.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(swift_data, f, ensure_ascii=False, indent=2)
        
        print(f"[保存] 训练数据已保存到: {output_path}")
        
        return output_path
    
    def _create_swift_config(self, train_data_path: str) -> str:
        """创建 Swift 训练配置文件"""
        swift_config = {
            # 模型
            "model": self.config.model_path,
            "model_type": "qwen2_5_vl",
            
            # 数据
            "dataset": train_data_path,
            "dataset_type": "grpo",
            
            # LoRA
            "train_type": "lora" if self.config.use_lora else "full",  # 新版 Swift 使用 train_type
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target_modules": self.config.lora_target_modules,
            
            # 训练
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": self.config.warmup_ratio,
            "max_length": self.config.max_length,
            
            # GRPO
            "rlhf_type": "grpo",
            "beta": self.config.beta,
            "num_generations": self.config.num_generations,
            
            # 其他
            "seed": self.config.seed,
            "save_steps": self.config.save_steps,
            "logging_steps": self.config.logging_steps,
            "save_total_limit": 3,
            "gradient_checkpointing": True,
            "bf16": True,
        }
        
        config_path = osp.join(self.config.output_dir, "swift_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(swift_config, f, ensure_ascii=False, indent=2)
        
        return config_path
    
    def train_with_swift_cli(self, train_data_path: str) -> str:
        """
        使用 Swift CLI 进行训练
        
        这是推荐的方式，直接调用 swift 命令行工具
        """
        import subprocess
        
        # 创建配置
        config_path = self._create_swift_config(train_data_path)
        
        # 构建 swift 命令
        cmd = [
            "swift", "rlhf",
            "--rlhf_type", "grpo",
            "--model", self.config.model_path,
            "--dataset", train_data_path,
            "--output_dir", self.config.output_dir,
            "--num_train_epochs", str(self.config.num_train_epochs),
            "--per_device_train_batch_size", str(self.config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps),
            "--learning_rate", str(self.config.learning_rate),
            "--warmup_ratio", str(self.config.warmup_ratio),
            "--max_length", str(self.config.max_length),
            "--beta", str(self.config.beta),
            "--save_steps", str(self.config.save_steps),
            "--logging_steps", str(self.config.logging_steps),
            "--gradient_checkpointing", "true",
            "--bf16", "true",
        ]
        
        if self.config.use_lora:
            cmd.extend([
                "--train_type", "lora",  # 新版 Swift 使用 --train_type 而不是 --sft_type
                "--lora_rank", str(self.config.lora_rank),
                "--lora_alpha", str(self.config.lora_alpha),
            ])
        
        print("\n[训练] 执行 Swift GRPO 训练...")
        print(f"[命令] {' '.join(cmd)}")
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print(f"\n[完成] 训练完成，模型保存在: {self.config.output_dir}")
        else:
            print(f"\n[错误] 训练失败，返回码: {result.returncode}")
        
        return self.config.output_dir
    
    def train_with_swift_api(self, train_data_path: str):
        """
        使用 Swift Python API 进行训练
        
        这种方式更灵活，可以自定义训练过程
        """
        try:
            from swift.llm import get_model_tokenizer, get_template
            from swift.llm.argument import RLHFArguments
            from swift.llm.rlhf import rlhf_main
        except ImportError:
            print("[错误] 无法导入 Swift，请先安装: pip install ms-swift")
            return None
        
        # 创建参数
        args = RLHFArguments(
            model=self.config.model_path,
            rlhf_type="grpo",
            dataset=train_data_path,
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_length=self.config.max_length,
            beta=self.config.beta,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            gradient_checkpointing=True,
            bf16=True,
            train_type="lora" if self.config.use_lora else "full",  # 新版 Swift 使用 train_type
            lora_rank=self.config.lora_rank if self.config.use_lora else None,
            lora_alpha=self.config.lora_alpha if self.config.use_lora else None,
        )
        
        print("\n[训练] 使用 Swift API 进行 GRPO 训练...")
        
        # 运行训练
        rlhf_main(args)
        
        print(f"\n[完成] 训练完成，模型保存在: {self.config.output_dir}")
        
        return self.config.output_dir
    
    def train_manual_grpo(
        self,
        trajectories: List[VideoTrajectory],
        num_epochs: int = None
    ):
        """
        手动实现 GRPO 训练循环
        
        这种方式最灵活，完全控制训练过程
        适用于 Swift 框架不支持的场景
        """
        import torch
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        if num_epochs is None:
            num_epochs = self.config.num_train_epochs
        
        print("\n[训练] 手动 GRPO 训练...")
        
        # 加载模型
        from modelscope import AutoModelForCausalLM, AutoProcessor
        
        processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果使用 LoRA
        if self.config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        model.train()
        
        # 准备数据
        samples = convert_trajectories_to_grpo_samples(trajectories)
        
        # 优化器
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        
        total_steps = len(samples) * num_epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # 训练循环
        global_step = 0
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
            
            epoch_loss = 0.0
            
            for i, sample in enumerate(samples):
                # 计算 GRPO Loss
                loss = self._compute_grpo_loss(
                    model, processor, sample
                )
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / (i + 1)
                        print(f"    Step {global_step}, Loss: {avg_loss:.4f}")
                    
                    if global_step % self.config.save_steps == 0:
                        save_path = osp.join(
                            self.config.output_dir,
                            f"checkpoint-{global_step}"
                        )
                        model.save_pretrained(save_path)
                        processor.save_pretrained(save_path)
                        print(f"    [保存] Checkpoint: {save_path}")
            
            print(f"[Epoch {epoch + 1}] Average Loss: {epoch_loss / len(samples):.4f}")
        
        # 保存最终模型
        final_path = osp.join(self.config.output_dir, "final")
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        print(f"\n[完成] 最终模型保存到: {final_path}")
        
        return final_path
    
    def _compute_grpo_loss(
        self,
        model,
        processor,
        sample: GRPOSample
    ) -> 'torch.Tensor':
        """计算单个样本的 GRPO Loss"""
        import torch
        import torch.nn.functional as F
        
        device = next(model.parameters()).device
        
        # 获取 rewards 并找到最佳和最差响应
        rewards = torch.tensor(sample.rewards, dtype=torch.float32)
        best_idx = rewards.argmax().item()
        worst_idx = rewards.argmin().item()
        
        if best_idx == worst_idx:
            # 如果所有 reward 相同，返回零 loss
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        # 构建输入
        query = sample.query
        chosen_response = sample.responses[best_idx]
        rejected_response = sample.responses[worst_idx]
        
        # 编码 chosen
        chosen_text = f"{query}\n{chosen_response}"
        chosen_inputs = processor(
            text=[chosen_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(device)
        
        # 编码 rejected
        rejected_text = f"{query}\n{rejected_response}"
        rejected_inputs = processor(
            text=[rejected_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(device)
        
        # 前向传播
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            chosen_outputs = model(**chosen_inputs, labels=chosen_inputs["input_ids"])
            rejected_outputs = model(**rejected_inputs, labels=rejected_inputs["input_ids"])
        
        # 计算 log probabilities
        chosen_logprob = -chosen_outputs.loss
        rejected_logprob = -rejected_outputs.loss
        
        # GRPO/DPO Loss
        # loss = -log(sigmoid(beta * (chosen_logprob - rejected_logprob)))
        logits_diff = self.config.beta * (chosen_logprob - rejected_logprob)
        loss = -F.logsigmoid(logits_diff)
        
        return loss

    def train_grpo_with_importance_sampling(
        self,
        trajectories: List[VideoTrajectory],
        num_epochs: int = None
    ):
        """
        带重要性采样和 PPO-Clip 的 GRPO 训练
        
        核心改进:
        1. 使用预计算的 reward（来自 Phase 1/2）
        2. 重要性采样：ratio = exp(current_logprob - old_logprob)
        3. PPO-Clip：clamp(ratio, 1-ε, 1+ε)
        4. 可选的 KL 惩罚
        
        Args:
            trajectories: 视频轨迹列表（包含 sampled_paths, old_log_probs, rewards）
            num_epochs: 训练轮数
        """
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        if num_epochs is None:
            num_epochs = self.config.num_train_epochs
        
        print("\n" + "="*60)
        print("[GRPO] 带重要性采样和 PPO-Clip 的 GRPO 训练")
        print("="*60)
        print(f"  - 重要性采样: {self.config.use_importance_sampling}")
        print(f"  - PPO Clip ε: {self.config.ppo_clip_epsilon}")
        print(f"  - KL 惩罚: {self.config.use_kl_penalty} (coef={self.config.kl_penalty_coef})")
        print(f"  - Beta: {self.config.beta}")
        print("="*60)
        
        # 加载模型
        from modelscope import AutoModelForCausalLM, AutoProcessor
        
        print("\n[加载] 加载模型...")
        processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果使用 LoRA
        if self.config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        model.train()
        device = next(model.parameters()).device
        
        # 准备数据
        samples = convert_trajectories_to_grpo_samples(trajectories)
        print(f"[数据] 共 {len(samples)} 个训练样本")
        
        # 检查是否有 old_log_probs
        has_old_log_probs = any(len(s.old_log_probs) > 0 for s in samples)
        if self.config.use_importance_sampling and not has_old_log_probs:
            print("[警告] 启用了重要性采样但没有 old_log_probs，将使用 0.0 作为默认值")
        
        # 优化器
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        
        total_steps = len(samples) * num_epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # 训练循环
        global_step = 0
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
            
            epoch_loss = 0.0
            epoch_clip_ratio = 0.0
            epoch_kl = 0.0
            num_samples_processed = 0
            
            for i, sample in enumerate(samples):
                # 计算带重要性采样的 GRPO Loss
                loss, metrics = self._compute_grpo_loss_with_importance_sampling(
                    model, processor, sample, device
                )
                
                if loss is None:
                    continue  # 跳过无效样本
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                epoch_clip_ratio += metrics.get("clip_fraction", 0.0)
                epoch_kl += metrics.get("kl_divergence", 0.0)
                num_samples_processed += 1
                
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / max(num_samples_processed, 1)
                        avg_clip = epoch_clip_ratio / max(num_samples_processed, 1)
                        avg_kl = epoch_kl / max(num_samples_processed, 1)
                        print(f"    Step {global_step}, Loss: {avg_loss:.4f}, "
                              f"Clip%: {avg_clip:.2%}, KL: {avg_kl:.4f}")
                    
                    if global_step % self.config.save_steps == 0:
                        save_path = osp.join(
                            self.config.output_dir,
                            f"checkpoint-{global_step}"
                        )
                        model.save_pretrained(save_path)
                        processor.save_pretrained(save_path)
                        print(f"    [保存] Checkpoint: {save_path}")
            
            avg_epoch_loss = epoch_loss / max(num_samples_processed, 1)
            print(f"[Epoch {epoch + 1}] Average Loss: {avg_epoch_loss:.4f}")
        
        # 保存最终模型
        final_path = osp.join(self.config.output_dir, "final")
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        print(f"\n[完成] 最终模型保存到: {final_path}")
        
        return final_path
    
    def _compute_grpo_loss_with_importance_sampling(
        self,
        model,
        processor,
        sample: GRPOSample,
        device
    ) -> Tuple[Optional['torch.Tensor'], Dict[str, float]]:
        """
        计算带重要性采样和 PPO-Clip 的 GRPO Loss
        
        GRPO Loss 公式:
        L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
        
        其中:
        - ratio = exp(π_θ(a|s) / π_old(a|s)) = exp(new_logprob - old_logprob)
        - A = advantage = R_chosen - baseline (这里用 R_chosen - R_mean)
        
        Returns:
            Tuple[loss, metrics_dict]
        """
        import torch
        import torch.nn.functional as F
        
        metrics = {
            "clip_fraction": 0.0,
            "kl_divergence": 0.0,
            "importance_ratio": 1.0
        }
        
        # 获取 rewards 并找到最佳响应
        rewards = torch.tensor(sample.rewards, dtype=torch.float32)
        
        if len(rewards) < 2:
            return None, metrics
        
        # 计算 advantage（相对于均值的优势）
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        advantages = (rewards - reward_mean) / reward_std  # 归一化 advantage
        
        # 找到最佳和最差响应
        best_idx = rewards.argmax().item()
        worst_idx = rewards.argmin().item()
        
        if best_idx == worst_idx:
            return None, metrics
        
        # 获取 old log probs
        old_log_probs = sample.old_log_probs if sample.old_log_probs else [0.0] * len(sample.responses)
        
        # 构建输入 - chosen response
        query = sample.query
        chosen_response = sample.responses[best_idx]
        chosen_text = f"{query}\n{chosen_response}"
        
        try:
            chosen_inputs = processor(
                text=[chosen_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(device)
        except Exception as e:
            print(f"[警告] 处理输入失败: {e}")
            return None, metrics
        
        # 前向传播获取当前 log probability
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**chosen_inputs, labels=chosen_inputs["input_ids"])
            new_log_prob = -outputs.loss  # 负的 loss 就是 log prob
        
        # 获取 old log prob
        old_log_prob = old_log_probs[best_idx] if best_idx < len(old_log_probs) else 0.0
        old_log_prob_tensor = torch.tensor(old_log_prob, dtype=torch.float32, device=device)
        
        # 计算重要性比率
        if self.config.use_importance_sampling:
            log_ratio = new_log_prob - old_log_prob_tensor
            ratio = torch.exp(log_ratio)
            metrics["importance_ratio"] = ratio.item()
        else:
            ratio = torch.tensor(1.0, device=device)
        
        # 获取 advantage
        advantage = advantages[best_idx].to(device)
        
        # PPO-Clip
        epsilon = self.config.ppo_clip_epsilon
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        
        # 计算两种 objective
        surrogate1 = ratio * advantage
        surrogate2 = clipped_ratio * advantage
        
        # 取最小值（悲观估计）
        policy_loss = -torch.min(surrogate1, surrogate2)
        
        # 统计被截断的比例
        is_clipped = (ratio < (1 - epsilon)) | (ratio > (1 + epsilon))
        metrics["clip_fraction"] = is_clipped.float().item()
        
        # 可选的 KL 惩罚
        if self.config.use_kl_penalty:
            # KL divergence ≈ (ratio - 1) - log(ratio)
            kl_divergence = (ratio - 1) - torch.log(ratio + 1e-8)
            kl_divergence = torch.clamp(kl_divergence, min=0.0)
            metrics["kl_divergence"] = kl_divergence.item()
            
            policy_loss = policy_loss + self.config.kl_penalty_coef * kl_divergence
        
        # 添加 rejected response 的对比损失（可选，增强区分度）
        if len(sample.responses) > 1:
            rejected_response = sample.responses[worst_idx]
            rejected_text = f"{query}\n{rejected_response}"
            
            try:
                rejected_inputs = processor(
                    text=[rejected_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                ).to(device)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    rejected_outputs = model(**rejected_inputs, labels=rejected_inputs["input_ids"])
                    rejected_log_prob = -rejected_outputs.loss
                
                # DPO 风格的对比损失
                logits_diff = self.config.beta * (new_log_prob - rejected_log_prob)
                preference_loss = -F.logsigmoid(logits_diff)
                
                # 组合损失：PPO policy loss + DPO preference loss
                total_loss = 0.5 * policy_loss + 0.5 * preference_loss
                
            except Exception as e:
                total_loss = policy_loss
        else:
            total_loss = policy_loss
        
        return total_loss, metrics
    
    def train_with_swift_api_custom_reward(
        self,
        trajectories: List[VideoTrajectory],
        num_epochs: int = None
    ) -> str:
        """
        使用 Swift API + 自定义 Reward Function 的 GRPO 训练
        
        这种方式结合了：
        1. Swift 框架的优化（分布式、混合精度、梯度累积等）
        2. 我们预计算的 reward（来自 Phase 1/2）
        3. 重要性采样修正
        
        核心思路：
        - 构建 reward 查找表，将预计算的 reward 映射到样本
        - 定义自定义 reward_func，Swift 调用时返回预计算值
        - 利用 Swift 的 GRPO 训练流程
        
        Args:
            trajectories: 视频轨迹列表（包含 sampled_paths, old_log_probs, rewards）
            num_epochs: 训练轮数
            
        Returns:
            训练后模型保存路径
        """
        import torch
        from typing import List as TList, Optional as TOptional, Dict as TDict
        
        if num_epochs is None:
            num_epochs = self.config.num_train_epochs
        
        print("\n" + "="*60)
        print("[GRPO] 使用 Swift API + 自定义 Reward Function")
        print("="*60)
        print(f"  - 重要性采样: {self.config.use_importance_sampling}")
        print(f"  - PPO Clip ε: {self.config.ppo_clip_epsilon}")
        print(f"  - Beta: {self.config.beta}")
        print("="*60)
        
        # ========== 1. 构建 reward 查找表和训练数据 ==========
        print("\n[准备] 构建 reward 查找表...")
        
        reward_lookup = {}  # key: (video_id, window_idx, path_idx) -> reward
        old_logprob_lookup = {}  # key: (video_id, window_idx, path_idx) -> old_log_prob
        sample_data = []
        
        for traj in trajectories:
            for window in traj.windows:
                # 构建 prompt
                prompt = self._build_window_prompt_for_swift(window)
                
                # 获取 rewards
                rewards = window.r_total_scores if window.r_total_scores else window.r_teacher_scores
                old_log_probs = window.old_log_probs if window.old_log_probs else []
                
                for i, path in enumerate(window.sampled_paths):
                    # 响应文本
                    response = str(path)  # e.g., "['MERGE', 'CREATE_NEW', 'MERGE']"
                    
                    # 获取 reward
                    reward = rewards[i] if i < len(rewards) else 0.0
                    
                    # 获取 old_log_prob
                    old_log_prob = old_log_probs[i] if i < len(old_log_probs) else 0.0
                    
                    # 存储到查找表
                    key = (traj.video_id, window.window_idx, i)
                    reward_lookup[key] = reward
                    old_logprob_lookup[key] = old_log_prob
                    
                    # 添加到训练数据
                    sample_data.append({
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "old_log_prob": old_log_prob,
                        "video_id": traj.video_id,
                        "window_idx": window.window_idx,
                        "path_idx": i
                    })
        
        print(f"[数据] 构建了 {len(reward_lookup)} 个 reward 映射")
        print(f"[数据] 共 {len(sample_data)} 个训练样本")
        
        # ========== 2. 定义自定义 reward function ==========
        # 这个闭包会捕获 reward_lookup 和 old_logprob_lookup
        
        def precomputed_reward_func(
            completions: TList[str],
            prompts: TOptional[TList[str]] = None,
            **kwargs
        ) -> TList[float]:
            """
            自定义 reward function
            
            Swift 在评估生成结果时会调用此函数
            我们直接返回预计算的 reward 值
            
            Args:
                completions: 模型生成的响应列表
                prompts: 对应的 prompt 列表
                **kwargs: 可能包含额外信息
                
            Returns:
                每个 completion 对应的 reward 值
            """
            rewards = []
            
            for i, completion in enumerate(completions):
                matched = False
                
                # 尝试通过响应文本匹配
                for data in sample_data:
                    # 检查响应是否匹配
                    if data["response"] == completion or completion.strip() == data["response"].strip():
                        rewards.append(data["reward"])
                        matched = True
                        break
                    
                    # 模糊匹配（处理格式差异）
                    if not matched:
                        # 尝试解析动作列表
                        try:
                            import re
                            comp_actions = re.findall(r'(?:CREATE_NEW|MERGE|DISCARD|UPDATE)', completion.upper())
                            data_actions = re.findall(r'(?:CREATE_NEW|MERGE|DISCARD|UPDATE)', data["response"].upper())
                            if comp_actions and comp_actions == data_actions:
                                rewards.append(data["reward"])
                                matched = True
                                break
                        except:
                            pass
                
                if not matched:
                    # 未找到匹配，使用默认值（中性 reward）
                    all_rewards = [d["reward"] for d in sample_data]
                    default_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
                    rewards.append(default_reward)
            
            return rewards
        
        # ========== 3. 保存 Swift 格式训练数据 ==========
        print("\n[保存] 准备 Swift 格式训练数据...")
        
        # 按 prompt 分组，Swift GRPO 格式需要每个 prompt 对应多个 response
        from collections import defaultdict
        grouped_data = defaultdict(list)
        
        for data in sample_data:
            grouped_data[data["prompt"]].append({
                "response": data["response"],
                "reward": data["reward"],
                "old_log_prob": data["old_log_prob"]
            })
        
        swift_format_data = []
        for prompt, responses_data in grouped_data.items():
            swift_format_data.append({
                "query": prompt,
                "response": [r["response"] for r in responses_data],
                "reward": [r["reward"] for r in responses_data],
                # 存储 old_log_probs 用于重要性采样
                "old_log_probs": [r["old_log_prob"] for r in responses_data]
            })
        
        train_data_path = osp.join(self.config.output_dir, "grpo_train_data_with_logprobs.json")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        with open(train_data_path, 'w', encoding='utf-8') as f:
            json.dump(swift_format_data, f, ensure_ascii=False, indent=2)
        
        print(f"[保存] 训练数据: {train_data_path}")
        print(f"[信息] 共 {len(swift_format_data)} 个 unique prompts")
        
        # ========== 4. 尝试使用 Swift API ==========
        print("\n[训练] 尝试使用 Swift API...")
        
        try:
            from swift.llm import rlhf_main
            from swift.llm.argument import RLHFArguments
            
            # 创建 Swift RLHF 参数
            # 注意：Swift RLHFArguments 不支持 lora_target_modules，使用默认值
            args = RLHFArguments(
                model=self.config.model_path,
                rlhf_type="grpo",
                dataset=[train_data_path],
                output_dir=self.config.output_dir,
                
                # 训练参数
                num_train_epochs=num_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                max_length=self.config.max_length,
                
                # GRPO 参数
                beta=self.config.beta,
                num_generations=self.config.num_generations,
                
                # 关键：传入自定义 reward function
                reward_funcs=[precomputed_reward_func],
                
                # LoRA 配置 (Swift 使用默认 target_modules)
                train_type="lora" if self.config.use_lora else "full",
                lora_rank=self.config.lora_rank if self.config.use_lora else None,
                lora_alpha=self.config.lora_alpha if self.config.use_lora else None,
                lora_dropout=self.config.lora_dropout if self.config.use_lora else None,
                # lora_target_modules 由 Swift 自动选择，不需要手动指定
                
                # 其他
                seed=self.config.seed,
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                gradient_checkpointing=True,
                bf16=True,
            )
            
            print("[训练] 开始 Swift GRPO 训练...")
            
            # 运行训练
            result = rlhf_main(args)
            
            print(f"\n[完成] 模型保存到: {self.config.output_dir}")
            return self.config.output_dir
            
        except ImportError as e:
            print(f"[警告] Swift 导入失败: {e}")
            print("[回退] 使用手动实现的 GRPO 训练...")
            return self.train_grpo_with_importance_sampling(trajectories, num_epochs)
            
        except TypeError as e:
            # Swift 可能不支持 reward_funcs 参数
            if "reward_funcs" in str(e):
                print(f"[警告] Swift 不支持 reward_funcs 参数: {e}")
                print("[回退] 尝试使用 Swift CLI 方式...")
                return self._train_swift_cli_with_precomputed_rewards(train_data_path)
            else:
                raise
            
        except Exception as e:
            print(f"[错误] Swift 训练失败: {e}")
            print("[回退] 使用手动实现的 GRPO 训练...")
            return self.train_grpo_with_importance_sampling(trajectories, num_epochs)
    
    def _build_window_prompt_for_swift(self, window: WindowTrajectory) -> str:
        """为 Swift 构建窗口的 prompt"""
        scenes_text = ""
        for i, (scene_idx, text) in enumerate(zip(window.window_l1_scene_indices, window.window_l1_texts)):
            scenes_text += f"Scene {scene_idx}: {text}\n"
        
        prompt = (
            "You are a video memory agent organizing scenes into semantic groups.\n\n"
            f"Previous L2 memory: {window.prev_l2_text or 'None'}\n"
            f"Current L2 memory: {window.current_l2_text or 'None'}\n"
            f"New scenes to process:\n{scenes_text}\n"
            "For each scene, choose one action:\n"
            "- CREATE_NEW: Start a new semantic memory (L2)\n"
            "- MERGE: Merge into current L2\n"
            "- DISCARD: Skip (noise/unimportant)\n"
            "- UPDATE: Update current L2 description\n\n"
            "Output the actions as a list, e.g., ['MERGE', 'CREATE_NEW', 'MERGE']"
        )
        
        return prompt
    
    def _train_swift_cli_with_precomputed_rewards(self, train_data_path: str) -> str:
        """
        使用 Swift CLI 训练，数据中已包含预计算的 reward
        
        这是 Swift 不支持 reward_funcs 时的备用方案
        Swift 会直接使用数据中的 reward 字段
        """
        import subprocess
        
        print("\n[训练] 使用 Swift CLI + 预计算 rewards...")
        
        cmd = [
            "swift", "rlhf",
            "--rlhf_type", "grpo",
            "--model", self.config.model_path,
            "--dataset", train_data_path,
            "--output_dir", self.config.output_dir,
            "--num_train_epochs", str(self.config.num_train_epochs),
            "--per_device_train_batch_size", str(self.config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps),
            "--learning_rate", str(self.config.learning_rate),
            "--warmup_ratio", str(self.config.warmup_ratio),
            "--max_length", str(self.config.max_length),
            "--beta", str(self.config.beta),
            "--save_steps", str(self.config.save_steps),
            "--logging_steps", str(self.config.logging_steps),
            "--gradient_checkpointing", "true",
            "--bf16", "true",
        ]
        
        if self.config.use_lora:
            cmd.extend([
                "--train_type", "lora",
                "--lora_rank", str(self.config.lora_rank),
                "--lora_alpha", str(self.config.lora_alpha),
            ])
        
        print(f"[命令] {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print(f"\n[完成] 训练完成，模型保存在: {self.config.output_dir}")
        else:
            print(f"\n[错误] 训练失败，返回码: {result.returncode}")
            print("[回退] 使用手动实现...")
            # 需要重新加载 trajectories，这里无法直接调用
            # 返回当前路径，让调用方处理
        
        return self.config.output_dir
    
    def train(
        self,
        trajectories: List[VideoTrajectory],
        method: str = "importance_sampling",
        num_epochs: int = None
    ):
        """
        统一的训练入口
        
        Args:
            trajectories: 视频轨迹列表
            method: 训练方法
                - "importance_sampling": 带重要性采样和 PPO-Clip 的手动 GRPO（推荐）
                - "swift_custom_reward": 使用 Swift API + 自定义 reward function（推荐）
                - "manual": 简单的手动 GRPO（无重要性采样）
                - "swift_cli": 使用 Swift CLI
                - "swift_api": 使用 Swift API
            num_epochs: 训练轮数
        """
        print(f"\n[训练] 使用方法: {method}")
        
        if method == "importance_sampling":
            return self.train_grpo_with_importance_sampling(trajectories, num_epochs)
        elif method == "swift_custom_reward":
            return self.train_with_swift_api_custom_reward(trajectories, num_epochs)
        elif method == "manual":
            return self.train_manual_grpo(trajectories, num_epochs)
        elif method == "swift_cli":
            train_data_path = self.prepare_training_data(trajectories)
            return self.train_with_swift_cli(train_data_path)
        elif method == "swift_api":
            train_data_path = self.prepare_training_data(trajectories)
            return self.train_with_swift_api(train_data_path)
        else:
            raise ValueError(f"未知的训练方法: {method}")
