# merge_lora.py
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# 路径配置
base_model_path = "/data/tempuser2/MMAgent/Qwen-VL/qwen_models/Qwen3-VL-2B-Instruct"
lora_path = "/data/tempuser2/MMAgent/MM-Mem/output/grpo_training/grpo_checkpoints/v2-20251223-160524/checkpoint-3"
merged_output_path = "/data/tempuser2/MMAgent/MM-Mem/output/grpo_training/merged_model"

# 加载 base model
print("加载 base model...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 LoRA adapter
print("加载 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_path)

# 合并
print("合并 LoRA...")
merged_model = model.merge_and_unload()

# 保存合并后的模型
print(f"保存到: {merged_output_path}")
merged_model.save_pretrained(merged_output_path)

# 复制 tokenizer/processor
processor = AutoProcessor.from_pretrained(base_model_path)
processor.save_pretrained(merged_output_path)

print("完成!")