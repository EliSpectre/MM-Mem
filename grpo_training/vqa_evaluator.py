"""
Phase 2: VQA 评估器

功能:
1. 读取 QA JSON 文件
2. 检索对应视频的记忆
3. 进行 VQA 问答
4. 计算 R_vqa 并回溯到所有窗口
"""

import os
import os.path as osp
import json
import glob
from typing import List, Dict, Tuple, Optional, Any

from .data_structures import VideoTrajectory, QAResult, WindowTrajectory


class VQAEvaluator:
    """VQA 评估器"""
    
    def __init__(
        self,
        # 记忆检索 VQA 系统配置
        memory_dir: str = "./output/memories",
        api_base_url: str = "http://localhost:8002/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen2.5-VL-7B-Instruct",
        
        # Embedding/Rerank 配置
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        rerank_model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        
        # 检索配置
        top_k_l2: int = 5,
        top_k_l1: int = 3,
        
        # 奖励配置
        alpha: float = 0.5,  # R_vqa 的权重
    ):
        # 确保 memory_dir 是绝对路径
        self.memory_dir = osp.abspath(memory_dir)
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name
        self.device = device
        self.top_k_l2 = top_k_l2
        self.top_k_l1 = top_k_l1
        self.alpha = alpha
        
        print(f"[VQAEvaluator] memory_dir 使用绝对路径: {self.memory_dir}")
        
        # 延迟加载检索系统
        self._retrieval_system = None
    
    def _load_retrieval_system(self):
        """延迟加载记忆检索 VQA 系统"""
        if self._retrieval_system is not None:
            return
        
        import sys
        sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
        
        from memory_retrieval_vqa import MemoryRetrievalVQA
        
        self._retrieval_system = MemoryRetrievalVQA(
            memory_dir=self.memory_dir,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            embedding_model_name=self.embedding_model_name,
            rerank_model_name=self.rerank_model_name,
            device=self.device
        )
        print("[完成] 记忆检索 VQA 系统加载成功")
    
    def load_qa_json(self, qa_json_path: str) -> Dict[str, Dict]:
        """
        加载 QA JSON 文件
        
        JSON 格式:
        {
            "qa_id_1": {
                "inputs": {"video 1": {"id": "video_id"}},
                "question": "...",
                "choices": ["A", "B", "C", "D"],
                "correct_idx": 0
            },
            ...
        }
        """
        with open(qa_json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        print(f"[加载] 从 {qa_json_path} 加载了 {len(qa_data)} 个问题")
        return qa_data
    
    def load_qa_from_folder(self, qa_folder: str) -> Dict[str, Dict]:
        """从文件夹加载所有 QA JSON 文件"""
        all_qa = {}
        
        json_files = glob.glob(osp.join(qa_folder, "*.json"))
        print(f"[加载] 在 {qa_folder} 发现 {len(json_files)} 个 JSON 文件")
        
        for json_file in json_files:
            try:
                qa_data = self.load_qa_json(json_file)
                all_qa.update(qa_data)
            except Exception as e:
                print(f"[警告] 加载 {json_file} 失败: {e}")
        
        print(f"[完成] 共加载 {len(all_qa)} 个问题")
        return all_qa
    
    def _extract_video_id_from_qa(self, qa_item: Dict) -> Optional[str]:
        """从 QA 项中提取视频 ID"""
        inputs = qa_item.get("inputs", {})
        
        # 尝试从 "video 1" 获取
        for key in inputs:
            if "video" in key.lower():
                video_info = inputs[key]
                if isinstance(video_info, dict) and "id" in video_info:
                    return video_info["id"]
        
        return None
    
    def _format_choices(self, choices: List[str]) -> List[str]:
        """格式化选项"""
        formatted = []
        letters = "ABCDEFGHIJ"
        
        for i, choice in enumerate(choices):
            if i < len(letters):
                # 检查是否已经有字母前缀
                if not choice.strip().startswith(letters[i]):
                    formatted.append(f"{letters[i]}. {choice}")
                else:
                    formatted.append(choice)
            else:
                formatted.append(choice)
        
        return formatted
    
    def evaluate_single_qa(
        self,
        qa_id: str,
        qa_item: Dict,
        verbose: bool = False
    ) -> QAResult:
        """评估单个问题"""
        self._load_retrieval_system()
        
        video_id = self._extract_video_id_from_qa(qa_item)
        if not video_id:
            print(f"[警告] 问题 {qa_id} 无法提取视频 ID")
            return QAResult(
                qa_id=qa_id,
                video_id="",
                question=qa_item.get("question", ""),
                choices=qa_item.get("choices", []),
                correct_idx=qa_item.get("correct_idx", -1),
                is_correct=False
            )
        
        question = qa_item.get("question", "")
        choices = qa_item.get("choices", [])
        correct_idx = qa_item.get("correct_idx", -1)
        
        # 格式化选项
        formatted_choices = self._format_choices(choices)
        
        if verbose:
            print(f"\n[评估] QA: {qa_id}")
            print(f"    Video: {video_id}")
            print(f"    Question: {question[:80]}...")
        
        try:
            # 检索并回答
            result = self._retrieval_system.retrieve_and_answer(
                video_id=video_id,
                question_text=question,
                options=formatted_choices,
                top_k_l2=self.top_k_l2,
                top_k_l1=self.top_k_l1,
                verbose=False
            )
            
            answer_info = result.get('answer', {})
            model_answer = answer_info.get('normalized_answer', '')
            confidence = answer_info.get('confidence', 0.0)
            
            # 将答案转换为索引
            model_answer_idx = -1
            if model_answer in "ABCDEFGHIJ":
                model_answer_idx = ord(model_answer) - ord('A')
            
            is_correct = (model_answer_idx == correct_idx)
            
            if verbose:
                print(f"    Model Answer: {model_answer} (idx={model_answer_idx})")
                print(f"    Correct: {correct_idx}")
                print(f"    Is Correct: {is_correct}")
            
            return QAResult(
                qa_id=qa_id,
                video_id=video_id,
                question=question,
                choices=choices,
                correct_idx=correct_idx,
                model_answer=model_answer,
                model_answer_idx=model_answer_idx,
                is_correct=is_correct,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"[错误] 评估 {qa_id} 失败: {e}")
            return QAResult(
                qa_id=qa_id,
                video_id=video_id,
                question=question,
                choices=choices,
                correct_idx=correct_idx,
                is_correct=False
            )
    
    def evaluate_all_qa(
        self,
        qa_data: Dict[str, Dict],
        verbose: bool = False
    ) -> Tuple[List[QAResult], Dict[str, float]]:
        """
        评估所有问题
        
        Returns:
            (qa_results, video_accuracies): QA 结果列表，每个视频的准确率
        """
        qa_results = []
        video_qa_counts = {}  # video_id -> (correct, total)
        
        total = len(qa_data)
        for i, (qa_id, qa_item) in enumerate(qa_data.items()):
            if (i + 1) % 10 == 0:
                print(f"[进度] {i+1}/{total}")
            
            result = self.evaluate_single_qa(qa_id, qa_item, verbose=verbose)
            qa_results.append(result)
            
            # 统计每个视频的准确率
            vid = result.video_id
            if vid:
                if vid not in video_qa_counts:
                    video_qa_counts[vid] = [0, 0]
                video_qa_counts[vid][1] += 1
                if result.is_correct:
                    video_qa_counts[vid][0] += 1
        
        # 计算每个视频的准确率
        video_accuracies = {}
        for vid, (correct, total) in video_qa_counts.items():
            video_accuracies[vid] = correct / total if total > 0 else 0.0
        
        # 输出统计
        total_correct = sum(1 for r in qa_results if r.is_correct)
        overall_acc = total_correct / len(qa_results) if qa_results else 0.0
        
        print(f"\n[统计] 总体准确率: {total_correct}/{len(qa_results)} = {overall_acc:.2%}")
        print(f"[统计] 涉及 {len(video_accuracies)} 个视频")
        
        return qa_results, video_accuracies
    
    def update_trajectories_with_vqa_reward(
        self,
        trajectories: List[VideoTrajectory],
        video_accuracies: Dict[str, float],
        qa_results: List[QAResult]
    ) -> List[VideoTrajectory]:
        """
        用 VQA 奖励更新轨迹
        
        对于每个视频：
        - R_vqa = 该视频的 QA 准确率
        - 该视频的所有窗口都加上 α * R_vqa
        """
        # 按视频 ID 组织 QA 结果
        video_qa_map = {}
        for qa_result in qa_results:
            vid = qa_result.video_id
            if vid:
                if vid not in video_qa_map:
                    video_qa_map[vid] = []
                video_qa_map[vid].append(qa_result)
        
        for traj in trajectories:
            vid = traj.video_id
            
            # 获取 R_vqa
            r_vqa = video_accuracies.get(vid, 0.0)
            traj.r_vqa = r_vqa
            
            # 添加 QA 结果
            traj.qa_results = video_qa_map.get(vid, [])
            
            # 更新每个窗口的 R_total
            for window in traj.windows:
                window.r_vqa = r_vqa
                
                # R_total = R_teacher + α * R_vqa
                window.r_total_scores = [
                    r_teacher + self.alpha * r_vqa
                    for r_teacher in window.r_teacher_scores
                ]
            
            print(f"[更新] Video {vid}: R_vqa = {r_vqa:.4f}, "
                  f"更新了 {len(traj.windows)} 个窗口")
        
        return trajectories
    
    def run_evaluation(
        self,
        trajectories: List[VideoTrajectory],
        qa_data: Dict[str, Dict],
        output_path: str = None,
        verbose: bool = False
    ) -> List[VideoTrajectory]:
        """
        运行完整的 VQA 评估流程
        
        Args:
            trajectories: 视频轨迹列表
            qa_data: QA 数据
            output_path: 更新后轨迹的保存路径
            verbose: 是否详细输出
            
        Returns:
            更新后的轨迹列表
        """
        print("\n" + "="*60)
        print("[Phase 2] VQA 评估")
        print("="*60)
        
        # Step 1: 评估所有 QA
        qa_results, video_accuracies = self.evaluate_all_qa(qa_data, verbose=verbose)
        
        # Step 2: 更新轨迹
        trajectories = self.update_trajectories_with_vqa_reward(
            trajectories, video_accuracies, qa_results
        )
        
        # Step 3: 保存更新后的轨迹
        if output_path:
            all_data = {
                "video_accuracies": video_accuracies,
                "qa_results": [r.to_dict() for r in qa_results],
                "trajectories": [t.to_dict() for t in trajectories]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n[保存] 评估结果已保存到: {output_path}")
        
        return trajectories
    
    def load_evaluated_trajectories(self, path: str) -> Tuple[List[VideoTrajectory], Dict]:
        """加载已评估的轨迹数据"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trajectories = [VideoTrajectory.from_dict(t) for t in data.get("trajectories", [])]
        video_accuracies = data.get("video_accuracies", {})
        
        return trajectories, video_accuracies
