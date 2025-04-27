#python baseline_eval.py --dataset_name aime --model_size 32b

import os
import re
import pickle
import argparse
import logging
import numpy as np
from collections import defaultdict
from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_final_answer(reasoning, dataset_name):
    """
    从推理文本中提取最终答案
    """
    if not reasoning:
        return None
        
    # 使用正则表达式查找 \boxed{answer} 模式
    match = re.search(r"\\boxed\{(.+?)\}", reasoning)
    if match:
        answer = match.group(1).strip()
        # 对于GPQA，我们期望A、B、C或D
        if dataset_name == "gpqa":
            if answer in ["A", "B", "C", "D"]:
                return answer
            else:
                # 尝试从完整答案中找到选项字母
                sub_match = re.match(r"([A-D])[\)\.]?", answer)
                if sub_match:
                    return sub_match.group(1)
                logging.warning(f"无法从boxed答案中提取有效的GPQA选项: {answer}")
                return None
        # 对于AIME/MATH，尝试转换为数字
        else:
            try:
                # 首先尝试简单的整数转换
                return int(answer)
            except ValueError:
                # 添加更复杂的解析（如分数等）
                logging.warning(f"无法将boxed答案转换为整数: {answer}")
                return answer  # 如果转换失败，则返回字符串
    else:
        # 备用方案：检查最后一步是否包含"Answer: X"（用于GPQA）
        if dataset_name == "gpqa":
            match_answer = re.search(r"[Aa]nswer:\s*([A-D])", reasoning)
            if match_answer:
                return match_answer.group(1).upper()

        logging.warning(f"在最终步骤中找不到\\boxed{{}}模式")
        return None

def get_ground_truth(dataset, problem_idx, dataset_name):
    """
    从数据集中获取正确答案
    """
    if dataset_name == "aime":
        try:
            # 示例：数据集可能将答案存储为字符串'### 123'
            answer_str = dataset[problem_idx].get("final_answer", "")
            match = re.search(r"(\d+)$", answer_str)
            if match:
                return int(match.group(1))
            else:
                # 如果没有模式，尝试直接转换
                return int(answer_str)
        except (ValueError, TypeError, KeyError) as e:
            logging.error(f"获取AIME索引{problem_idx}的正确答案时出错: {e}. 答案字符串: {dataset[problem_idx].get('final_answer', 'N/A')}")
            return None
    elif dataset_name == "math":
        try:
            solution_str = dataset[problem_idx].get("solution", "")
            match = re.search(r"\\boxed\{(.+?)\}", solution_str)
            if match:
                answer = match.group(1).strip()
                try:
                    # 首先尝试转换为整数
                    return int(answer)
                except ValueError:
                    # 处理分数或其他格式
                    logging.warning(f"MATH正确答案不是简单整数: {answer}")
                    return answer  # 如果不是整数，则返回字符串
            else:
                logging.error(f"无法从MATH索引{problem_idx}的解决方案中提取正确答案: {solution_str}")
                return None
        except (KeyError, TypeError) as e:
            logging.error(f"访问MATH索引{problem_idx}的解决方案时出错: {e}")
            return None
    elif dataset_name == "gpqa":
        try:
            # 找到与正确答案文本对应的字母
            correct_answer_text = dataset[problem_idx].get("Correct Answer")
            options_text = {
                "A": dataset[problem_idx].get("Answer A"),
                "B": dataset[problem_idx].get("Answer B"),
                "C": dataset[problem_idx].get("Answer C"),
                "D": dataset[problem_idx].get("Answer D"),
            }
            for letter, text in options_text.items():
                if text == correct_answer_text:
                    return letter
            logging.error(f"无法为GPQA索引{problem_idx}找到匹配的正确答案字母")
            return None
        except (KeyError, TypeError) as e:
            logging.error(f"获取GPQA索引{problem_idx}的正确答案时出错: {e}")
            return None
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

def compare_answers(model_answer, ground_truth, dataset_name):
    """
    比较模型答案与正确答案
    """
    if model_answer is None or ground_truth is None:
        return False
        
    # 规范化答案进行比较（例如，转换为相同类型）
    try:
        if dataset_name in ["aime", "math"]:
            # 尝试数值比较
            return int(model_answer) == int(ground_truth)
        elif dataset_name == "gpqa":
            # 简单的字符串比较A、B、C、D
            return str(model_answer).upper() == str(ground_truth).upper()
        else:
            return str(model_answer) == str(ground_truth)
    except (ValueError, TypeError):
        # 如果数值转换失败，则回退到字符串比较
        logging.warning(f"{dataset_name}的数值比较失败。以字符串形式比较: '{model_answer}' vs '{ground_truth}'")
        return str(model_answer).strip() == str(ground_truth).strip()

def evaluate_model(args):
    """评估模型在数据集上的表现"""
    logging.info(f"开始评估数据集: {args.dataset_name}")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # --- 加载数据集 ---
    try:
        if args.dataset_name == "aime":
            dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
            id_offset = 60  # AIME问题ID从60开始
        elif args.dataset_name == "math":
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
            id_offset = 0
        elif args.dataset_name == "gpqa":
            if os.getenv("HF_HUB_OFFLINE", "0") == "1":
                dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
            else:    
                dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
            id_offset = 0
        else:
            raise ValueError(f"不支持的数据集: {args.dataset_name}")
        logging.info(f"数据集加载成功。总样本数: {len(dataset)}")
    except Exception as e:
        logging.error(f"加载数据集'{args.dataset_name}'失败: {e}")
        return

    # 检查结果目录是否存在
    model_dir = os.path.join(args.results_dir, args.model_size, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化统计数据
    problem_stats = {}  # 每个问题的统计信息
    total_correct = 0   # 所有正确的次数
    total_attempts = 0  # 所有尝试的次数
    problems_with_correct = 0  # 至少有一次正确的问题数

    # 遍历结果文件
    for filename in os.listdir(model_dir):
        if not filename.endswith(".pickle"):
            continue
            
        # 解析文件名获取问题ID和重复ID
        match = re.match(r"(\d+)-(\d+)\.pickle", filename)
        if not match:
            logging.warning(f"跳过无法解析的文件名: {filename}")
            continue
            
        problem_id = int(match.group(1))
        repeat_id = int(match.group(2))
        
        # 计算数据集中的索引
        problem_idx = problem_id - id_offset
        if problem_idx < 0 or problem_idx >= len(dataset):
            logging.warning(f"问题ID {problem_id}（索引 {problem_idx}）超出数据集大小 {len(dataset)}。跳过。")
            continue
            
        # 加载结果文件
        file_path = os.path.join(model_dir, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            # 提取模型的推理结果
            reasoning = data.get("reasoning", {}).get("reasoning", "")
            model_answer = extract_final_answer(reasoning, args.dataset_name)
            ground_truth = get_ground_truth(dataset, problem_idx, args.dataset_name)
            
            # 检查答案是否正确
            if model_answer is not None and ground_truth is not None:
                is_correct = compare_answers(model_answer, ground_truth, args.dataset_name)
                
                # 更新问题统计信息
                if problem_id not in problem_stats:
                    problem_stats[problem_id] = {"attempts": 0, "correct": 0}
                    
                problem_stats[problem_id]["attempts"] += 1
                if is_correct:
                    problem_stats[problem_id]["correct"] += 1
                    total_correct += 1
                    
                total_attempts += 1
                
                logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}: 模型='{model_answer}'，正确答案='{ground_truth}'，正确={is_correct}")
            else:
                logging.warning(f"无法评估问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}: 缺少模型答案（'{model_answer}'）或正确答案（'{ground_truth}'）。")
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 计算每个问题是否至少有一次正确
    for problem_id, stats in problem_stats.items():
        if stats["correct"] > 0:
            problems_with_correct += 1

    # 计算准确率
    total_problems = len(problem_stats)
    if total_problems > 0:
        accuracy_method1 = (problems_with_correct / total_problems) * 100
    else:
        accuracy_method1 = 0
        
    if total_attempts > 0:
        accuracy_method2 = (total_correct / total_attempts) * 100
    else:
        accuracy_method2 = 0

    # 打印结果
    logging.info("=" * 50)
    logging.info("评估摘要:")
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"模型: {args.model_size}")
    logging.info(f"结果目录: {model_dir}")
    logging.info(f"总问题数: {total_problems}")
    logging.info(f"总尝试次数: {total_attempts}")
    logging.info(f"总正确次数: {total_correct}")
    logging.info(f"至少有一次正确的问题数: {problems_with_correct}")
    logging.info(f"方法1准确率 (至少一次正确的问题/总问题数): {accuracy_method1:.2f}%")
    logging.info(f"方法2准确率 (总正确次数/总尝试次数): {accuracy_method2:.2f}%")
    logging.info("=" * 50)
    
    # 打印每个问题的详细统计信息
    logging.info("每个问题的详细统计信息:")
    for problem_id in sorted(problem_stats.keys()):
        stats = problem_stats[problem_id]
        logging.info(f"问题 {problem_id}: 尝试={stats['attempts']}, 正确={stats['correct']}, 正确率={stats['correct']/stats['attempts']*100:.2f}%")
    
    return {
        "total_problems": total_problems,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "problems_with_correct": problems_with_correct,
        "accuracy_method1": accuracy_method1,
        "accuracy_method2": accuracy_method2,
        "problem_stats": problem_stats
    }

def main():
    parser = argparse.ArgumentParser(description="评估基线模型在数据集上的表现")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], required=True,
                        help="要评估的数据集")
    parser.add_argument("--model_size", type=str, choices=["1.5b", "32b"], required=True,
                        help="要评估的模型大小")
    parser.add_argument("--results_dir", type=str, default="results/baseline_test",
                        help="包含结果文件的目录")
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()