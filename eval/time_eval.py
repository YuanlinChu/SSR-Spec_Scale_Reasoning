#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python time_eval.py --dataset_name aime --model_name QwQ-32B --results_dir results/baseline_vllm_test
# python time_eval.py --dataset_name aime --results_dir results/spec_scale_m5 --spec_scale

import os
import re
import pickle
import argparse
import logging
import numpy as np
from collections import defaultdict
from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_baseline_time(args):
    """评估baseline模型的时间消耗"""
    logging.info(f"开始评估baseline模型的时间消耗: {args.dataset_name}")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # --- 加载数据集 ---
    try:
        if args.dataset_name == "aime":
            dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
            id_offset = 60  # AIME问题ID从60开始
        elif args.dataset_name == "live":
            dataset = load_dataset("opencompass/LiveMathBench", "v202412_AMC_en")["test"]
            id_offset = 0
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

    model_dir = os.path.join(args.results_dir, args.model_name, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化时间统计数据
    all_times = []  # 所有情况下的时间消耗
    finished_times = []  # 只有finished=true的时间消耗
    
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
                
            # 提取时间数据
            time_stats = data.get("time_stats", {})
            total_time = time_stats.get("total_time")
            
            if total_time is not None:
                # 记录所有情况下的时间
                all_times.append(total_time)
                
                # 检查是否完成
                reasoning_data = data.get("reasoning", {})
                if reasoning_data.get("finished", False):
                    finished_times.append(total_time)
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 计算统计信息
    if all_times:
        avg_all_time = np.mean(all_times)
        median_all_time = np.median(all_times)
        min_all_time = np.min(all_times)
        max_all_time = np.max(all_times)
    else:
        avg_all_time = median_all_time = min_all_time = max_all_time = 0
        
    if finished_times:
        avg_finished_time = np.mean(finished_times)
        median_finished_time = np.median(finished_times)
        min_finished_time = np.min(finished_times)
        max_finished_time = np.max(finished_times)
        finished_percentage = (len(finished_times) / len(all_times)) * 100 if all_times else 0
    else:
        avg_finished_time = median_finished_time = min_finished_time = max_finished_time = 0
        finished_percentage = 0

    # 打印结果
    logging.info("=" * 50)
    logging.info("时间消耗统计摘要:")
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"模型: {args.model_name}")
    logging.info(f"结果目录: {model_dir}")
    logging.info(f"总样本数: {len(all_times)}")
    logging.info(f"完成的样本数: {len(finished_times)} ({finished_percentage:.2f}%)")
    logging.info("\n所有情况下的时间消耗:")
    logging.info(f"  - 平均时间: {avg_all_time:.2f}秒")
    logging.info(f"  - 中位数时间: {median_all_time:.2f}秒")
    logging.info(f"  - 最小时间: {min_all_time:.2f}秒")
    logging.info(f"  - 最大时间: {max_all_time:.2f}秒")
    logging.info("\n只有finished=true的时间消耗:")
    logging.info(f"  - 平均时间: {avg_finished_time:.2f}秒")
    logging.info(f"  - 中位数时间: {median_finished_time:.2f}秒")
    logging.info(f"  - 最小时间: {min_finished_time:.2f}秒")
    logging.info(f"  - 最大时间: {max_finished_time:.2f}秒")
    logging.info("=" * 50)
    
    return {
        "all_times": {
            "count": len(all_times),
            "mean": avg_all_time,
            "median": median_all_time,
            "min": min_all_time,
            "max": max_all_time
        },
        "finished_times": {
            "count": len(finished_times),
            "mean": avg_finished_time,
            "median": median_finished_time,
            "min": min_finished_time,
            "max": max_finished_time,
            "percentage": finished_percentage
        }
    }

def evaluate_spec_scale_time(args):
    """评估spec_scale方法的时间消耗"""
    logging.info(f"开始评估spec_scale方法的时间消耗: {args.dataset_name}")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # --- 加载数据集 ---
    try:
        if args.dataset_name == "aime":
            dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
            id_offset = 60  # AIME问题ID从60开始
        elif args.dataset_name == "math":
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
            id_offset = 0
        elif args.dataset_name == "live":
            dataset = load_dataset("opencompass/LiveMathBench", "v202412_AMC_en")["test"]
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
    model_dir = os.path.join(args.results_dir, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化时间统计数据
    all_times = []  # 所有情况下的时间消耗
    best_result_times = []  # 只有best_result不为None的时间消耗
    data_count = 0
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
            # 提取时间数据
            total_time = data.get("total_time")
            
            if total_time is not None:
                # 记录所有情况下的时间
                all_times.append(total_time)

                # 检查best_result是否为None
                best_result = data.get("best_result")
                answer = data.get("answer") #为了兼容spec_reason.py的结果
                if best_result is not None or answer is not None:
                    best_result_times.append(total_time)
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")
    # 计算统计信息
    if all_times:
        avg_all_time = np.mean(all_times)
        median_all_time = np.median(all_times)
        min_all_time = np.min(all_times)
        max_all_time = np.max(all_times)
    else:
        avg_all_time = median_all_time = min_all_time = max_all_time = 0
        
    if best_result_times:
        avg_best_time = np.mean(best_result_times)
        median_best_time = np.median(best_result_times)
        min_best_time = np.min(best_result_times)
        max_best_time = np.max(best_result_times)
        best_percentage = (len(best_result_times) / len(all_times)) * 100 if all_times else 0
    else:
        avg_best_time = median_best_time = min_best_time = max_best_time = 0
        best_percentage = 0

    # 打印结果
    logging.info("=" * 50)
    logging.info("时间消耗统计摘要:")
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"结果目录: {model_dir}")
    logging.info(f"总样本数: {len(all_times)}")
    logging.info(f"best_result不为None的样本数: {len(best_result_times)} ({best_percentage:.2f}%)")
    logging.info("\n所有情况下的时间消耗:")
    logging.info(f"  - 平均时间: {avg_all_time:.2f}秒")
    logging.info(f"  - 中位数时间: {median_all_time:.2f}秒")
    logging.info(f"  - 最小时间: {min_all_time:.2f}秒")
    logging.info(f"  - 最大时间: {max_all_time:.2f}秒")
    logging.info("\n只有best_result不为None的时间消耗:")
    logging.info(f"  - 平均时间: {avg_best_time:.2f}秒")
    logging.info(f"  - 中位数时间: {median_best_time:.2f}秒")
    logging.info(f"  - 最小时间: {min_best_time:.2f}秒")
    logging.info(f"  - 最大时间: {max_best_time:.2f}秒")
    logging.info("=" * 50)
    
    return {
        "all_times": {
            "count": len(all_times),
            "mean": avg_all_time,
            "median": median_all_time,
            "min": min_all_time,
            "max": max_all_time
        },
        "best_result_times": {
            "count": len(best_result_times),
            "mean": avg_best_time,
            "median": median_best_time,
            "min": min_best_time,
            "max": max_best_time,
            "percentage": best_percentage
        }
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估模型的时间消耗")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], default="aime",
                        help="数据集名称")
    parser.add_argument("--model_name", type=str, default=None,
                        help="baseline模型名称（用于baseline_vllm_test目录）")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="结果文件的目录")
    parser.add_argument("--spec_scale", action="store_true",
                        help="是否评估spec_scale方法的时间消耗")
    args = parser.parse_args()
    
    # 根据参数选择评估方法
    if args.spec_scale:
        evaluate_spec_scale_time(args)
    else:
        # 检查参数
        if args.model_name is None:
            parser.error("评估baseline模型时必须指定--model_name参数")
        evaluate_baseline_time(args)

if __name__ == "__main__":
    main()