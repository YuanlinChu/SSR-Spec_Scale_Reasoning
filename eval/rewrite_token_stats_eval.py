#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python rewrite_token_stats_eval.py --dataset_name aime --results_dir results/spec_scale_2_m5_t7
import os
import re
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_rewrite_tokens(args):
    """分析改写率和token数"""
    logging.info(f"开始分析改写率和token数")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # 检查结果目录是否存在
    model_dir = os.path.join(args.results_dir, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化统计数据
    problem_stats = defaultdict(lambda: {"rewrite_rates": [], "token_counts": []})
    
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
        
        # 加载结果文件
        file_path = os.path.join(model_dir, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            # 提取改写率
            avg_rewrite_rate = data.get("avg_rewrite_rate")
            if avg_rewrite_rate is not None:
                problem_stats[problem_id]["rewrite_rates"].append(avg_rewrite_rate)
            
            # 提取token数
            best_result = data.get("best_result", {})
            if best_result and "num_tokens" in best_result:
                problem_stats[problem_id]["token_counts"].append(best_result["num_tokens"])
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 计算每个问题的平均值
    results = []
    for problem_id, stats in problem_stats.items():
        avg_rewrite_rate = np.mean(stats["rewrite_rates"]) if stats["rewrite_rates"] else np.nan
        avg_num_tokens = np.mean(stats["token_counts"]) if stats["token_counts"] else np.nan
        
        results.append({
            "problem_id": problem_id,
            "avg_rewrite_rate": avg_rewrite_rate,
            "avg_num_tokens": avg_num_tokens,
            "num_samples": len(stats["rewrite_rates"])
        })
    
    # 创建DataFrame并排序
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("problem_id")
        
        # 打印结果
        logging.info("\n" + "=" * 80)
        logging.info("改写率和Token数统计:")
        logging.info(f"数据集: {args.dataset_name}")
        logging.info(f"结果目录: {args.results_dir}")
        logging.info(f"总问题数: {len(df)}")
        logging.info("=" * 80)
        
        # 打印表格
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        logging.info("\n" + df.to_string(index=False))
        
        # 计算总体平均值
        overall_avg_rewrite = df["avg_rewrite_rate"].mean()
        overall_avg_tokens = df["avg_num_tokens"].mean()
        logging.info("=" * 80)
        logging.info(f"总体平均改写率: {overall_avg_rewrite:.2f}%")
        logging.info(f"总体平均Token数: {overall_avg_tokens:.2f}")
        logging.info("=" * 80)
        
        # 保存到CSV文件
        output_file = os.path.join(args.results_dir, f"{args.dataset_name}_rewrite_token_stats.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"结果已保存到: {output_file}")
        
        return df
    else:
        logging.error("未找到有效数据")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析改写率和token数")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], default="aime",
                        help="数据集名称")
    parser.add_argument("--results_dir", type=str, default="results/spec_scale_2_m5_t7",
                        help="结果文件的目录")
    args = parser.parse_args()
    
    analyze_rewrite_tokens(args)

if __name__ == "__main__":
    main()