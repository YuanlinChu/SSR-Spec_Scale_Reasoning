#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python score_eval.py --dataset_name aime --results_dir results/spec_scale_m3 --plot

import os
import re
import pickle
import argparse
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_scores(args):
    """分析大模型对小模型的打分分布"""
    logging.info(f"开始分析分数分布")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # 检查结果目录是否存在
    model_dir = os.path.join(args.results_dir, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化分数统计
    score_counts = defaultdict(int)  # 每个分数的计数
    total_scores = 0  # 总分数计数
    method_score_counts = defaultdict(lambda: defaultdict(int))  # 每个方法的分数分布
    
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
            
            # 检查是否有method_tracks字段
            if "method_tracks" not in data:
                logging.warning(f"文件 {filename} 中没有找到method_tracks字段，跳过")
                continue
                
            method_tracks = data["method_tracks"]
            
            # 遍历每个方法轨道
            for method_name, track in method_tracks.items():
                # 遍历每个步骤的元数据
                for metadata in track["metadata"]:
                    # 检查是否有score字段
                    if "score" in metadata:
                        score = metadata["score"]
                        if score is not None:
                            # 将分数转换为整数（0-9）
                            score_int = int(score)
                            # 更新总体分数统计
                            score_counts[score_int] += 1
                            total_scores += 1
                            # 更新方法特定的分数统计
                            method_score_counts[method_name][score_int] += 1
            
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 如果没有找到任何分数
    if total_scores == 0:
        logging.error("未找到任何分数数据")
        return

    # 计算总体分数分布
    score_distribution = {}
    for score in range(10):  # 0-9
        count = score_counts.get(score, 0)
        percentage = (count / total_scores) * 100 if total_scores > 0 else 0
        score_distribution[score] = {
            "count": count,
            "percentage": percentage
        }

    # 计算每个方法的分数分布
    method_score_distributions = {}
    for method_name, counts in method_score_counts.items():
        method_total = sum(counts.values())
        method_distribution = {}
        for score in range(10):  # 0-9
            count = counts.get(score, 0)
            percentage = (count / method_total) * 100 if method_total > 0 else 0
            method_distribution[score] = {
                "count": count,
                "percentage": percentage
            }
        method_score_distributions[method_name] = {
            "distribution": method_distribution,
            "total": method_total
        }

    # 打印总体分数分布
    logging.info("=" * 50)
    logging.info("总体分数分布:")
    for score in range(10):
        dist = score_distribution[score]
        logging.info(f"分数 {score}: {dist['count']} 次 ({dist['percentage']:.2f}%)")
    logging.info(f"总计: {total_scores} 个分数")
    logging.info("=" * 50)
    
    # 打印每个方法的分数分布
    logging.info("各方法分数分布:")
    for method_name, data in method_score_distributions.items():
        logging.info(f"方法 {method_name} (总计: {data['total']} 个分数):")
        for score in range(10):
            dist = data["distribution"][score]
            logging.info(f"  分数 {score}: {dist['count']} 次 ({dist['percentage']:.2f}%)")
        logging.info("-" * 30)
    
    # 生成分数分布图表
    if args.plot:
        # 总体分数分布图
        plt.figure(figsize=(12, 6))
        scores = list(range(10))
        percentages = [score_distribution[score]["percentage"] for score in scores]
        
        plt.bar(scores, percentages, color='skyblue')
        plt.xlabel('score')
        plt.ylabel('percentage(%)')
        plt.title(f'Overall score distribution (total {total_scores} scores)')
        plt.xticks(scores)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, v in enumerate(percentages):
            plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        # 保存图表
        plot_path = os.path.join(args.results_dir, f"{args.dataset_name}_score_distribution.png")
        plt.savefig(plot_path)
        logging.info(f"总体分数分布图已保存至: {plot_path}")
        
        # # 为每个方法生成分数分布图
        # if len(method_score_distributions) <= 10:  # 限制方法数量，避免图表过多
        #     for method_name, data in method_score_distributions.items():
        #         plt.figure(figsize=(10, 5))
        #         method_percentages = [data["distribution"][score]["percentage"] for score in scores]
                
        #         plt.bar(scores, method_percentages, color='lightgreen')
        #         plt.xlabel('分数')
        #         plt.ylabel('百分比 (%)')
        #         plt.title(f'方法 {method_name} 分数分布 (总计: {data["total"]} 个分数)')
        #         plt.xticks(scores)
        #         plt.grid(axis='y', linestyle='--', alpha=0.7)
                
        #         # 添加数值标签
        #         for i, v in enumerate(method_percentages):
        #             plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
                
        #         # 保存图表
        #         method_plot_path = os.path.join(args.results_dir, f"{args.dataset_name}_{method_name}_score_distribution.png")
        #         plt.savefig(method_plot_path)
        #         logging.info(f"方法 {method_name} 分数分布图已保存至: {method_plot_path}")
    
    return {
        "total_scores": total_scores,
        "score_distribution": score_distribution,
        "method_score_distributions": method_score_distributions
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析大模型对小模型的打分分布")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], default="math",
                        help="数据集名称")
    parser.add_argument("--results_dir", type=str, default="results/spec_scale_m3",
                        help="结果文件的目录")
    parser.add_argument("--plot", action="store_true",
                        help="是否生成分数分布图表")
    args = parser.parse_args()
    
    analyze_scores(args)

if __name__ == "__main__":
    main()