#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python run_experiments.py --dataset aime --problem_ids 62-89 --num_repeats 16 --base_model Qwen/Qwen-32B --small_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --base_model_abbrv Qwen-32B --small_model_abbrv deepseek-1.5B --token_budget 8192 --judge_scheme greedy --threshold 9

import os
import time
import argparse
import subprocess
import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行 SpecReason 实验")
    parser.add_argument("--cluster", type=str, default="hpc",
                        help="集群名称")
    parser.add_argument("--dataset", type=str, default="aime",
                        help="数据集名称")
    parser.add_argument("--judge_scheme", type=str, default="greedy",
                        help="评分方案")
    parser.add_argument("--threshold", type=float, default=9,
                        help="分数阈值")
    parser.add_argument("--num_repeats", type=int, default=16,
                        help="每个问题的重复次数")
    parser.add_argument("--base_model", type=str, default="Qwen/QwQ-32B",
                        help="基础模型名称")
    parser.add_argument("--small_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="小模型名称")
    parser.add_argument("--base_model_abbrv", type=str, default="Qwen-32B",
                        help="基础模型缩写")
    parser.add_argument("--small_model_abbrv", type=str, default="deepseek-1.5B",
                        help="小模型缩写")
    parser.add_argument("--token_budget", type=int, default=8192,
                        help="每个步骤的最大输出令牌数")
    parser.add_argument("--problem_ids", type=str, default="60-89",
                        help="问题ID范围，格式为'start-end'")
    args = parser.parse_args()

    # 设置输出目录
    output_dir = f"results/{args.judge_scheme}_{args.threshold}/{args.dataset}/{args.base_model_abbrv}_{args.small_model_abbrv}"
    logfile_dir = f"logs"

    # 创建必要的目录
    for dir_path in [output_dir, logfile_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"确保目录存在: {dir_path}")

    # 解析问题ID范围
    start_id, end_id = map(int, args.problem_ids.split('-'))
    problem_ids = list(range(start_id, end_id + 1))
    logging.info(f"将处理问题ID: {problem_ids}")

    # 运行实验
    for problem_id in problem_ids:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = f"{logfile_dir}/{timestamp}.log"
        logging.info(f"运行问题 {problem_id}，并行处理 {args.num_repeats} 次运行，日志文件 {logfile}")

        # 初始化进程列表
        processes = []

        # 为每个重复ID启动一个进程
        for repeat_id in range(args.num_repeats):
            cmd = [
                "python", "spec_reason.py",
                "--dataset_name", args.dataset,
                "--problem_id", str(problem_id),
                "--repeat_id", str(repeat_id),
                "--score_threshold", str(args.threshold),
                "--score_method", args.judge_scheme,
                "--token_budget", str(args.token_budget),
                "--output_dir", output_dir
            ]

            with open(logfile, 'a') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
                processes.append(process)

        # 等待所有进程完成
        for process in processes:
            process.wait()

        logging.info(f"问题 {problem_id} 已完成")
        
        # 休息2秒
        time.sleep(2)

if __name__ == "__main__":
    main()