# python spec_scale_reason_2_watch.py --dataset_name aime --problem_id 60 --repeat_id 2 --output_dir results/spec_scale_2 --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 3

import os
import time
import pickle
import pprint
import logging
import argparse
import numpy as np
from collections import Counter
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
import re
import subprocess
import psutil
import threading
import json
from datetime import datetime
# 导入choose-prompts模块中的选择prompt
from prompts.choose_prompts import math_choose_prompt, gpqa_choose_prompt
from prompts.method_prompt import A_prompt, B_prompt, C_prompt, D_prompt, E_prompt, F_prompt, G_prompt, H_prompt, I_prompt, J_prompt, K_prompt, L_prompt, M_prompt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResourceMonitor:
    """资源监控类，用于监控GPU和CPU使用情况"""
    
    def __init__(self, interval=1.0, gpu_ids=None):
        """
        初始化资源监控器
        
        Args:
            interval: 采样间隔（秒）
            gpu_ids: 要监控的GPU ID列表，默认为None（监控所有可用GPU）
        """
        self.interval = interval
        self.gpu_ids = gpu_ids
        self.running = False
        self.thread = None
        self.gpu_memory_usage = []
        self.gpu_utilization = []
        self.cpu_usage = []
        self.timestamp = []
        self.lock = threading.Lock()
        
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            # 使用nvidia-smi命令获取GPU信息
            cmd = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            lines = output.split('\n')
            
            gpu_info = {}
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    gpu_id = int(parts[0].strip())
                    if self.gpu_ids is None or gpu_id in self.gpu_ids:
                        memory_used = int(parts[1].strip())
                        gpu_util = int(parts[2].strip())
                        gpu_info[gpu_id] = {
                            'memory_used': memory_used,  # MB
                            'utilization': gpu_util      # %
                        }
            return gpu_info
        except Exception as e:
            logging.error(f"获取GPU信息失败: {e}")
            return {}
    
    def monitor_resources(self):
        """资源监控主循环"""
        while self.running:
            try:
                # 获取当前时间戳
                current_time = datetime.now()
                
                # 获取GPU信息
                gpu_info = self.get_gpu_info()
                
                # 获取CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # 计算平均GPU内存使用和利用率
                avg_gpu_memory = 0
                avg_gpu_util = 0
                
                if gpu_info:
                    memory_values = [info['memory_used'] for info in gpu_info.values()]
                    util_values = [info['utilization'] for info in gpu_info.values()]
                    avg_gpu_memory = sum(memory_values) / len(memory_values)
                    avg_gpu_util = sum(util_values) / len(util_values)
                
                # 加锁更新数据
                with self.lock:
                    self.gpu_memory_usage.append(avg_gpu_memory)
                    self.gpu_utilization.append(avg_gpu_util)
                    self.cpu_usage.append(cpu_percent)
                    self.timestamp.append(current_time)
                
                # 等待下一个采样间隔
                time.sleep(self.interval)
            except Exception as e:
                logging.error(f"资源监控异常: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_resources)
            self.thread.daemon = True
            self.thread.start()
            logging.info("资源监控已启动")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logging.info("资源监控已停止")
    
    def get_stats(self):
        """获取统计数据"""
        with self.lock:
            if not self.gpu_memory_usage:
                return {
                    'avg_gpu_memory_mb': 0,
                    'max_gpu_memory_mb': 0,
                    'avg_gpu_utilization': 0,
                    'max_gpu_utilization': 0,
                    'avg_cpu_usage': 0,
                    'max_cpu_usage': 0,
                    'samples': 0
                }
            
            stats = {
                'avg_gpu_memory_mb': sum(self.gpu_memory_usage) / len(self.gpu_memory_usage),
                'max_gpu_memory_mb': max(self.gpu_memory_usage),
                'avg_gpu_utilization': sum(self.gpu_utilization) / len(self.gpu_utilization),
                'max_gpu_utilization': max(self.gpu_utilization),
                'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage),
                'max_cpu_usage': max(self.cpu_usage),
                'samples': len(self.gpu_memory_usage)
            }
            return stats
    
    def save_to_file(self, filename):
        """保存监控数据到文件"""
        with self.lock:
            data = {
                'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S.%f') for ts in self.timestamp],
                'gpu_memory_mb': self.gpu_memory_usage,
                'gpu_utilization': self.gpu_utilization,
                'cpu_usage': self.cpu_usage,
                'stats': self.get_stats()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"资源监控数据已保存到 {filename}")

# 新增函数: 获取方法提示
def get_method_prompt(method_code):
    """根据方法代码返回对应的方法提示"""
    method_prompts = {
        'A': A_prompt,
        'B': B_prompt,
        'C': C_prompt,
        'D': D_prompt,
        'E': E_prompt,
        'F': F_prompt,
        'G': G_prompt,
        'H': H_prompt,
        'I': I_prompt,
        'J': J_prompt,
        'K': K_prompt,
        'L': L_prompt,
        'M': M_prompt
    }
    return method_prompts.get(method_code, M_prompt)

def get_first_user_msg(problem, options=None, method_code=None):
    # 获取解题方法提示
    method_prompt = get_method_prompt(method_code) if method_code else M_prompt
        
    if options is None:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        
        Tips: {method_prompt}
        
        Problem: {problem}
        """
        return system_prompt.format(method_prompt=method_prompt, problem=problem)
    else:
        system_prompt = """
        What is the correct answer to the following problem? Please reason step by step. 
        Separate logical reasoning steps with two newline characters (\n\n).
        Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

        **Example output:** \\boxed{{A}}
        
        Tips: {method_prompt}

        Problem: {problem}.
        Choices: 
        (A) {ans_a}
        (B) {ans_b}
        (C) {ans_c}
        (D) {ans_d}
        """
        return system_prompt.format(
            method_prompt=method_prompt,
            problem=problem,
            ans_a=options["A"],
            ans_b=options["B"],
            ans_c=options["C"],
            ans_d=options["D"],
        )

def process_vllm_logprobs(token, logprobs, method, temp=1.0):
    """处理vLLM返回的logprobs"""
    # 过滤出数字token的概率
    digit_logprobs = {k: v for k, v in logprobs.items()}
    
    if method == "greedy":
        # 返回生成的token
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        # 转换为概率并归一化
        probs = {tok: np.exp(lp / temp) for tok, lp in digit_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if str(i) not in probs:
                probs[str(i)] = 0
        
        # 计算加权平均分数
        avg_score = sum([int(t) * p for t, p in probs.items()])
        logging.info(f"Avg score: {avg_score}")
        return avg_score
    else:
        raise NotImplementedError

def select_best_result(results):
    """根据指定规则选择最佳结果"""
    # 过滤出成功完成的结果
    finished_results = [r for r in results if r.get("stop_reason") == "finished" and r.get("answer") is not None]
    
    if not finished_results:
        logging.warning("所有推理都未能完成或提取答案")
        return None
    
    # 统计答案
    answers = [r.get("answer") for r in finished_results]
    answer_counts = Counter(answers)
    
    # 如果有答案相同的，取多数
    if len(answer_counts) < len(finished_results):
        most_common_answer, count = answer_counts.most_common(1)[0]
        if count > 1:  # 确保至少有两个相同答案
            # 找出具有最常见答案的所有结果
            selected_results = [r for r in finished_results if r.get("answer") == most_common_answer]
            # 从中选择token数最多的
            selected_result = max(selected_results, key=lambda r: r.get("num_tokens", 0))
            selected_result["selection_reason"] = f"多数原则: {count}/{len(finished_results)}个结果给出相同答案"
            return selected_result
    
    # 如果答案各不相同，取token数最多的
    selected_result = max(finished_results, key=lambda r: r.get("num_tokens", 0))
    selected_result["selection_reason"] = "答案各不相同，选择token数最多的结果"
    return selected_result

def extract_final_answer(reasoning, dataset_name):
    """从推理文本中提取最终答案"""
    if not reasoning:
        return None
        
    # 使用正则表达式查找 \boxed{...} 模式的起始位置
    match = re.search(r"\\boxed\{", reasoning)
    if match:
        start_pos = match.end()
        # 使用括号计数器处理嵌套花括号
        open_braces = 1
        close_pos = start_pos
        
        for i in range(start_pos, len(reasoning)):
            if reasoning[i] == '{':
                open_braces += 1
            elif reasoning[i] == '}':
                open_braces -= 1
                if open_braces == 0:
                    close_pos = i
                    break
        
        if open_braces == 0:
            answer = reasoning[start_pos:close_pos].strip()
            
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
            # 对于AIME/MATH，尝试转换为数字或直接返回
            else:
                try:
                    # 首先尝试简单的整数转换
                    return int(answer)
                except ValueError:
                    # 对于MATH数据集，可能包含复杂的数学表达式，直接返回
                    if dataset_name == "math":
                        # 规范化答案格式，移除多余空格
                        return answer.strip()
                    # 添加更复杂的解析（如分数等）
                    logging.warning(f"无法将boxed答案转换为整数: {answer}")
                    return answer  # 如果转换失败，则返回字符串
        else:
            logging.warning("在推理文本中找到了\\boxed{但没有找到匹配的}，可能是格式错误")
            return None
    else:
        # 备用方案：检查最后一步是否包含"Answer: X"（用于GPQA）
        if dataset_name == "gpqa":
            match_answer = re.search(r"[Aa]nswer:\s*([A-D])", reasoning)
            if match_answer:
                return match_answer.group(1).upper()

        logging.warning("在最终步骤中找不到\\boxed{}模式")
        return None

def get_dataset(dataset_name):
    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "gpqa":
        if os.getenv("HF_HUB_OFFLINE", "0") == "1":
            dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
        else:    
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    else:
        raise NotImplementedError
    return dataset

def initialize_models(GPU_num):
    """初始化小模型和大模型"""
    logging.info("正在初始化模型...")
    models = {}
    try:
        # 初始化小模型
        logging.info("正在加载草稿模型...")
        models["draft"] = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.2,
            trust_remote_code=True
        )
        
        # 初始化大模型
        logging.info("正在加载目标模型...")
        models["target"] = LLM(
            # model="Qwen/QwQ-32B",
            model="/hpc2hdd/home/bwang423/.cache/modelscope/hub/models/Qwen/QwQ-32B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.7,
            trust_remote_code=True
        )
        logging.info("模型初始化完成")
        return models
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
        exit(1)

def parse_problem_range(problem_id_str):
    """解析问题ID范围，支持单个ID或范围（如60-89）"""
    if '-' in problem_id_str:
        start, end = map(int, problem_id_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(problem_id_str)]

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Runs speculative reasoning using a small model")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="aime",
                        help="Dataset")
    parser.add_argument("--score_threshold", type=float, default=7.0, 
                        help="Acceptance threshold")
    parser.add_argument("--token_budget", type=int, default=8192,
                        help="Max num of total output tokens in each step")
    # problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA
    parser.add_argument("--problem_id", type=str, default="60",
                        help="Query ID (60-89 for AIME), can be a single ID or a range like 60-89")
    parser.add_argument("--repeat_id", type=int, default=1,
                        help="Repeat ID (1-16, k=16)")
    parser.add_argument("--score_method", type=str, choices=["greedy", "average"], default="greedy",
                        help="Scoring method")
    parser.add_argument("--output_dir", type=str, default="results/spec_scale_Inf", 
                        help="Where result pickle files will be written to")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallel size for model initialization")
    parser.add_argument("--method_num", type=int, default=3,
                        help="Number of solution methods to use (default: 3)")
    return parser.parse_known_args()

def prepare_problem_data(args):
    """准备问题数据"""
    if args.dataset_name == "aime":
        problem = args.dataset["problem"][args.problem_id - 60]
        options = None
    elif args.dataset_name == "math":
        problem = args.dataset["problem"][args.problem_id]
        options = None
    elif args.dataset_name == "gpqa":
        problem = args.dataset["Question"][args.problem_id]
        options = {
            "A": args.dataset["Correct Answer"][args.problem_id],
            "B": args.dataset["Incorrect Answer 1"][args.problem_id],
            "C": args.dataset["Incorrect Answer 2"][args.problem_id],
            "D": args.dataset["Incorrect Answer 3"][args.problem_id],
        }
    return problem, options

def choose_solution_methods(problem, options, dataset_name, models, method_num=3):
    """选择最合适的解题方法
    
    Args:
        problem: 问题文本
        options: 选项（如有）
        dataset_name: 数据集名称
        models: 模型字典
        method_num: 需要选择的解题方法数量，默认为3
        
    Returns:
        包含方法代码的列表
    """
    logging.info(f"选择{method_num}个解题方法...")
    
    # 选择合适的prompt
    if dataset_name == "aime" or dataset_name == "math":
        choose_prompt = math_choose_prompt
    elif dataset_name == "gpqa":
        choose_prompt = gpqa_choose_prompt
    else:
        logging.warning(f"未知数据集: {dataset_name}，默认使用math_choose_prompt")
        choose_prompt = math_choose_prompt
    
    # 准备提示，根据method_num调整要求数量
    prompt = f"{choose_prompt}\n\nProblem: {problem}"
    if options:
        prompt += "\nChoices: "
        for key, value in options.items():
            prompt += f"\n({key}) {value}"
    prompt = f"{prompt}\n\nExamine the problem and select the **{method_num}** strategies (by their codes, e.g. {'B,E,F' if method_num == 3 else 'B,E,F,A' if method_num == 4 else 'B,E'}) that you believe are most promising for solving it. You only need to output {method_num} codes, without any other symbols or text\n<think></think>I think the best {method_num} methods are: "
    
    # 设置生成参数，根据method_num调整max_tokens
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=method_num * 2-1,  # 每个方法可能需要2个token（字母+逗号）
    )
    
    # 使用目标模型生成方法选择
    outputs = models["target"].generate([prompt], sampling_params)
    method_text = outputs[0].outputs[0].text.strip()
    
    # print(prompt)
    print(method_text)

    # 提取方法代码
    methods_match = re.findall(r'[A-M]', method_text)
    
    # 如果提取的方法数量不足，使用'M'填充
    if len(methods_match) < method_num:
        logging.warning(f"从'{method_text}'中只提取到{len(methods_match)}个方法代码，不足{method_num}个，使用'M'填充")
        while len(methods_match) < method_num:
            methods_match.append('M')
    elif len(methods_match) > method_num:
        # 如果提取的方法数量超过需要的数量，截取前method_num个
        logging.warning(f"从'{method_text}'中提取到{len(methods_match)}个方法代码，超过{method_num}个，只使用前{method_num}个")
        methods_match = methods_match[:method_num]
    
    logging.info(f"选择的解题方法: {''.join(methods_match)}")

    return methods_match

def run_reasoning_process(args, problem, options):
    """运行推理过程"""
    start_time_total = time.time()  # 记录整个过程的开始时间
    
    # 初始化资源监控器
    resource_monitor = ResourceMonitor(interval=1.0, gpu_ids=[0, 1, 2, 3])
    resource_monitor.start()
    
    # 初始化模型调用计数器
    model_call_counts = {
        "draft": 0,  # 小模型调用次数
        "target": 0  # 大模型调用次数
    }
    
    try:
        # 使用大模型选择解题方法，使用args.method_num参数
        solution_methods = choose_solution_methods(problem, options, args.dataset_name, models, args.method_num)
        # 增加大模型调用计数
        model_call_counts["target"] += 1
        
        # 为每个方法创建独立的推理轨道，使用索引区分相同方法的不同实例
        method_tracks = {}
        method_count = {}  # 用于跟踪每个方法出现的次数
        
        for i, method_code in enumerate(solution_methods):
            # 如果是重复的方法，给它一个唯一的标识符
            if method_code in method_count:
                method_count[method_code] += 1
                track_id = f"{method_code}{method_count[method_code]}"
            else:
                method_count[method_code] = 1
                track_id = f"{method_code}1"
                
            method_tracks[track_id] = {
                "steps": [], 
                "finished": False, 
                "metadata": [], 
                "rewrite_count": 0, 
                "total_steps": 0,
                "method_code": method_code  # 原始方法代码，不含索引
            }
        
        step_id = 0
        
        while True:
            # 检查是否所有轨道都已完成
            active_methods = [method for method, track in method_tracks.items() if not track["finished"]]
            if not active_methods:
                break
                
            logging.info(f"[Step {step_id}] 活跃方法: {active_methods}")
            
            # 1. 为每个活跃方法并行生成下一步推理
            active_prompts = []
            method_mapping = []  # 用于跟踪每个提示对应的方法
            
            for method in active_methods:
                track = method_tracks[method]
                # 获取原始方法代码（不含索引）
                original_method_code = track["method_code"]
                
                if not track["steps"]:  # 第一步
                    prompt = get_first_user_msg(problem, options, method_code=original_method_code)
                else:  # 继续之前的消息
                    steps_so_far_str = "\n\n".join(track["steps"]) + "\n\n"
                    prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code)}\n<think>{steps_so_far_str}"
                
                active_prompts.append(prompt)
                method_mapping.append(method)
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=512,
                stop=["\n\n"]
            )
            
            # 批量生成小模型的推理步骤
            small_outputs = models["draft"].generate(active_prompts, sampling_params)
            # 增加小模型调用计数
            model_call_counts["draft"] += 1
            
            # 2. 处理小模型输出并准备评分
            score_prompts = []
            score_method_mapping = []
            small_results = {}
            
            for i, output in enumerate(small_outputs):
                method = method_mapping[i]
                generated_text = output.outputs[0].text
                num_tokens = len(output.outputs[0].token_ids)
                
                # 检查是否完成
                finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
                
                # 存储小模型结果
                small_results[method] = {
                    "text": generated_text,
                    "tokens": num_tokens,
                    "finished": finished
                }
                
                # 准备评分提示
                track = method_tracks[method]
                steps_for_scoring = track["steps"] + [generated_text]
                steps_so_far_str = "\n\n".join(steps_for_scoring) + "\n\n"
                
                score_prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code)}\n<think>{steps_so_far_str}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
                
                score_prompts.append(score_prompt)
                score_method_mapping.append(method)
            
            # 3. 批量评分
            scoring_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=10
            )
            
            score_outputs = models["target"].generate(score_prompts, scoring_params)
            
            # 4. 处理评分结果，确定哪些需要重写
            rewrite_prompts = []
            rewrite_method_mapping = []
            scores = {}
            
            for i, output in enumerate(score_outputs):
                method = score_method_mapping[i]
                token = output.outputs[0].text
                logprobs = output.outputs[0].logprobs[0]
                
                score = process_vllm_logprobs(token, logprobs, method=args.score_method)
                scores[method] = score
                
                # 如果分数低于阈值，需要重写
                if score is None or score < args.score_threshold:
                    logging.info(f"[Step {step_id}] 方法 {method} 分数 {score}，需要重写")
                    track = method_tracks[method]
                    
                    if not track["steps"]:  # 第一步
                        prompt = get_first_user_msg(problem, options, method_code=original_method_code)
                    else:  # 继续之前的消息
                        steps_so_far_str = "\n\n".join(track["steps"]) + "\n\n"
                        prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code)}\n<think>{steps_so_far_str}"
                    
                    rewrite_prompts.append(prompt)
                    rewrite_method_mapping.append(method)
                else:
                    logging.info(f"[Step {step_id}] 方法 {method} 分数 {score}，已接受")
            
            # 5. 如果有需要重写的，批量重写
            rewrite_results = {}
            if rewrite_prompts:
                rewrite_outputs = models["target"].generate(rewrite_prompts, sampling_params)
                
                for i, output in enumerate(rewrite_outputs):
                    method = rewrite_method_mapping[i]
                    generated_text = output.outputs[0].text
                    num_tokens = len(output.outputs[0].token_ids)
                    
                    # 检查是否完成
                    finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
                    
                    rewrite_results[method] = {
                        "text": generated_text,
                        "tokens": num_tokens,
                        "finished": finished
                    }
            
            # 6. 更新每个方法的轨道
            for method in active_methods:
                track = method_tracks[method]
                
                # 确定最终使用的步骤
                if method in rewrite_results:
                    # 使用重写的结果
                    step_text = rewrite_results[method]["text"]
                    num_tokens = rewrite_results[method]["tokens"]
                    finished = rewrite_results[method]["finished"]
                    used_base_model = True
                    # 增加改写计数
                    track["rewrite_count"] += 1
                else:
                    # 使用小模型的结果
                    step_text = small_results[method]["text"]
                    num_tokens = small_results[method]["tokens"]
                    finished = small_results[method]["finished"]
                    used_base_model = False
                
                # 增加总步骤计数
                track["total_steps"] += 1
                
                # 处理特殊情况
                if "</think>" in step_text and not any([x in step_text for x in ["boxed", "Answer:", "ANSWER:"]]):
                    logging.warning(f"Warning: 方法 {method} 的步骤包含 </think>，正在移除")
                    step_text = step_text.replace("</think>", "")
                    warning_flag = True
                else:
                    warning_flag = False
                
                # 添加到轨道的步骤中
                track["steps"].append(step_text)
                
                # 创建元数据
                metadata = {
                    "step_id": step_id,
                    "method_code": original_method_code,
                    "step_str": step_text,
                    "small_model_step": small_results[method]["text"],
                    "num_output_tokens_small": small_results[method]["tokens"],
                    "score": scores[method],
                    "base_model_step": rewrite_results.get(method, {}).get("text") if used_base_model else None,
                    "num_output_tokens_base": rewrite_results.get(method, {}).get("tokens") if used_base_model else None,
                    "final_num_output_tokens": num_tokens,
                    "used_base_model": used_base_model
                }
                
                if warning_flag:
                    metadata["warning"] = "step_str had a </think>"
                
                track["metadata"].append(metadata)
                
                # 检查是否完成
                if finished:
                    track["finished"] = True
                    track["stop_reason"] = "finished"
                elif len(track["steps"]) > 2 and track["steps"][-1] == track["steps"][-2]:
                    # 处理重复步骤的边缘情况
                    track["finished"] = True
                    track["stop_reason"] = "repeated"
                elif sum([m["final_num_output_tokens"] for m in track["metadata"]]) >= args.token_budget:
                    track["finished"] = True
                    track["stop_reason"] = "budget"
            
            step_id += 1

        # 计算总时间
        total_time = time.time() - start_time_total

        # 停止资源监控
        resource_monitor.stop()
        
        # 获取资源使用统计
        resource_stats = resource_monitor.get_stats()
        
        # 计算改写率
        total_steps = sum(track["total_steps"] for track in method_tracks.values())
        total_rewrites = sum(track["rewrite_count"] for track in method_tracks.values())
        avg_rewrite_rate = (total_rewrites / total_steps) * 100 if total_steps > 0 else 0
        
        # 记录资源使用情况
        logging.info(f"平均GPU内存使用: {resource_stats['avg_gpu_memory_mb']:.2f} MB")
        logging.info(f"最大GPU内存使用: {resource_stats['max_gpu_memory_mb']:.2f} MB")
        logging.info(f"平均GPU利用率: {resource_stats['avg_gpu_utilization']:.2f}%")
        logging.info(f"最大GPU利用率: {resource_stats['max_gpu_utilization']:.2f}%")
        logging.info(f"平均CPU使用率: {resource_stats['avg_cpu_usage']:.2f}%")
        logging.info(f"最大CPU使用率: {resource_stats['max_cpu_usage']:.2f}%")
        logging.info(f"小模型调用次数: {model_call_counts['draft']}")
        logging.info(f"大模型调用次数: {model_call_counts['target']}")
        logging.info(f"总调用次数: {model_call_counts['draft'] + model_call_counts['target']}")

    except ValueError as e:
        logging.error(f"ValueError caught in chat template application: {e}")
        method_tracks = {}
        total_time = 0
        avg_rewrite_rate = 0
        resource_stats = {}
        model_call_counts = {"draft": 0, "target": 0}
        
        # 确保停止资源监控
        resource_monitor.stop()
         
    return method_tracks, total_time, avg_rewrite_rate, resource_stats, model_call_counts

def process_results(method_tracks, total_time, avg_rewrite_rate, args, problem, options, resource_stats=None, model_call_counts=None):
    """处理推理结果，提取答案和统计信息"""
    results = []
    
    for method, track in method_tracks.items():
        if track["steps"]:
            final_reasoning = "\n\n".join(track["steps"])
            answer = extract_final_answer(final_reasoning, args.dataset_name)
            
            # 计算token数量
            num_tokens = sum(m.get("num_tokens", 0) for m in track["metadata"])
            
            # 确定停止原因
            if track["finished"]:
                stop_reason = "finished"
            elif num_tokens >= args.token_budget:
                stop_reason = "budget"
            else:
                stop_reason = "unknown"
            
            result = {
                "method": method,
                "method_code": track["method_code"],
                "steps": track["steps"],
                "metadata": track["metadata"],
                "final_reasoning": final_reasoning,
                "answer": answer,
                "num_tokens": num_tokens,
                "rewrite_count": track["rewrite_count"],
                "total_steps": track["total_steps"],
                "rewrite_rate": (track["rewrite_count"] / track["total_steps"]) * 100 if track["total_steps"] > 0 else 0,
                "stop_reason": stop_reason
            }
            results.append(result)
    
    # 选择最佳结果
    best_result = select_best_result(results)
    
    # 创建包含统计信息的元数据
    metadata_with_stats = {
        "problem": problem,
        "options": options,
        "dataset_name": args.dataset_name,
        "problem_id": args.problem_id,
        "repeat_id": args.repeat_id,
        "total_time": total_time,
        "avg_rewrite_rate": avg_rewrite_rate,
        "score_threshold": args.score_threshold,
        "token_budget": args.token_budget,
        "score_method": args.score_method,
        "method_num": args.method_num,
        "results": results,
        "best_result": best_result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 添加资源统计信息
    if resource_stats:
        metadata_with_stats["resource_stats"] = resource_stats
    
    # 添加模型调用次数
    if model_call_counts:
        metadata_with_stats["model_call_counts"] = model_call_counts
    
    return metadata_with_stats

def save_results(metadata_with_stats, output_filename):
    """保存结果到文件"""
    with open(f"{output_filename}.pickle", "wb") as f:
        pickle.dump(metadata_with_stats, f)

    with open(f"{output_filename}.txt", "w") as f:
        pprint.pprint(metadata_with_stats, stream=f)
    
    logging.info(f"结果已保存到 {output_filename}.pickle 和 {output_filename}.txt")

def main():
    """主函数，整合所有流程"""
    # 解析命令行参数
    args, _ = parse_arguments()
    
    # 初始化模型
    global models
    models = initialize_models(args.tensor_parallel_size)
    
    # 加载数据集
    args.dataset = get_dataset(args.dataset_name)
    
    # 解析问题ID范围
    problem_ids = parse_problem_range(args.problem_id)
    
    # 准备输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每个问题ID，并对每个问题执行指定次数的重复
    total_tasks = len(problem_ids)
    completed_tasks = 0
    
    logging.info(f"开始处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次")
    
    for problem_id in problem_ids:
        for repeat_idx in range(args.repeat_id):
            logging.info(f"处理问题 ID: {problem_id}，重复 {repeat_idx}/{args.repeat_id}")
            
            # 设置当前问题的输出文件名
            output_filename = os.path.join(output_dir, f"{problem_id}-{repeat_idx}")
            
            # 检查是否已经处理过
            if os.path.exists(f"{output_filename}.pickle"):
                logging.info(f"问题 {problem_id} 重复 {repeat_idx} 已解决，跳过")
                continue
            
            # 准备问题数据
            args.problem_id = problem_id  # 更新当前处理的问题ID
            problem, options = prepare_problem_data(args)
            
            # 运行推理过程
            method_tracks, total_time, avg_rewrite_rate, resource_stats, model_call_counts = run_reasoning_process(args, problem, options)
            
            # 处理结果
            metadata_with_stats = process_results(method_tracks, total_time, avg_rewrite_rate, args, problem, options, resource_stats, model_call_counts)
            
            # 保存结果
            save_results(metadata_with_stats, output_filename)
            
            completed_tasks += 1
            logging.info(f"完成进度: {completed_tasks}/{total_tasks * args.repeat_id} ({completed_tasks/(total_tasks * args.repeat_id)*100:.2f}%)")
    
    logging.info(f"所有任务完成！共处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次")

if __name__ == "__main__":
    main()


