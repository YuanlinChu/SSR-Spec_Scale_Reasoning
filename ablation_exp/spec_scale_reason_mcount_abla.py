# python ablation_exp/spec_scale_reason_mcount_abla.py --dataset_name aime --problem_id 60-62 --strategy_pool_size 6 --method_num 3 --repeat_id 1 --output_dir results/spec_scale_strategy_6

import time
# import openai
import pickle
import pprint
import logging
import argparse
import numpy as np
# from openai import OpenAI
# import statistics
from collections import Counter
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
import re
# import random
# 导入消融实验的prompts
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts.abla_choose_prompts import method_selection_prompt_6, method_selection_prompt_9, method_selection_prompt_15
from prompts import abla_method_prompt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 新增函数: 获取方法提示
def get_method_prompt(method_code, strategy_pool_size=12):
    """根据方法代码和策略池大小返回对应的方法提示"""
    # 基础方法提示（6个策略池）
    base_prompts = {
        'A': abla_method_prompt.A_prompt,
        'B': abla_method_prompt.B_prompt,
        'C': abla_method_prompt.C_prompt,
        'D': abla_method_prompt.D_prompt,
        'E': abla_method_prompt.E_prompt,
        'F': abla_method_prompt.F_prompt,
        'M': abla_method_prompt.M_prompt
    }
    
    # 9个策略池的额外方法
    extended_prompts_9 = {
        'G': abla_method_prompt.G_prompt,
        'H': abla_method_prompt.H_prompt,
        'I': abla_method_prompt.I_prompt
    }
    
    # 15个策略池的额外方法
    extended_prompts_15 = {
        'J': abla_method_prompt.J_prompt,
        'K': abla_method_prompt.K_prompt,
        'L': abla_method_prompt.L_prompt,
        'N': abla_method_prompt.N_prompt,
        'O': abla_method_prompt.O_prompt,
        'P': abla_method_prompt.P_prompt
    }
    
    # 根据策略池大小构建可用的方法提示字典
    if strategy_pool_size == 6:
        prompts = base_prompts
    elif strategy_pool_size == 9:
        prompts = base_prompts.copy()
        prompts.update(extended_prompts_9)
    elif strategy_pool_size == 15:
        prompts = base_prompts.copy()
        prompts.update(extended_prompts_9)
        prompts.update(extended_prompts_15)
    else:
        # 默认12个方法（原始版本）
        prompts = base_prompts.copy()
        prompts.update(extended_prompts_9)
        prompts.update({
            'J': abla_method_prompt.J_prompt,
            'K': abla_method_prompt.K_prompt,
            'L': abla_method_prompt.L_prompt
        })
    
    return prompts.get(method_code, "None")

def get_first_user_msg(problem, options=None, method_code=None, strategy_pool_size=12):
    # 获取解题方法提示
    method_prompt = get_method_prompt(method_code, strategy_pool_size) if method_code else abla_method_prompt.M_prompt
        
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
    """根据指定规则选择最佳结果
    
    策略：
    1. 默认使用Majority Voting：选择最频繁的答案
    2. 在平局或所有答案都不同的情况下，使用基于分数的投票机制：
       计算每条路径的平均步骤分数，选择平均分数最高的路径
    """
    # 过滤出成功完成的结果
    finished_results = [r for r in results if r.get("stop_reason") == "finished" and r.get("answer") is not None]
    
    if not finished_results:
        logging.warning("所有推理都未能完成或提取答案")
        return None
    
    # 统计答案
    answers = [r.get("answer") for r in finished_results]
    answer_counts = Counter(answers)
    
    # 检查是否有明确的多数答案
    most_common_answer, max_count = answer_counts.most_common(1)[0]
    
    # 如果有唯一的多数答案（不是平局）
    if max_count > 1 and len([count for count in answer_counts.values() if count == max_count]) == 1:
        # 找出具有最常见答案的所有结果
        selected_results = [r for r in finished_results if r.get("answer") == most_common_answer]
        # 从中选择token数最多的
        selected_result = max(selected_results, key=lambda r: r.get("num_tokens", 0))
        selected_result["selection_reason"] = f"多数原则: {max_count}/{len(finished_results)}个结果给出相同答案"
        return selected_result
    
    # 平局或所有答案都不同的情况：使用基于平均分数的投票机制
    logging.info("检测到平局或答案各不相同，使用基于平均分数的投票机制")
    
    # 为每个结果计算平均步骤分数
    def calculate_avg_score(result):
        """计算结果的平均SPM分数（用于答案聚合）"""
        return result.get("avg_spm_score", 0.0)
    
    # 显示每个结果的平均分数
    for result in finished_results:
        method_code = result.get("method_code", "未知")
        avg_spm_score = calculate_avg_score(result)
        avg_original_score = result.get("avg_score", 0.0)
        answer = result.get("answer")
        logging.info(f"方法 {method_code}: 答案={answer}, 原始平均分数={avg_original_score:.2f}, SPM平均分数={avg_spm_score:.2f}")
    
    # 计算每个结果的平均分数并选择最高的
    selected_result = max(finished_results, key=calculate_avg_score)
    avg_score = calculate_avg_score(selected_result)
    
    if max_count > 1:
        selected_result["selection_reason"] = f"平局情况下基于平均分数选择 (平均分数: {avg_score:.2f})"
    else:
        selected_result["selection_reason"] = f"答案各不相同，基于平均分数选择 (平均分数: {avg_score:.2f})"
    
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
            model="Qwen/QwQ-32B",
            # model="/hpc2hdd/home/bwang423/.cache/modelscope/hub/models/Qwen/QwQ-32B",
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
    parser.add_argument("--output_dir", type=str, default="results/spec_scale", 
                        help="Where result pickle files will be written to")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallel size for model initialization")
    parser.add_argument("--method_num", type=int, default=3,
                        help="Number of solution methods to use (default: 3)")
    parser.add_argument("--strategy_pool_size", type=int, choices=[6, 9, 12, 15], default=12,
                        help="Size of strategy pool for ablation study (6, 9, 12, 15)")
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

def choose_solution_methods(problem, options, dataset_name, models, method_num=3, strategy_pool_size=12):
    """选择最合适的解题方法
    
    Args:
        problem: 问题文本
        options: 选项（如有）
        dataset_name: 数据集名称
        models: 模型字典
        method_num: 需要选择的解题方法数量，默认为3
        strategy_pool_size: 策略池大小，默认为12
        
    Returns:
        包含方法代码的列表
    """
    logging.info(f"选择{method_num}个解题方法（策略池大小: {strategy_pool_size}）...")
    
    # 根据策略池大小选择合适的prompt
    if strategy_pool_size == 6:
        choose_prompt = method_selection_prompt_6
    elif strategy_pool_size == 9:
        choose_prompt = method_selection_prompt_9
    elif strategy_pool_size == 15:
        choose_prompt = method_selection_prompt_15
    else:
        # 默认使用9个策略池的prompt
        choose_prompt = method_selection_prompt_9
        logging.warning(f"策略池大小 {strategy_pool_size} 不在预定义范围内，使用9个策略池的prompt")
    
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

    # 根据策略池大小确定有效的方法代码范围
    if strategy_pool_size == 6:
        valid_methods = r'[A-F]|M'
        fallback_method = 'M'
    elif strategy_pool_size == 9:
        valid_methods = r'[A-I]|M'
        fallback_method = 'M'
    elif strategy_pool_size == 15:
        valid_methods = r'[A-P]'
        fallback_method = 'M'
    else:
        # 默认12个策略池
        valid_methods = r'[A-M]'
        fallback_method = 'M'
    
    # 提取方法代码
    methods_match = re.findall(valid_methods, method_text)
    
    # 如果提取的方法数量不足，使用fallback方法填充
    if len(methods_match) < method_num:
        logging.warning(f"从'{method_text}'中只提取到{len(methods_match)}个方法代码，不足{method_num}个，使用'{fallback_method}'填充")
        while len(methods_match) < method_num:
            methods_match.append(fallback_method)
    elif len(methods_match) > method_num:
        # 如果提取的方法数量超过需要的数量，截取前method_num个
        logging.warning(f"从'{method_text}'中提取到{len(methods_match)}个方法代码，超过{method_num}个，只使用前{method_num}个")
        methods_match = methods_match[:method_num]
    
    logging.info(f"选择的解题方法: {''.join(methods_match)}")

    return methods_match

def run_reasoning_process(args, problem, options):
    """运行推理过程"""
    start_time_total = time.time()  # 记录整个过程的开始时间
    
    try:
        # 使用大模型选择解题方法，使用args.method_num参数
        solution_methods = choose_solution_methods(problem, options, args.dataset_name, models, args.method_num, args.strategy_pool_size)
        
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
                    prompt = get_first_user_msg(problem, options, method_code=original_method_code, strategy_pool_size=args.strategy_pool_size)
                else:  # 继续之前的消息
                    steps_so_far_str = "\n\n".join(track["steps"]) + "\n\n"
                    prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code, strategy_pool_size=args.strategy_pool_size)}\n<think>{steps_so_far_str}"
                
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
                
                score_prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code, strategy_pool_size=args.strategy_pool_size)}\n<think>{steps_so_far_str}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
                
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
                        prompt = get_first_user_msg(problem, options, method_code=original_method_code, strategy_pool_size=args.strategy_pool_size)
                    else:  # 继续之前的消息
                        steps_so_far_str = "\n\n".join(track["steps"]) + "\n\n"
                        prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code, strategy_pool_size=args.strategy_pool_size)}\n<think>{steps_so_far_str}"
                    
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
                
                # 确定SPM优化分数：重写的步骤分配9分，未重写的使用原始评分
                spm_score = 9.0 if used_base_model else scores[method]
                
                if used_base_model:
                    logging.info(f"[Step {step_id}] 方法 {method} 使用重写步骤，SPM分数从 {scores[method]} 调整为 9.0")
                
                # 创建元数据
                metadata = {
                    "step_id": step_id,
                    "method_code": original_method_code,
                    "step_str": step_text,
                    "small_model_step": small_results[method]["text"],
                    "num_output_tokens_small": small_results[method]["tokens"],
                    "score": scores[method],  # 保持原始评分
                    "spm_score": spm_score,  # SPM优化分数，用于答案聚合
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

        # 计算改写率
        # rewrite_rates = {}
        total_rewrite_rate = 0.0
        valid_methods = 0
        
        for method, track in method_tracks.items():
            if track["total_steps"] > 0:
                rewrite_rate = track["rewrite_count"] / track["total_steps"] * 100
                track["rewrite_rate"] = rewrite_rate
                total_rewrite_rate += rewrite_rate
                valid_methods += 1
                logging.info(f"方法 {method} 改写率: {rewrite_rate:.2f}% ({track['rewrite_count']}/{track['total_steps']})")
        
        # 计算平均改写率
        avg_rewrite_rate = total_rewrite_rate / valid_methods if valid_methods > 0 else 0
        logging.info(f"平均改写率: {avg_rewrite_rate:.2f}%")

    except ValueError as e:
        logging.error(f"ValueError caught in chat template application: {e}")
        method_tracks = {}
        total_time = 0
        avg_rewrite_rate = 0
         
    return method_tracks, total_time, avg_rewrite_rate

def process_results(method_tracks, total_time, avg_rewrite_rate, args, problem, options):
    """处理结果并保存"""
    # 合并所有方法的元数据
    all_metadata = []
    for method, track in method_tracks.items():
        for metadata in track["metadata"]:
            all_metadata.append(metadata)

    # 为每个方法选择最终结果
    final_results = {}
    for method, track in method_tracks.items():
        if track["finished"]:
            # 提取答案
            answer = extract_final_answer("\n\n".join(track["steps"]), args.dataset_name)
            
            # 计算平均步骤分数
            scores = [m.get("score") for m in track["metadata"] if m.get("score") is not None]
            spm_scores = [m.get("spm_score") for m in track["metadata"] if m.get("spm_score") is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            avg_spm_score = sum(spm_scores) / len(spm_scores) if spm_scores else 0.0
            
            final_results[method] = {
                "steps": track["steps"],
                "answer": answer,
                "stop_reason": track.get("stop_reason", "unknown"),
                "num_tokens": sum([m["final_num_output_tokens"] for m in track["metadata"]]),
                "num_steps": len(track["steps"]),
                "method_code": track["method_code"],  # 记录使用的方法代码
                "avg_score": avg_score,  # 原始平均分数
                "avg_spm_score": avg_spm_score,  # SPM平均分数，用于答案聚合
                "step_scores": scores,  # 原始分数列表
                "spm_step_scores": spm_scores  # SPM分数列表
            }

    # 选择最佳结果
    best_result = select_best_result(list(final_results.values()))

    # 添加统计信息
    metadata_with_stats = {
        "problem_id": args.problem_id,
        "repeat_id": args.repeat_id,
        "dataset_name": args.dataset_name,
        "problem": problem,
        "options": options,
        "method_num": args.method_num,  # 记录使用的方法数量
        "strategy_pool_size": args.strategy_pool_size,  # 记录策略池大小
        "solution_methods": list(method_tracks.keys()),  # 记录使用的解题方法
        "method_tracks": method_tracks,
        "final_results": final_results,
        "best_result": best_result,
        "total_time": total_time,
        "score_threshold": args.score_threshold,
        "token_budget": args.token_budget,
        "score_method": args.score_method,
        "avg_rewrite_rate": avg_rewrite_rate
    }
    
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
    output_dir = os.path.join(args.output_dir + f"_m{args.method_num}_t{args.score_threshold}_pool{args.strategy_pool_size}", args.dataset_name)
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
            method_tracks, total_time, avg_rewrite_rate = run_reasoning_process(args, problem, options)
            
            # 处理结果
            metadata_with_stats = process_results(method_tracks, total_time, avg_rewrite_rate, args, problem, options)
            
            # 保存结果
            save_results(metadata_with_stats, output_filename)
            
            completed_tasks += 1
            logging.info(f"完成进度: {completed_tasks}/{total_tasks * args.repeat_id} ({completed_tasks/(total_tasks * args.repeat_id)*100:.2f}%)")
    
    logging.info(f"所有任务完成！共处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次")

if __name__ == "__main__":
    main()


