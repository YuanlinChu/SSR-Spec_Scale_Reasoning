# python spec_scale_reason.py --dataset_name aime --problem_id 60 --repeat_id 1 --output_dir results/spec_scale_Inf --score_threshold 7.0 --token_budget 8192 --score_method greedy

import os
import time
import openai
import pickle
import pprint
import logging
import argparse
import numpy as np
from openai import OpenAI
import statistics
from collections import Counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
import re
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_first_user_msg(problem, options=None, role_type=1):
    from prompts.role_prompts import role_prompt_1, role_prompt_2, role_prompt_3
    if role_type == 1:
        role_prompt = role_prompt_1
    elif role_type == 2:
        role_prompt = role_prompt_2
    elif role_type == 3:
        role_prompt = role_prompt_3
    else:
        role_prompt = role_prompt_1
        
    if options is None:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        combined_prompt = f"{role_prompt}\n\n{system_prompt}"
        return combined_prompt.format(problem=problem)
    else:
        system_prompt = """
        What is the correct answer to the following problem? Please reason step by step. 
        Separate logical reasoning steps with two newline characters (\n\n).
        Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

        **Example output:** \\boxed{{A}}

        Problem: {problem}.
        Choices: 
        (A) {ans_a}
        (B) {ans_b}
        (C) {ans_c}
        (D) {ans_d}
        """
        combined_prompt = f"{role_prompt}\n\n{system_prompt}"
        return combined_prompt.format(
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

        logging.warning(f"在最终步骤中找不到\\boxed{{}}模式")
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
            gpu_memory_utilization=0.1,
            trust_remote_code=True
        )
        
        # 初始化大模型
        logging.info("正在加载目标模型...")
        models["target"] = LLM(
            model="Qwen/QwQ-32B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.8,
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

def run_reasoning_process(args, problem, options):
    """运行推理过程"""
    start_time_total = time.time()  # 记录整个过程的开始时间
    
    try:
        # 为每个角色创建独立的推理轨道
        role_tracks = {
            1: {"steps": [], "finished": False, "metadata": [], "rewrite_count": 0, "total_steps": 0},
            2: {"steps": [], "finished": False, "metadata": [], "rewrite_count": 0, "total_steps": 0},
            3: {"steps": [], "finished": False, "metadata": [], "rewrite_count": 0, "total_steps": 0}
        }
        
        step_id = 0
        
        while True:
            # 检查是否所有轨道都已完成
            active_roles = [role for role, track in role_tracks.items() if not track["finished"]]
            if not active_roles:
                break
                
            logging.info(f"[Step {step_id}] 活跃角色: {active_roles}")
            
            # 1. 为每个活跃角色并行生成下一步推理
            active_prompts = []
            role_mapping = []  # 用于跟踪每个提示对应的角色
            
            for role in active_roles:
                track = role_tracks[role]
                if not track["steps"]:  # 第一步
                    prompt = get_first_user_msg(problem, options, role_type=role)
                else:  # 继续之前的消息
                    steps_so_far_str = "\n\n".join(track["steps"]) + "\n\n"
                    prompt = f"{get_first_user_msg(problem, options, role_type=role)}\n<think>{steps_so_far_str}"
                
                active_prompts.append(prompt)
                role_mapping.append(role)
            
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
            score_role_mapping = []
            small_results = {}
            
            for i, output in enumerate(small_outputs):
                role = role_mapping[i]
                generated_text = output.outputs[0].text
                num_tokens = len(output.outputs[0].token_ids)
                
                # 检查是否完成
                finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
                
                # 存储小模型结果
                small_results[role] = {
                    "text": generated_text,
                    "tokens": num_tokens,
                    "finished": finished
                }
                
                # 准备评分提示
                track = role_tracks[role]
                steps_for_scoring = track["steps"] + [generated_text]
                steps_so_far_str = "\n\n".join(steps_for_scoring) + "\n\n"
                
                score_prompt = f"{get_first_user_msg(problem, options, role_type=role)}\n<think>{steps_so_far_str}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
                
                score_prompts.append(score_prompt)
                score_role_mapping.append(role)
            
            # 3. 批量评分
            scoring_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=10
            )
            
            score_outputs = models["target"].generate(score_prompts, scoring_params)
            
            # 4. 处理评分结果，确定哪些需要重写
            rewrite_prompts = []
            rewrite_role_mapping = []
            scores = {}
            
            for i, output in enumerate(score_outputs):
                role = score_role_mapping[i]
                token = output.outputs[0].text
                logprobs = output.outputs[0].logprobs[0]
                
                score = process_vllm_logprobs(token, logprobs, method=args.score_method)
                scores[role] = score
                
                # 如果分数低于阈值，需要重写
                if score is None or score < args.score_threshold:
                    logging.info(f"[Step {step_id}] 角色 {role} 分数 {score}，需要重写")
                    track = role_tracks[role]
                    
                    if not track["steps"]:  # 第一步
                        prompt = get_first_user_msg(problem, options, role_type=role)
                    else:  # 继续之前的消息
                        steps_so_far_str = "\n\n".join(track["steps"]) + "\n\n"
                        prompt = f"{get_first_user_msg(problem, options, role_type=role)}\n<think>{steps_so_far_str}"
                    
                    rewrite_prompts.append(prompt)
                    rewrite_role_mapping.append(role)
                else:
                    logging.info(f"[Step {step_id}] 角色 {role} 分数 {score}，已接受")
            
            # 5. 如果有需要重写的，批量重写
            rewrite_results = {}
            if rewrite_prompts:
                rewrite_outputs = models["target"].generate(rewrite_prompts, sampling_params)
                
                for i, output in enumerate(rewrite_outputs):
                    role = rewrite_role_mapping[i]
                    generated_text = output.outputs[0].text
                    num_tokens = len(output.outputs[0].token_ids)
                    
                    # 检查是否完成
                    finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
                    
                    rewrite_results[role] = {
                        "text": generated_text,
                        "tokens": num_tokens,
                        "finished": finished
                    }
            
            # 6. 更新每个角色的轨道
            for role in active_roles:
                track = role_tracks[role]
                
                # 确定最终使用的步骤
                if role in rewrite_results:
                    # 使用重写的结果
                    step_text = rewrite_results[role]["text"]
                    num_tokens = rewrite_results[role]["tokens"]
                    finished = rewrite_results[role]["finished"]
                    used_base_model = True
                    # 增加改写计数
                    track["rewrite_count"] += 1
                else:
                    # 使用小模型的结果
                    step_text = small_results[role]["text"]
                    num_tokens = small_results[role]["tokens"]
                    finished = small_results[role]["finished"]
                    used_base_model = False
                
                # 增加总步骤计数
                track["total_steps"] += 1
                
                # 处理特殊情况
                if "</think>" in step_text and not any([x in step_text for x in ["boxed", "Answer:", "ANSWER:"]]):
                    logging.warning(f"Warning: 角色 {role} 的步骤包含 </think>，正在移除")
                    step_text = step_text.replace("</think>", "")
                    warning_flag = True
                else:
                    warning_flag = False
                
                # 添加到轨道的步骤中
                track["steps"].append(step_text)
                
                # 创建元数据
                metadata = {
                    "step_id": step_id,
                    "role_type": role,
                    "step_str": step_text,
                    "small_model_step": small_results[role]["text"],
                    "num_output_tokens_small": small_results[role]["tokens"],
                    "score": scores[role],
                    "base_model_step": rewrite_results.get(role, {}).get("text") if used_base_model else None,
                    "num_output_tokens_base": rewrite_results.get(role, {}).get("tokens") if used_base_model else None,
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
        rewrite_rates = {}
        total_rewrite_rate = 0.0
        valid_roles = 0
        
        for role, track in role_tracks.items():
            if track["total_steps"] > 0:
                rewrite_rate = track["rewrite_count"] / track["total_steps"] * 100
                track["rewrite_rate"] = rewrite_rate
                total_rewrite_rate += rewrite_rate
                valid_roles += 1
                logging.info(f"角色 {role} 改写率: {rewrite_rate:.2f}% ({track['rewrite_count']}/{track['total_steps']})")
        
        # 计算平均改写率
        avg_rewrite_rate = total_rewrite_rate / valid_roles if valid_roles > 0 else 0
        logging.info(f"平均改写率: {avg_rewrite_rate:.2f}%")

    except ValueError as e:
        logging.error(f"ValueError caught in chat template application: {e}")
         
    return role_tracks, total_time, avg_rewrite_rate

def process_results(role_tracks, total_time, avg_rewrite_rate, args, problem, options):
    """处理结果并保存"""
    # 合并所有角色的元数据
    all_metadata = []
    for role, track in role_tracks.items():
        for metadata in track["metadata"]:
            all_metadata.append(metadata)

    # 为每个角色选择最终结果
    final_results = {}
    for role, track in role_tracks.items():
        if track["finished"]:
            # 提取答案
            answer = extract_final_answer("\n\n".join(track["steps"]), args.dataset_name)
            
            final_results[role] = {
                "steps": track["steps"],
                "answer": answer,
                "stop_reason": track.get("stop_reason", "unknown"),
                "num_tokens": sum([m["final_num_output_tokens"] for m in track["metadata"]]),
                "num_steps": len(track["steps"])
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
        "role_tracks": role_tracks,
        "final_results": final_results,
        "best_result": best_result,
        "total_time": total_time,
        "score_threshold": args.score_threshold,
        "token_budget": args.token_budget,
        "score_method": args.score_method,
        "avg_rewrite_rate":avg_rewrite_rate
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
            role_tracks, total_time, avg_rewrite_rate = run_reasoning_process(args, problem, options)
            
            # 处理结果
            metadata_with_stats = process_results(role_tracks, total_time, avg_rewrite_rate, args, problem, options)
            
            # 保存结果
            save_results(metadata_with_stats, output_filename)
            
            completed_tasks += 1
            logging.info(f"完成进度: {completed_tasks}/{total_tasks * args.repeat_id} ({completed_tasks/(total_tasks * args.repeat_id)*100:.2f}%)")
    
    logging.info(f"所有任务完成！共处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次")

if __name__ == "__main__":
    main()


