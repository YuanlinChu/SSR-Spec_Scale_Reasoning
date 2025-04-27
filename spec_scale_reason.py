# %%
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
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_avg_score(scores):
    return statistics.mean([x for x in scores if x is not None])

def get_frequency(scores):
    return dict(Counter(scores))

def get_model(model_size):
    return models[model_size]  # 直接返回vLLM模型实例

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

def generate_new_step_parallel(problem, steps_so_far, model_size, options=None, stop_token="\n\n"):
    model = models[model_size]

    # start_time = time.time()
    # 创建三个不同角色的提示
    prompts = []
    for role_type in [1, 2, 3]:
        if steps_so_far == []:  # first step
            prompt = get_first_user_msg(problem, options, role_type=role_type)
        else:  # continuing on from a previous message
            steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
            # 根据vLLM的格式构建提示，可能需要调整
            prompt = f"{get_first_user_msg(problem, options, role_type=role_type)}\n<think>{steps_so_far_str}"
        
        prompts.append(prompt)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=512,
        stop=[stop_token]           #这里可能会有达到512还没有出现stop_token的情况
    )
    
    # 批量生成
    outputs = model.generate(prompts, sampling_params)
    
    # 处理输出结果
    results = []
    total_tokens = 0
    is_finished = False
    
    for output in outputs:
        generated_text = output.outputs[0].text
        # 计算生成的token数量
        num_output_tokens = len(output.outputs[0].token_ids)
        
        # 检查是否完成
        finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
        
        results.append((generated_text, finished, num_output_tokens))
        total_tokens += num_output_tokens
        if finished:
            is_finished = True
    
    # elapsed_time = time.time() - start_time
    
    return results, is_finished, total_tokens

# 同样修改评分函数
def get_score_parallel(args, problem, steps_so_far, model_size="32b", options=None):
    """使用vLLM原生API进行评分"""
    model = models[model_size]
    
    steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
    
    # 构建评分提示
    prompt = f"{get_first_user_msg(problem, options)}\n<think>{steps_so_far_str}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
    
    # 设置采样参数，使用logprobs获取概率分布
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=10  # 获取前10个最可能的token的概率
    )
    
    # 生成评分
    output = model.generate([prompt], sampling_params)[0]
    
    # 获取生成的token及其概率
    generated_token = output.outputs[0].text
    token_logprobs = output.outputs[0].logprobs[0]
    
    # 处理logprobs，类似于原来的process_logprobs函数
    # 这部分需要根据vLLM API的实际返回格式调整
    score = process_vllm_logprobs(generated_token, token_logprobs, method=args.score_method)
    
    return score, generated_token, output

def process_vllm_logprobs(token, logprobs, method, temp=1.0):
    """处理vLLM返回的logprobs"""
    # 过滤出数字token的概率
    digit_logprobs = {k: v for k, v in logprobs.items() if k.isdigit()}
    
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





# %%
parser = argparse.ArgumentParser(description="Runs speculative reasoning using a small model")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="aime",
                    help="Dataset")
parser.add_argument("--score_threshold", type=float, default=7.0, 
                    help="Acceptance threshold")
parser.add_argument("--token_budget", type=int, default=8192,
                    help="Max num of total output tokens in each step")
# problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA
parser.add_argument("--problem_id", type=int, default=60,
                    help="Query ID (60-89 for AIME)")
parser.add_argument("--repeat_id", type=int, default=0,
                    help="Repeat ID (0-15, k=16)")
parser.add_argument("--score_method", type=str, choices=["greedy", "average"], default="greedy",
                    help="Scoring method")
parser.add_argument("--output_dir", type=str, default="results/only_Inf", 
                    help="Where result pickle files will be written to")
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args.dataset = get_dataset(args.dataset_name)

# %%
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

problem_id = f"{args.dataset_name}_{args.problem_id}"


output_filename = os.path.join(args.output_dir, f"{args.problem_id}/{args.repeat_id}")
if os.path.exists(f"{output_filename}.pickle"):
    logging.info(f"Problem {args.problem_id} repeat {args.repeat_id} resolved, exiting")
    exit()
    
steps_so_far = []
step_id = 0
metadata_list = []

model_names = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "32b": "Qwen/QwQ-32B",
}

# 初始化vLLM模型
models = {}
for size, full_name in model_names.items():
    # 根据模型大小选择合适的配置
    tensor_parallel_size = 2
    
    models[size] = LLM(
        model=full_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,  # 可以根据实际情况调整
        trust_remote_code=True,
        dtype="bfloat16",  # 或者 "float16"，根据您的硬件支持情况选择
    )
    logging.info(f"已加载模型 {full_name}")


start_time_total = time.time()  # 记录整个过程的开始时间
try:
    while True:
        warning_flag = False
        
        # 1. generate a reasoning step using a small model
        step_str, finished, num_output_tokens = generate_new_step_parallel(problem, steps_so_far, "1.5b", options=options)
        small_model_step, num_output_tokens_small = step_str, num_output_tokens

        # 2. use the base model to score the step
        score, justification, response = get_score_parallel(args, problem, steps_so_far + [step_str], options=options)
        
        # 3. if over a threshold, accept. else, generate a reasoning step using the base model.
        if score is not None and score >= args.score_threshold:
            logging.info(f"[Step {step_id}] score {score}, accepted!")
            base_model_step, num_output_tokens_base = None, None
        else:
            logging.info(f"[Step {step_id}] score {score} rejected, falling back to base model")
            step_str, finished, num_output_tokens = generate_new_step_parallel(problem, steps_so_far, "32b", options=options)
            base_model_step, num_output_tokens_base = step_str, num_output_tokens
        # NOTE(ruipan): potential optimization is to pipeline the decoding of these two models rather than sequentially
        
        if "</think>" in step_str and not any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]]):
            # FIXME(ruipan): handles a very rare edge case of generating a stop thinking token midway through answering.
            # Although it could be that thinking finished, but the last step didn't format the answer with \boxed{}
            logging.warning(f"Warning: step_str had a </think>, removing. {step_str}")
            step_str = step_str.replace("</think>", "")
            warning_flag = True
        
        # 4. repeat until an answer gets generated in the response
        steps_so_far.append(step_str)
        logging.info(f"[Step {step_id}] final step_str: {step_str}")
        
        metadata = {
            "step_id": step_id,
            "step_str": step_str,
            "small_model_step": small_model_step,
            "num_output_tokens_small": num_output_tokens_small,
            "score": score,
            "base_model_step": base_model_step,
            "num_output_tokens_base": num_output_tokens_base,
            "final_num_output_tokens": num_output_tokens_base if num_output_tokens_base is not None else num_output_tokens_small,
            "justification": justification,
        }
        if warning_flag:
            metadata["warning"] = "step_str had a </think>"
        metadata_list.append(metadata)
        step_id += 1
        
        if len(steps_so_far) > 2:
            finished = finished or steps_so_far[-1] == steps_so_far[-2]  # NOTE(ruipan): handles another edge case where model repeats previous reasoning steps
        
        if finished or sum([m["final_num_output_tokens"] for m in metadata_list]) >= args.token_budget:
            if sum([m["final_num_output_tokens"] for m in metadata_list]) >= args.token_budget:
                metadata_list[-1]["stop_reason"] = "budget"
            else:
                metadata_list[-1]["stop_reason"] = "finished"
            break
except ValueError:
    logging.error(f"ValueError caught in chat template application, continuing")

total_time = time.time() - start_time_total
total_tokens = sum([m["final_num_output_tokens"] for m in metadata_list])

logging.info(f"总时间: {total_time:.2f}秒")
logging.info(f"最终推理结果总token数: {total_tokens}")

os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_list, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_list, stream=f)


