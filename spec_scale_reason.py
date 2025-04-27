# python spec_scale_reason.py --dataset_name aime --problem_id 60 --repeat_id 1 --output_dir results/spec_scale_Inf --score_threshold 7.0 --token_budget 8192 --score_method greedy

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
import re

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
    print(logprobs)
    print(token)
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

def extract_answer(reasoning):
    """从推理文本中提取最终答案"""
    if not reasoning:
        return None
        
    # 使用正则表达式查找 \boxed{answer} 模式
    match = re.search(r"\\boxed\{(.+?)\}", reasoning)
    if match:
        answer = match.group(1).strip()
        return answer
    else:
        # 备用方案：检查最后一步是否包含"Answer: X"
        match_answer = re.search(r"[Aa]nswer:\s*([A-Za-z0-9]+)", reasoning)
        if match_answer:
            return match_answer.group(1).upper()
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
parser.add_argument("--output_dir", type=str, default="results/spec_scale_Inf", 
                    help="Where result pickle files will be written to")
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args.dataset = get_dataset(args.dataset_name)

# 初始化模型
logging.info("正在初始化模型...")
models = {}
try:
    # 初始化小模型
    logging.info("正在加载1.5b模型...")
    models["1.5b"] = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_parallel_size=4,  # 根据您的GPU数量调整
        gpu_memory_utilization=0.1,
        trust_remote_code=True
    )
    
    # 初始化大模型
    logging.info("正在加载32b模型...")
    models["32b"] = LLM(
        model="Qwen/QwQ-32B",
        tensor_parallel_size=4,  # 根据您的GPU数量调整
        gpu_memory_utilization=0.8,
        trust_remote_code=True
    )
    logging.info("模型初始化完成")
except Exception as e:
    logging.error(f"模型初始化失败: {e}")
    exit(1)

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
start_time_total = time.time()  # 记录整个过程的开始时间
try:
    # 为每个角色创建独立的推理轨道
    role_tracks = {
        1: {"steps": [], "finished": False, "metadata": []},
        2: {"steps": [], "finished": False, "metadata": []},
        3: {"steps": [], "finished": False, "metadata": []}
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
        small_outputs = models["1.5b"].generate(active_prompts, sampling_params)
        
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
        
        score_outputs = models["32b"].generate(score_prompts, scoring_params)
        
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
            rewrite_outputs = models["32b"].generate(rewrite_prompts, sampling_params)
            
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
            else:
                # 使用小模型的结果
                step_text = small_results[role]["text"]
                num_tokens = small_results[role]["tokens"]
                finished = small_results[role]["finished"]
                used_base_model = False
            
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
        
except ValueError as e:
    logging.error(f"ValueError caught in chat template application: {e}")

# 计算总时间和统计信息
total_time = time.time() - start_time_total

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
        answer = extract_answer("\n\n".join(track["steps"]))
        
        final_results[role] = {
            "steps": track["steps"],
            "metadata": track["metadata"],
            "stop_reason": track.get("stop_reason", "unknown"),
            "answer": answer,
            "total_tokens": sum([m["final_num_output_tokens"] for m in track["metadata"]])
        }

# 选择最佳结果
best_results = [
    {
        "role_type": role,
        "reasoning": "\n\n".join(result["steps"]),
        "answer": result["answer"],
        "num_tokens": result["total_tokens"],
        "stop_reason": result["stop_reason"]
    }
    for role, result in final_results.items()
    if result["answer"] is not None
]

best_result = select_best_result(best_results) if best_results else None

# 记录结果
logging.info("=" * 50)
logging.info("时间统计信息:")
logging.info(f"问题ID: {args.problem_id}, 重复ID: {args.repeat_id}")
logging.info(f"总时间: {total_time:.2f}秒")

for role, result in final_results.items():
    tokens = result["total_tokens"]
    logging.info(f"角色 {role}:")
    logging.info(f"  - 总token数: {tokens}")
    logging.info(f"  - 停止原因: {result['stop_reason']}")
    if result["answer"]:
        logging.info(f"  - 答案: {result['answer']}")

if best_result:
    logging.info(f"选择的角色: {best_result['role_type']}")
    logging.info(f"选择原因: {best_result.get('selection_reason', '未知')}")
    logging.info(f"最终答案: {best_result.get('answer', '无法提取')}")
else:
    logging.info("无法选择有效结果: 所有推理都失败")

logging.info("=" * 50)

# 保存结果
metadata_with_stats = {
    "reasoning": best_result if best_result else {"reasoning": "所有推理都失败", "stop_reason": "failed"},
    "time_stats": {
        "total_time": total_time,
        "role_results": final_results,
        "selected_role": best_result.get("role_type") if best_result else None,
        "selection_reason": best_result.get("selection_reason") if best_result else "所有推理都失败"
    },
    "all_metadata": all_metadata  # 保存所有步骤的详细信息
}

os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_with_stats, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_with_stats, stream=f)


