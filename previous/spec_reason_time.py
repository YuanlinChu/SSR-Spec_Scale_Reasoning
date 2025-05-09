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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_avg_score(scores):
    return statistics.mean([x for x in scores if x is not None])

def get_frequency(scores):
    return dict(Counter(scores))

def get_model(model_size):
    client = clients[model_size]
    models = client.models.list()
    model = models.data[0].id
    return model

# %%
model_names = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "32b": "Qwen/QwQ-32B",
}
ports = {
    "1.5b": "30001",
    "32b": "30000",
}
clients = {}
for size, full_name in model_names.items():
    clients[size] = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{ports[size]}/v1",
    )

def get_first_user_msg(problem, options=None):
    if options is None:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        return system_prompt.format(problem=problem)
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
        return system_prompt.format(
            problem=problem,
            ans_a=options["A"],
            ans_b=options["B"],
            ans_c=options["C"],
            ans_d=options["D"],
        )

# %%
def generate_new_step(problem, steps_so_far, model_size, options=None, stop_token="\n\n"):
    client = clients[model_size]
    
    start_time = time.time()  # 记录开始时间
    
    if steps_so_far == []:  # first step
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
        ]
        extra_body = {"add_generation_prompt": True}
    else:  # continuing on from a previous message
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ]
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.6, top_p=0.95, # https://huggingface.co/Qwen/QwQ-32B#usage-guidelines
        max_tokens=512,
        stop=[stop_token],
        extra_body=extra_body,
    )

    step_str = response.choices[0].message.content
    # num_input_tokens = response.usage.prompt_tokens
    num_output_tokens = response.usage.completion_tokens
    # finished = "boxed" in step_str
    finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]])
    
    elapsed_time = time.time() - start_time  # 计算耗时
    
    # 计算每个token的平均时间（毫秒）
    time_per_token = (elapsed_time * 1000) / num_output_tokens if num_output_tokens > 0 else 0
    
    return step_str, finished, num_output_tokens, elapsed_time, time_per_token


def get_score(args, problem, steps_so_far, model_size="32b", options=None):
    client = clients[model_size]
    
    start_time = time.time()  # 记录开始时间
    
    steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
        {"role": "assistant", "content": f"<think>{steps_so_far_str}"},  # a </think> cannot be added at the end, otherwise, none of the previous steps will be encoded
        {"role": "user", "content": "Evaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9."},
        {"role": "assistant", "content": "<think>I think the quality score is: "},
    ]
    
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.0,
        max_tokens=1,
        logprobs=True,  # the docs said that this should be an int hmmmmmm https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        top_logprobs=10,  # https://github.com/vllm-project/vllm/issues/13881 
        extra_body={
            "add_generation_prompt": False, "continue_final_message": True,
            # "return_tokens_as_token_ids": True,
        },
    )
    justification = response.choices[0].message.content
    
    score = process_logprobs(response, method=args.score_method)
    
    elapsed_time = time.time() - start_time  # 计算耗时
    
    return score, justification, response, elapsed_time


def process_logprobs(response, method, temp=1.0):
    assert len(response.choices[0].logprobs.content) == 1
    token = response.choices[0].logprobs.content[0].token# example: '1', '0'
    token_logprobs = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
    logging.info(f"Original token_logprobs: {token_logprobs}")
    token_logprobs = {k: v for k, v in token_logprobs.items() if k.isdigit()}  # filter out non-digit values

    if method == "greedy":
        # return the vanilla response
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        # Convert log probabilities to probabilities and normalize each distribution.
        probs = {tok: np.exp(lp / temp) for tok, lp in token_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if i not in probs:
                probs[i] = 0
        logging.info(f"Avg score: {sum([int(t) * p for t, p in probs.items()])}")
        return sum([int(t) * p for t, p in probs.items()])
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
parser.add_argument("--output_dir", type=str, default="results/test/math", 
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

# 初始化时间统计变量
small_model_times = []
large_model_scoring_times = []
large_model_generation_times = []
small_model_step_count = 0
large_model_step_count = 0
total_small_model_time = 0
total_large_model_scoring_time = 0
total_large_model_generation_time = 0

# 初始化token时间统计变量
small_model_token_times = []  # 小模型每个token的时间列表
large_model_token_times = []  # 大模型每个token的时间列表

start_time_total = time.time()  # 记录整个过程的开始时间

try:
    while True:
        warning_flag = False
        
        # 1. 使用小模型生成推理步骤
        step_str, finished, num_output_tokens, small_model_time, small_token_time = generate_new_step(problem, steps_so_far, "1.5b", options=options)
        small_model_step, num_output_tokens_small = step_str, num_output_tokens
        small_model_times.append(small_model_time)
        small_model_token_times.append(small_token_time)  # 记录小模型每个token的时间
        total_small_model_time += small_model_time
        small_model_step_count += 1
        
        # 2. 使用大模型对步骤进行评分
        score, justification, response, scoring_time = get_score(args, problem, steps_so_far + [step_str], options=options)
        large_model_scoring_times.append(scoring_time)
        total_large_model_scoring_time += scoring_time
        
        # 3. 如果分数超过阈值，接受小模型的步骤；否则，使用大模型重新生成
        if score is not None and score >= args.score_threshold:
            logging.info(f"[Step {step_id}] score {score}, accepted! Time: {small_model_time:.2f}s, Token time: {small_token_time:.2f}ms/token")
            base_model_step, num_output_tokens_base, large_model_time, large_token_time = None, None, 0, 0
        else:
            logging.info(f"[Step {step_id}] score {score} rejected, falling back to base model")
            step_str, finished, num_output_tokens, large_model_time, large_token_time = generate_new_step(problem, steps_so_far, "32b", options=options)
            base_model_step, num_output_tokens_base = step_str, num_output_tokens
            large_model_generation_times.append(large_model_time)
            large_model_token_times.append(large_token_time)  # 记录大模型每个token的时间
            total_large_model_generation_time += large_model_time
            large_model_step_count += 1
            logging.info(f"[Step {step_id}] Large model generation time: {large_model_time:.2f}s, Token time: {large_token_time:.2f}ms/token")
        
        if "</think>" in step_str and not any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]]):
            # FIXME(ruipan): handles a very rare edge case of generating a stop thinking token midway through answering.
            # Although it could be that thinking finished, but the last step didn't format the answer with \boxed{}
            logging.warning(f"Warning: step_str had a </think>, removing. {step_str}")
            step_str = step_str.replace("</think>", "")
            warning_flag = True
        
        # 4. 重复直到生成答案
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
            "small_model_time": small_model_time,
            "small_model_token_time": small_token_time,  # 毫秒/token
            "scoring_time": scoring_time,
            "large_model_time": large_model_time if large_model_time else 0,
            "large_model_token_time": large_token_time if large_token_time else 0,  # 毫秒/token
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

# 计算总时间和各部分时间占比
total_time = time.time() - start_time_total
total_model_time = total_small_model_time + total_large_model_scoring_time + total_large_model_generation_time

# 计算平均时间
avg_small_model_time = sum(small_model_times) / len(small_model_times) if small_model_times else 0
avg_large_model_scoring_time = sum(large_model_scoring_times) / len(large_model_scoring_times) if large_model_scoring_times else 0
avg_large_model_generation_time = sum(large_model_generation_times) / len(large_model_generation_times) if large_model_generation_times else 0

# 计算平均token时间
avg_small_model_token_time = sum(small_model_token_times) / len(small_model_token_times) if small_model_token_times else 0
avg_large_model_token_time = sum(large_model_token_times) / len(large_model_token_times) if large_model_token_times else 0

# 计算大模型重写步骤的比例
large_model_rewrite_ratio = large_model_step_count / (small_model_step_count + large_model_step_count) if (small_model_step_count + large_model_step_count) > 0 else 0

# 计算最终推理结果的总token数
total_tokens = sum([m["final_num_output_tokens"] for m in metadata_list])

# 添加时间统计信息到元数据
time_stats = {
    "total_time": total_time,
    "small_model_stats": {
        "total_time": total_small_model_time,
        "avg_time": avg_small_model_time,
        "step_count": small_model_step_count,
        "time_percentage": (total_small_model_time / total_time) * 100 if total_time > 0 else 0,
        "avg_token_time": avg_small_model_token_time,  # 毫秒/token
    },
    "large_model_scoring_stats": {
        "total_time": total_large_model_scoring_time,
        "avg_time": avg_large_model_scoring_time,
        "step_count": len(large_model_scoring_times),
        "time_percentage": (total_large_model_scoring_time / total_time) * 100 if total_time > 0 else 0,
    },
    "large_model_generation_stats": {
        "total_time": total_large_model_generation_time,
        "avg_time": avg_large_model_generation_time,
        "step_count": large_model_step_count,
        "time_percentage": (total_large_model_generation_time / total_time) * 100 if total_time > 0 else 0,
        "avg_token_time": avg_large_model_token_time,  # 毫秒/token
    },
    "large_model_rewrite_ratio": large_model_rewrite_ratio * 100,  # 转换为百分比
    "token_time_comparison": {
        "small_model_avg_token_time": avg_small_model_token_time,  # 毫秒/token
        "large_model_avg_token_time": avg_large_model_token_time,  # 毫秒/token
        "ratio": (avg_large_model_token_time / avg_small_model_token_time) if avg_small_model_token_time > 0 else 0  # 大模型/小模型
    },
    "total_tokens": total_tokens  # 添加最终推理结果的总token数
}

# 打印时间统计信息
logging.info("=" * 50)
logging.info("时间统计信息:")
logging.info(f"总时间: {total_time:.2f}秒")
logging.info(f"小模型生成步骤:")
logging.info(f"  - 总时间: {total_small_model_time:.2f}秒 ({(total_small_model_time/total_time)*100:.2f}%)")
logging.info(f"  - 平均时间: {avg_small_model_time:.2f}秒")
logging.info(f"  - 平均token时间: {avg_small_model_token_time:.2f}毫秒/token")
logging.info(f"  - 步骤数: {small_model_step_count}")
logging.info(f"大模型评分:")
logging.info(f"  - 总时间: {total_large_model_scoring_time:.2f}秒 ({(total_large_model_scoring_time/total_time)*100:.2f}%)")
logging.info(f"  - 平均时间: {avg_large_model_scoring_time:.2f}秒")
logging.info(f"  - 步骤数: {len(large_model_scoring_times)}")
logging.info(f"大模型生成步骤:")
logging.info(f"  - 总时间: {total_large_model_generation_time:.2f}秒 ({(total_large_model_generation_time/total_time)*100:.2f}%)")
logging.info(f"  - 平均时间: {avg_large_model_generation_time:.2f}秒")
logging.info(f"  - 平均token时间: {avg_large_model_token_time:.2f}毫秒/token")
logging.info(f"  - 步骤数: {large_model_step_count}")
logging.info(f"大模型重写步骤比例: {large_model_rewrite_ratio*100:.2f}%")
logging.info(f"Token时间比较:")
logging.info(f"  - 小模型平均token时间: {avg_small_model_token_time:.2f}毫秒/token")
logging.info(f"  - 大模型平均token时间: {avg_large_model_token_time:.2f}毫秒/token")
logging.info(f"  - 比例(大/小): {(avg_large_model_token_time/avg_small_model_token_time):.2f}倍")
logging.info(f"最终推理结果总token数: {total_tokens}")
logging.info("=" * 50)

os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

# 将时间统计信息添加到元数据
metadata_with_stats = {
    "steps": metadata_list,
    "time_stats": time_stats
}

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_with_stats, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_with_stats, stream=f)

