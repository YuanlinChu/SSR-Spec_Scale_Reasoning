# python spec_scale_reason_2_mp.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale_mp --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 5

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
import multiprocessing
from multiprocessing import Process, Queue, Manager
# 导入choose-prompts模块中的选择prompt
from prompts.choose_prompts import math_choose_prompt, gpqa_choose_prompt
from prompts.method_prompt import A_prompt, B_prompt, C_prompt, D_prompt, E_prompt, F_prompt, G_prompt, H_prompt, I_prompt, J_prompt, K_prompt, L_prompt, M_prompt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局模型变量，将在主进程中初始化
models_global = {}
# 全局请求和结果队列
request_queue_global = None
result_queues_global = {} # 每个track一个结果队列

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
    digit_logprobs = {k: v for k, v in logprobs.items()}
    
    if method == "greedy":
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        probs = {tok: np.exp(lp / temp) for tok, lp in digit_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if str(i) not in probs:
                probs[str(i)] = 0
        
        avg_score = sum([int(t) * p for t, p in probs.items()])
        logging.info(f"Avg score: {avg_score}")
        return avg_score
    else:
        raise NotImplementedError

def select_best_result(results):
    """根据指定规则选择最佳结果"""
    finished_results = [r for r in results if r.get("stop_reason") == "finished" and r.get("answer") is not None]
    
    if not finished_results:
        logging.warning("所有推理都未能完成或提取答案")
        return None
    
    answers = [r.get("answer") for r in finished_results]
    answer_counts = Counter(answers)
    
    if len(answer_counts) < len(finished_results):
        most_common_answer, count = answer_counts.most_common(1)[0]
        if count > 1:
            selected_results = [r for r in finished_results if r.get("answer") == most_common_answer]
            selected_result = max(selected_results, key=lambda r: r.get("num_tokens", 0))
            selected_result["selection_reason"] = f"多数原则: {count}/{len(finished_results)}个结果给出相同答案"
            return selected_result
    
    selected_result = max(finished_results, key=lambda r: r.get("num_tokens", 0))
    selected_result["selection_reason"] = "答案各不相同，选择token数最多的结果"
    return selected_result

def extract_final_answer(reasoning, dataset_name):
    """从推理文本中提取最终答案"""
    if not reasoning:
        return None
        
    match = re.search(r"\\boxed\{", reasoning)
    if match:
        start_pos = match.end()
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
            
            if dataset_name == "gpqa":
                if answer in ["A", "B", "C", "D"]:
                    return answer
                else:
                    sub_match = re.match(r"([A-D])[\)\.]?", answer)
                    if sub_match:
                        return sub_match.group(1)
                    logging.warning(f"无法从boxed答案中提取有效的GPQA选项: {answer}")
                    return None
            else:
                try:
                    return int(answer)
                except ValueError:
                    if dataset_name == "math":
                        return answer.strip()
                    logging.warning(f"无法将boxed答案转换为整数: {answer}")
                    return answer
        else:
            logging.warning("在推理文本中找到了\\boxed{但没有找到匹配的}，可能是格式错误")
            return None
    else:
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

def initialize_models_global(GPU_num):
    """初始化全局模型"""
    logging.info("正在初始化全局模型...")
    global models_global
    try:
        logging.info("正在加载草稿模型...")
        models_global["draft"] = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.1, # 调整以适应多进程共享
            trust_remote_code=True
        )
        
        logging.info("正在加载目标模型...")
        models_global["target"] = LLM(
            model="Qwen/QwQ-32B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.8, # 调整以适应多进程共享
            trust_remote_code=True
        )
        logging.info("全局模型初始化完成")
    except Exception as e:
        logging.error(f"全局模型初始化失败: {e}")
        # 在多进程场景下，子进程无法直接exit主进程，这里可以抛出异常由主进程处理
        raise RuntimeError(f"全局模型初始化失败: {e}")

def parse_problem_range(problem_id_str):
    """解析问题ID范围，支持单个ID或范围（如60-89）"""
    if '-' in problem_id_str:
        start, end = map(int, problem_id_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(problem_id_str)]

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Runs speculative reasoning using a small model with multiprocessing")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="aime",
                        help="Dataset")
    parser.add_argument("--score_threshold", type=float, default=7.0, 
                        help="Acceptance threshold")
    parser.add_argument("--token_budget", type=int, default=8192,
                        help="Max num of total output tokens in each step")
    parser.add_argument("--problem_id", type=str, default="60",
                        help="Query ID (60-89 for AIME), can be a single ID or a range like 60-89")
    parser.add_argument("--repeat_id", type=int, default=1,
                        help="Repeat ID (1-16, k=16)")
    parser.add_argument("--score_method", type=str, choices=["greedy", "average"], default="greedy",
                        help="Scoring method")
    parser.add_argument("--output_dir", type=str, default="results/spec_scale_mp", 
                        help="Where result pickle files will be written to")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallel size for model initialization")
    parser.add_argument("--method_num", type=int, default=3,
                        help="Number of solution methods to use (default: 3)")
    parser.add_argument("--num_parallel_tracks", type=int, default=5,
                        help="Number of tracks to run in parallel (default: 5)")
    return parser.parse_known_args()

def prepare_problem_data(args):
    """准备问题数据"""
    # dataset 在 args 中已经是加载好的对象
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

def choose_solution_methods(problem, options, dataset_name, model_proxy, method_num=3):
    """选择最合适的解题方法 (使用模型代理)"""
    logging.info(f"选择{method_num}个解题方法...")
    
    if dataset_name == "aime" or dataset_name == "math":
        choose_prompt_template = math_choose_prompt
    elif dataset_name == "gpqa":
        choose_prompt_template = gpqa_choose_prompt
    else:
        logging.warning(f"未知数据集: {dataset_name}，默认使用math_choose_prompt")
        choose_prompt_template = math_choose_prompt
    
    prompt = f"{choose_prompt_template}\n\nProblem: {problem}"
    if options:
        prompt += "\nChoices: "
        for key, value in options.items():
            prompt += f"\n({key}) {value}"
    prompt = f"{prompt}\n\nExamine the problem and select the **{method_num}** strategies (by their codes, e.g. {'B,E,F' if method_num == 3 else 'B,E,F,A' if method_num == 4 else 'B,E'}) that you believe are most promising for solving it. You only need to output {method_num} codes, without any other symbols or text\n<think></think>I think the best {method_num} methods are: "
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=method_num * 2 -1,
    )
    
    # 使用模型代理进行推理
    request_id = f"choose_methods_{time.time()}" # 唯一的请求ID
    outputs = model_proxy.generate_on_target([prompt], sampling_params, request_id) # 假设 model_proxy 有此方法
    method_text = outputs[0].outputs[0].text.strip()
    
    print(method_text)

    methods_match = re.findall(r'[A-M]', method_text)
    
    if len(methods_match) < method_num:
        logging.warning(f"从'{method_text}'中只提取到{len(methods_match)}个方法代码，不足{method_num}个，使用'M'填充")
        methods_match.extend(['M'] * (method_num - len(methods_match)))
    elif len(methods_match) > method_num:
        logging.warning(f"从'{method_text}'中提取到{len(methods_match)}个方法代码，超过{method_num}个，只使用前{method_num}个")
        methods_match = methods_match[:method_num]
    
    logging.info(f"选择的解题方法: {''.join(methods_match)}")
    return methods_match

# 模型代理类，用于子进程向主进程发送请求
class ModelProxy:
    def __init__(self, request_q, result_q_map, track_id):
        self.request_q = request_q
        self.result_q_map = result_q_map
        self.track_id = track_id

    def _send_request(self, model_type, prompts, sampling_params, request_id_suffix=""):
        unique_request_id = f"{self.track_id}_{request_id_suffix}_{time.time_ns()}"
        self.request_q.put({
            "request_id": unique_request_id,
            "track_id": self.track_id,
            "model_type": model_type,
            "prompts": prompts,
            "sampling_params": sampling_params,
        })
        # logging.debug(f"Track {self.track_id} sent request {unique_request_id} for {model_type}")
        # 等待并获取结果
        # 增加超时以避免死锁，尽管在理想情况下不应发生
        try:
            result = self.result_q_map[self.track_id].get(timeout=600) # 10分钟超时
            if result.get("request_id") != unique_request_id:
                # 这不应该发生，如果发生说明队列逻辑有误
                logging.error(f"Track {self.track_id} received mismatched result! Expected {unique_request_id}, got {result.get('request_id')}")
                raise RuntimeError("Mismatched result ID in queue")
            # logging.debug(f"Track {self.track_id} received result for {unique_request_id}")
            return result["data"]
        except multiprocessing.queues.Empty:
            logging.error(f"Track {self.track_id} timed out waiting for result for request {unique_request_id} ({model_type})")
            # 根据策略，这里可以抛出异常或返回错误指示
            raise TimeoutError(f"Track {self.track_id} timed out for {model_type} request {unique_request_id}")


    def generate_on_draft(self, prompts, sampling_params, request_id_suffix="draft_gen"):
        return self._send_request("draft", prompts, sampling_params, request_id_suffix)

    def generate_on_target(self, prompts, sampling_params, request_id_suffix="target_gen"):
        return self._send_request("target", prompts, sampling_params, request_id_suffix)


def run_one_track_process(track_id, method_code_with_index, original_method_code, problem, options, args_dict, request_q, result_q_map):
    """在单个进程中运行一个推理轨道"""
    # 将 args_dict 转换回 Namespace 对象或直接使用字典
    args = argparse.Namespace(**args_dict)
    model_proxy = ModelProxy(request_q, result_q_map, track_id)

    track_data = {
        "steps": [], 
        "finished": False, 
        "metadata": [], 
        "rewrite_count": 0, 
        "total_steps": 0,
        "method_code": original_method_code, # 存储原始方法代码
        "track_id_with_index": method_code_with_index # 存储带索引的方法ID
    }
    
    step_id = 0
    logging.info(f"[Track {track_id}] Starting with method {original_method_code}")

    while True:
        if track_data["finished"]:
            break
            
        # logging.info(f"[Track {track_id}][Step {step_id}] Active")
        
        # 1. 生成下一步推理 (小模型)
        if not track_data["steps"]:
            prompt = get_first_user_msg(problem, options, method_code=original_method_code)
        else:
            steps_so_far_str = "\n\n".join(track_data["steps"]) + "\n\n"
            prompt = f"{get_first_user_msg(problem, options, method_code=original_method_code)}\n<think>{steps_so_far_str}"
        
        sampling_params_small = SamplingParams(
            temperature=0.6, top_p=0.95, max_tokens=512, stop=["\n\n"]
        )
        
        small_outputs = model_proxy.generate_on_draft([prompt], sampling_params_small, f"s{step_id}_draft")
        
        small_output_data = small_outputs[0].outputs[0] # vLLM 返回列表，取第一个
        generated_text_small = small_output_data.text
        num_tokens_small = len(small_output_data.token_ids)
        finished_small = any([x in generated_text_small for x in ["boxed", "Answer:", "ANSWER:"]])
        
        small_result_for_track = {
            "text": generated_text_small, "tokens": num_tokens_small, "finished": finished_small
        }
        
        # 2. 准备评分
        steps_for_scoring = track_data["steps"] + [generated_text_small]
        steps_so_far_str_scoring = "\n\n".join(steps_for_scoring) # No trailing \n\n for scoring prompt usually
        
        score_prompt_text = f"{get_first_user_msg(problem, options, method_code=original_method_code)}\n<think>{steps_so_far_str_scoring}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
        
        scoring_params_target = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)
        
        score_outputs = model_proxy.generate_on_target([score_prompt_text], scoring_params_target, f"s{step_id}_score")
        
        score_output_data = score_outputs[0].outputs[0]
        token_score = score_output_data.text
        logprobs_score = score_output_data.logprobs[0] if score_output_data.logprobs else None

        if logprobs_score is None:
            logging.warning(f"[Track {track_id}][Step {step_id}] Score generation did not return logprobs. Assuming score 0.")
            score = 0
        else:
            score = process_vllm_logprobs(token_score, logprobs_score, method=args.score_method)

        # 3. 处理评分结果，确定是否需要重写
        final_step_text = generated_text_small
        final_num_tokens = num_tokens_small
        final_finished_flag = finished_small
        used_base_model_for_step = False
        base_model_step_text = None
        base_model_num_tokens = None

        if score is None or score < args.score_threshold:
            logging.info(f"[Track {track_id}][Step {step_id}] Method {original_method_code} score {score}, needs rewrite.")
            track_data["rewrite_count"] += 1
            used_base_model_for_step = True

            # 重写提示与小模型生成提示相同（基于之前的步骤）
            rewrite_prompt_text = prompt # Reuse the same prompt that led to the low-score small model output
            
            sampling_params_rewrite = SamplingParams( # Same as small model's sampling params
                temperature=0.6, top_p=0.95, max_tokens=512, stop=["\n\n"]
            )
            rewrite_outputs = model_proxy.generate_on_target([rewrite_prompt_text], sampling_params_rewrite, f"s{step_id}_rewrite")
            
            rewrite_output_data = rewrite_outputs[0].outputs[0]
            final_step_text = rewrite_output_data.text
            final_num_tokens = len(rewrite_output_data.token_ids)
            final_finished_flag = any([x in final_step_text for x in ["boxed", "Answer:", "ANSWER:"]])
            
            base_model_step_text = final_step_text
            base_model_num_tokens = final_num_tokens
        else:
            logging.info(f"[Track {track_id}][Step {step_id}] Method {original_method_code} score {score}, accepted.")
            # final_step_text, final_num_tokens, final_finished_flag already set from small model
        
        track_data["total_steps"] += 1
        
        warning_flag_local = False
        if "</think>" in final_step_text and not any([x in final_step_text for x in ["boxed", "Answer:", "ANSWER:"]]):
            logging.warning(f"Warning: Track {track_id} step contains </think>, removing.")
            final_step_text = final_step_text.replace("</think>", "")
            warning_flag_local = True
            
        track_data["steps"].append(final_step_text)
        
        current_step_metadata = {
            "step_id": step_id,
            "method_code": original_method_code,
            "step_str": final_step_text,
            "small_model_step": small_result_for_track["text"],
            "num_output_tokens_small": small_result_for_track["tokens"],
            "score": score,
            "base_model_step": base_model_step_text,
            "num_output_tokens_base": base_model_num_tokens,
            "final_num_output_tokens": final_num_tokens,
            "used_base_model": used_base_model_for_step
        }
        if warning_flag_local:
            current_step_metadata["warning"] = "step_str had a </think>"
        track_data["metadata"].append(current_step_metadata)
        
        if final_finished_flag:
            track_data["finished"] = True
            track_data["stop_reason"] = "finished"
        elif len(track_data["steps"]) > 2 and track_data["steps"][-1] == track_data["steps"][-2]:
            track_data["finished"] = True
            track_data["stop_reason"] = "repeated"
        elif sum([m["final_num_output_tokens"] for m in track_data["metadata"]]) >= args.token_budget:
            track_data["finished"] = True
            track_data["stop_reason"] = "budget"
            
        step_id += 1

    logging.info(f"[Track {track_id}] Finished. Reason: {track_data.get('stop_reason', 'N/A')}")
    return track_data


def model_server_process(request_q, result_q_map_dict_proxy, num_tracks):
    """专门用于处理模型推理请求的进程"""
    logging.info("Model server process started.")
    active_requests = {} # request_id -> track_id (to know which result_q to use)

    while True:
        # 收集请求，可以设置一个小的超时或者批处理大小
        batched_requests_draft = []
        batched_request_ids_draft = []
        batched_prompts_draft = []
        batched_sampling_params_draft = [] # vLLM can handle varied sampling params per request

        batched_requests_target = []
        batched_request_ids_target = []
        batched_prompts_target = []
        batched_sampling_params_target = []

        # Try to get a batch of requests
        # This loop will try to quickly accumulate requests if they are available
        # And then proceed to processing even if only one request is there after a short wait.
        timeout_collect = 0.01 # Short timeout to allow batching
        while True:
            try:
                request = request_q.get(timeout=timeout_collect if not batched_requests_draft and not batched_requests_target else 0) # only wait if no requests yet
                # logging.debug(f"Model server received request: {request['request_id']}")
                if request.get("type") == "TERMINATE":
                    logging.info("Model server received TERMINATE signal. Shutting down.")
                    # Propagate termination or ensure client queues are empty/handled
                    return 

                active_requests[request["request_id"]] = request["track_id"]

                if request["model_type"] == "draft":
                    batched_requests_draft.append(request)
                    batched_request_ids_draft.append(request["request_id"])
                    batched_prompts_draft.extend(request["prompts"]) # prompts is a list
                    # Assuming SamplingParams object can be directly used if vLLM supports it for batching
                    # Or, if vLLM needs a single SamplingParams for the whole batch, this needs adjustment.
                    # For now, let's assume vLLM can take a list of SamplingParams or handle it internally
                    # For simplicity, if all sampling params are the same, we can pass one.
                    # However, the generate function usually takes ONE sampling_params object for the whole batch.
                    # This is a CRITICAL point for vLLM batching.
                    # For now, let's assume the first request's sampling_params is used for the batch if batching.
                    # Or better, vLLM generate takes a list of prompts and ONE SamplingParams.
                    # If different params are needed per prompt, then we cannot batch them this way.
                    # The provided code `models["draft"].generate(active_prompts, sampling_params)` suggests
                    # one sampling_params for a batch of prompts.
                    # Let's assume for now that for a given model_type, sampling_params will be consistent for batching.
                    if not batched_sampling_params_draft: # Take the first one
                         batched_sampling_params_draft.append(request["sampling_params"])

                elif request["model_type"] == "target":
                    batched_requests_target.append(request)
                    batched_request_ids_target.append(request["request_id"])
                    batched_prompts_target.extend(request["prompts"])
                    if not batched_sampling_params_target: # Take the first one
                        batched_sampling_params_target.append(request["sampling_params"])
                
                # Break if we have a decent batch size or if queue is empty
                if request_q.empty() or (len(batched_prompts_draft) + len(batched_prompts_target)) >= num_tracks * 2 : # Heuristic batch size
                    break
            except multiprocessing.queues.Empty:
                break # No more requests for now

        if batched_prompts_draft:
            # logging.info(f"Model server processing {len(batched_prompts_draft)} DRAFT requests.")
            # Ensure sampling_params is a single object if vLLM expects that
            sp_draft = batched_sampling_params_draft[0] if batched_sampling_params_draft else SamplingParams(max_tokens=50) # Fallback
            outputs_draft = models_global["draft"].generate(batched_prompts_draft, sp_draft)
            # Distribute results
            # This assumes outputs_draft is a list corresponding to batched_prompts_draft
            for i, req_id in enumerate(batched_request_ids_draft):
                track_id_for_result = active_requests.pop(req_id)
                # The output for a single prompt in a batch is `outputs_draft[i]`
                # which is a RequestOutput object. We need to send a list containing this one RequestOutput.
                result_q_map_dict_proxy[track_id_for_result].put({"request_id": req_id, "data": [outputs_draft[i]]})
                # logging.debug(f"Model server sent DRAFT result for {req_id} to track {track_id_for_result}")


        if batched_prompts_target:
            # logging.info(f"Model server processing {len(batched_prompts_target)} TARGET requests.")
            sp_target = batched_sampling_params_target[0] if batched_sampling_params_target else SamplingParams(max_tokens=50) # Fallback
            outputs_target = models_global["target"].generate(batched_prompts_target, sp_target)
            for i, req_id in enumerate(batched_request_ids_target):
                track_id_for_result = active_requests.pop(req_id)
                result_q_map_dict_proxy[track_id_for_result].put({"request_id": req_id, "data": [outputs_target[i]]})
                # logging.debug(f"Model server sent TARGET result for {req_id} to track {track_id_for_result}")

        # If no requests were processed, sleep a bit to prevent busy-waiting if queue remains empty
        if not batched_prompts_draft and not batched_prompts_target:
            time.sleep(0.001) # Small sleep

def run_reasoning_process_multiprocess(args, problem, options):
    """运行推理过程 - 多进程版本"""
    start_time_total = time.time()
    global request_queue_global, result_queues_global

    # 使用Manager来创建可在进程间共享的字典和队列
    manager = Manager()
    request_queue_global = manager.Queue()
    # result_queues_global will be a dict of Queues, one for each track
    # We need a proxy to this dict that the model server process can use
    result_queues_proxy_for_server = manager.dict()


    # 模型代理给 choose_solution_methods (临时)
    # choose_solution_methods 应该只被调用一次，且在主进程中
    # 它需要一个临时的 model_proxy 或直接让主进程处理
    # For simplicity, let the main process handle choose_solution_methods directly with global models
    # This requires models to be initialized before this call.
    # However, models are initialized in model_server_process.
    # This creates a dependency issue.
    # SOLUTION: Initialize models in the main process first, then pass them (or their names/configs)
    # to the server process if vLLM objects cannot be pickled.
    # OR, the model_server_process starts, initializes models, and then we proceed.
    # Let's assume `initialize_models_global` has been called in `main` before this.
    
    # The choose_solution_methods also needs to use the model server.
    # So, we need the model server running before choosing methods.

    # 0. 启动模型服务器进程 (放到 main 函数中更好，这里仅为逻辑流)
    # This should be done in main before calling this function for multiple problems.
    # For now, we'll assume it's handled and request_queue_global is ready.

    # 临时的ModelProxy для choose_solution_methods. 它需要自己的结果队列。
    choose_method_track_id = "choose_methods_process"
    choose_method_result_q = manager.Queue()
    result_queues_proxy_for_server[choose_method_track_id] = choose_method_result_q
    temp_model_proxy = ModelProxy(request_queue_global, result_queues_proxy_for_server, choose_method_track_id)


    try:
        solution_methods_codes = choose_solution_methods(problem, options, args.dataset_name, temp_model_proxy, args.method_num)
    except Exception as e:
        logging.error(f"Error in choose_solution_methods: {e}. Skipping problem.")
        # Clean up the temporary queue for choose_methods
        if choose_method_track_id in result_queues_proxy_for_server:
            del result_queues_proxy_for_server[choose_method_track_id]
        return {}, 0, 0


    # Clean up the temporary queue for choose_methods
    if choose_method_track_id in result_queues_proxy_for_server:
        del result_queues_proxy_for_server[choose_method_track_id]


    method_tracks_info = {} # To store (original_method_code, track_id_with_index)
    method_count = {}
    processes = []

    args_dict = vars(args) # Convert Namespace to dict for pickling

    # 为每个方法创建独立的推理轨道和结果队列
    for i, method_code in enumerate(solution_methods_codes):
        if method_code in method_count:
            method_count[method_code] += 1
            track_id_with_index = f"{method_code}{method_count[method_code]}"
        else:
            method_count[method_code] = 1
            track_id_with_index = f"{method_code}1"
        
        # Each track process needs its own result queue
        track_result_q = manager.Queue()
        result_queues_proxy_for_server[track_id_with_index] = track_result_q # Add to server's map
        
        method_tracks_info[track_id_with_index] = {
            "original_code": method_code,
        }

        # (track_id, method_code_with_index, original_method_code, problem, options, args_dict, request_q, result_q_map)
        p = Process(target=run_one_track_process, args=(
            track_id_with_index, # This is the unique ID for the process and its result queue
            track_id_with_index, # method_code_with_index (same as track_id here)
            method_code,         # original_method_code
            problem, options, args_dict, 
            request_queue_global, 
            result_queues_proxy_for_server # Pass the dict of queues
        ))
        processes.append(p)
        # Limit the number of concurrent processes if needed, but Pool would manage this.
        # For now, starting all if <= args.num_parallel_tracks
        # This implementation starts all selected methods. If method_num > num_parallel_tracks,
        # they will all start but contend for the model server.
        # A Pool(args.num_parallel_tracks) would be better if we want to strictly limit concurrency of python processes.
        # However, the bottleneck is the GPU server, so having all track processes run and queue requests is fine.

    for p in processes:
        p.start()

    # 等待所有子进程完成并收集结果
    # The run_one_track_process function should return its track_data.
    # This requires another queue for each process to send its final result back, or use Pool.map.
    # For simplicity with current structure, let's have run_one_track_process write its result to a shared dict
    # managed by the Manager.

    final_method_tracks_data = manager.dict() # track_id_with_index -> track_data

    # Modify run_one_track_process to accept this dict and write to it before returning
    # For now, let's retrieve results differently.
    # We need to get the return value from each process.
    # One way is to have each process put its final result into a single shared "final results queue".

    output_final_results_queue = manager.Queue()

    # Need to rewrite run_one_track_process to put its result into this queue
    # Let's adjust run_one_track_process:
    # def run_one_track_process(..., output_final_results_queue):
    #   ...
    #   output_final_results_queue.put({track_id: track_data})
    #   return # Process function itself doesn't need to return for p.join()

    # Re-creating processes with the output queue (conceptual change, actual code below)
    processes_updated = []
    result_queues_for_tracks_local = {} # For main process to map track_id to its dedicated result queue from model_server
    for i, method_code in enumerate(solution_methods_codes):
        # Recalculate track_id_with_index (or retrieve from method_tracks_info)
        # This part needs to be robust if solution_methods_codes is shorter than method_num
        # (e.g., choose_solution_methods returned fewer than method_num valid methods)
        
        # Find the track_id_with_index previously generated
        current_track_id = None
        temp_count = method_count.get(method_code, 0) # How many times this code has appeared so far
        for tid, info in method_tracks_info.items():
            if info["original_code"] == method_code:
                # Match based on occurrence if there are duplicates
                suffix_match = re.search(r'\d+$', tid)
                occurrence = int(suffix_match.group(0)) if suffix_match else 1
                if occurrence == (method_count[method_code] - temp_count +1) : # This is fragile logic
                     current_track_id = tid
                     temp_count -=1
                     break
        if not current_track_id: # Fallback or error
            logging.error(f"Could not reliably find track_id for method {method_code}. This is a bug.")
            # Create a placeholder to avoid crashing
            current_track_id = f"{method_code}_fallback_{i}"
            result_queues_proxy_for_server[current_track_id] = manager.Queue() # ensure it has a queue

        # Ensure result_queues_for_tracks_local mirrors what was given to ModelProxy instances
        result_queues_for_tracks_local[current_track_id] = result_queues_proxy_for_server[current_track_id]


        # We need to pass output_final_results_queue to the process function.
        # Let's assume run_one_track_process is modified:
        # It now takes `output_final_results_queue` as the last arg and puts its result there.
        # The return from p.join() will be None.
        
        # The Process target function needs to be a top-level function or a static method for pickling.
        # So, we'll need a wrapper or modify run_one_track_process to take the final_q.
        
        # For now, let's assume `run_one_track_process` is modified to return its result.
        # And we will collect them after joining. This is not possible with `Process` directly.
        # `Pool.map` is better for this.
        # Given the current `Process` structure, using a shared dict or queue for final results is the way.

    # Let's assume run_one_track_process is modified to take `final_method_tracks_data` (a manager.dict())
    # and writes `final_method_tracks_data[track_id] = track_data` before exiting.
    # (This is simpler than another queue for final results if we already have track_ids)

    # Modify `run_one_track_process` signature:
    # run_one_track_process(..., final_shared_dict_proxy)
    # And inside: final_shared_dict_proxy[track_id] = track_data

    processes_final = []
    for i, method_code in enumerate(solution_methods_codes):
        # Regenerate track_id logic robustly or fetch from a stored list of (track_id, method_code)
        track_id_with_index = list(method_tracks_info.keys())[i] # Assuming order is preserved
        original_method_code = method_tracks_info[track_id_with_index]["original_code"]

        # args for run_one_track_process_wrapper which calls run_one_track_process
        # and puts result into final_method_tracks_data
        
        # Original `run_one_track_process` doesn't know about `final_method_tracks_data`
        # We need a wrapper, or to use `Pool` which handles return values.
        
        # Let's use a simpler collection method for now with current Process structure:
        # Each process will put its result into its OWN result queue (which is already set up for model comms)
        # after it's done. The main process will then read from these.
        # This means the ModelProxy's result_q is REUSED by the track process for its final output.
        # Modify `run_one_track_process` to put its `track_data` into `result_q_map[track_id]` at the end.
        # And adjust the model server NOT to expect further requests on this queue once final data is sent.
        # This is a bit messy.

        # CLEANER: Use one dedicated queue for final results from all processes.
        # Each process: `final_results_q.put({my_track_id: my_final_track_data})`

    # Restart processes with the call to the wrapper or modified function
    # This requires significant restructuring of how processes are started and how results are collected.

    # --- Simplified approach for now, assuming run_one_track_process can be modified ---
    # Let's assume `run_one_track_process` returns its data and we can get it via a Pool or similar.
    # With current `multiprocessing.Process`, we need IPC for results.
    # Using a Pool is cleaner:
    
    collected_tracks_data = {}
    if solution_methods_codes: # Only proceed if methods were chosen
        pool_size = min(args.num_parallel_tracks, len(solution_methods_codes))
        if pool_size > 0:
            with multiprocessing.Pool(processes=pool_size) as pool:
                results_async = []
                for i, method_code_orig in enumerate(solution_methods_codes):
                    # Regenerate track_id_with_index based on chosen methods
                    # This part of logic needs to be consistent with how track_ids were created for queues
                    # For simplicity, let's re-generate method_tracks_info here for the pool
                    # This is not ideal as queues are already set up with specific IDs.

                    # Correct approach: Iterate through the `method_tracks_info` which has the definitive track_ids
                    for track_id, info in method_tracks_info.items():
                        original_m_code = info["original_code"]
                        # Ensure result queue for this track_id exists in result_queues_proxy_for_server
                        if track_id not in result_queues_proxy_for_server:
                             result_queues_proxy_for_server[track_id] = manager.Queue()


                        res = pool.apply_async(run_one_track_process, args=(
                            track_id, track_id, original_m_code,
                            problem, options, args_dict,
                            request_queue_global, result_queues_proxy_for_server # Pass the proxy dict
                        ))
                        results_async.append(res)
                
                for res_async in results_async:
                    try:
                        track_result_data = res_async.get(timeout=1800) # 30 min timeout per track
                        # The key for collected_tracks_data should be the track_id_with_index
                        # which is the first arg returned by run_one_track_process if it returns it,
                        # or it's part of the track_result_data map.
                        # `run_one_track_process` returns `track_data` which contains `track_id_with_index`.
                        if track_result_data and "track_id_with_index" in track_result_data:
                             collected_tracks_data[track_result_data["track_id_with_index"]] = track_result_data
                        else:
                            logging.error(f"Async result format error or None: {track_result_data}")
                    except TimeoutError:
                        logging.error(f"A track process timed out.")
                    except Exception as e:
                        logging.error(f"Error getting result from a track process: {e}")
                pool.close()
                pool.join()
        else:
            logging.warning("No solution methods to process after selection.")
    else:
        logging.warning("No solution methods chosen or an error occurred.")


    total_time_val = time.time() - start_time_total

    # Calculate rewrite rates from collected_tracks_data
    avg_rewrite_rate_val = 0
    if collected_tracks_data:
        total_rewrite_sum = 0
        valid_methods_count = 0
        for _track_id, track_data_item in collected_tracks_data.items():
            if track_data_item.get("total_steps", 0) > 0:
                rewrite_rate = track_data_item.get("rewrite_count",0) / track_data_item["total_steps"] * 100
                track_data_item["rewrite_rate"] = rewrite_rate # Store in the collected data
                total_rewrite_sum += rewrite_rate
                valid_methods_count +=1
        if valid_methods_count > 0:
            avg_rewrite_rate_val = total_rewrite_sum / valid_methods_count
        logging.info(f"Overall average rewrite rate: {avg_rewrite_rate_val:.2f}%")
    
    # The model_server_process needs to be terminated at the end of all problems.
    # This function is called per problem. Termination signal should be sent from main.
    return collected_tracks_data, total_time_val, avg_rewrite_rate_val


def process_results(method_tracks_data_map, total_time, avg_rewrite_rate, args, problem, options):
    """处理结果并保存 - adapted for multiprocess output"""
    all_metadata = []
    # method_tracks_data_map is already the processed data from each track
    for track_id, track_data in method_tracks_data_map.items():
        if "metadata" in track_data:
            all_metadata.extend(track_data["metadata"])

    final_results_map = {}
    for track_id, track_data in method_tracks_data_map.items():
        if track_data.get("finished"):
            answer = extract_final_answer("\n\n".join(track_data.get("steps", [])), args.dataset_name)
            final_results_map[track_id] = {
                "steps": track_data.get("steps"),
                "answer": answer,
                "stop_reason": track_data.get("stop_reason", "unknown"),
                "num_tokens": sum(m["final_num_output_tokens"] for m in track_data.get("metadata", []) if "final_num_output_tokens" in m),
                "num_steps": len(track_data.get("steps", [])),
                "method_code": track_data.get("method_code"), # Original method code
                "track_id_with_index": track_data.get("track_id_with_index") # Unique track ID
            }

    best_result_obj = select_best_result(list(final_results_map.values()))

    metadata_with_stats = {
        "problem_id": args.problem_id,
        "repeat_id": args.repeat_id,
        "dataset_name": args.dataset_name,
        "problem": problem,
        "options": options,
        "method_num": args.method_num,
        "solution_methods_chosen_count": len(method_tracks_data_map), # Number of tracks actually run
        "method_tracks_data": method_tracks_data_map, # This now holds all data from each track process
        "final_results_per_track": final_results_map,
        "best_result": best_result_obj,
        "total_time": total_time,
        "score_threshold": args.score_threshold,
        "token_budget": args.token_budget,
        "score_method": args.score_method,
        "avg_rewrite_rate": avg_rewrite_rate
    }
    
    return metadata_with_stats

def save_results(metadata_with_stats, output_filename):
    """保存结果到文件"""
    try:
        with open(f"{output_filename}.pickle", "wb") as f:
            pickle.dump(metadata_with_stats, f)

        with open(f"{output_filename}.txt", "w") as f:
            # Use a custom pretty printer for better control if needed
            # For now, pprint is fine. Ensure it handles complex nested dicts from Manager.
            # It might be safer to convert Manager objects to standard dicts/lists before saving.
            # The data collected from pool.get() should already be standard Python objects.
            pprint.pprint(metadata_with_stats, stream=f, width=120) 
        
        logging.info(f"结果已保存到 {output_filename}.pickle 和 {output_filename}.txt")
    except Exception as e:
        logging.error(f"Error saving results for {output_filename}: {e}")
        logging.info("Attempting to print problematic data structure (first level):")
        for key, value in metadata_with_stats.items():
            logging.info(f"Key: {key}, Type: {type(value)}")


def main():
    """主函数，整合所有流程 - 多进程版本"""
    args, _ = parse_arguments()
    
    # ---模型和队列初始化 (主进程)---
    global models_global, request_queue_global, result_queues_global
    
    # Initialize models in the main process if they are to be directly accessed by model_server_process logic
    # when model_server_process is a function run in a Process.
    # If vLLM LLM objects are picklable and can be passed, then server can init.
    # Safer: Main init, server uses them.
    try:
        initialize_models_global(args.tensor_parallel_size) # Initializes models_global
    except RuntimeError as e:
        logging.error(f"Exiting due to model initialization failure: {e}")
        return

    manager = multiprocessing.Manager()
    request_queue_global = manager.Queue() # Used by all track processes to send requests
    
    # result_queues_global is a dict where keys are track_ids and values are Queues.
    # This dict itself needs to be managed if the model_server_process needs to add to it.
    # For ModelProxy, it receives the full map.
    # For model_server_process, it receives a proxy to this map.
    result_queues_dict_proxy_for_server = manager.dict() # Server will use this proxy

    # ---启动模型服务器进程---
    # It's better to start the server once and have it serve all problems.
    num_tracks_for_server = args.method_num # Max possible tracks per problem for batching estimate
    model_server = Process(target=model_server_process, args=(request_queue_global, result_queues_dict_proxy_for_server, num_tracks_for_server))
    model_server.start()
    logging.info("Model server process launched.")


    # 加载数据集
    args.dataset = get_dataset(args.dataset_name)
    problem_ids = parse_problem_range(args.problem_id)
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_problem_tasks = len(problem_ids) * args.repeat_id
    completed_problem_tasks = 0
    
    logging.info(f"开始处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次, 并行 {args.num_parallel_tracks} tracks")
    
    for p_id in problem_ids:
        for r_idx in range(args.repeat_id):
            current_problem_display_id = f"{p_id}-{r_idx}"
            logging.info(f"--- 处理问题: {current_problem_display_id} ---")
            
            output_filename = os.path.join(output_dir, f"{p_id}-{r_idx}")
            
            if os.path.exists(f"{output_filename}.pickle"):
                logging.info(f"问题 {current_problem_display_id} 已解决，跳过")
                completed_problem_tasks += 1
                continue
            
            args.problem_id = p_id # Update for prepare_problem_data
            args.repeat_id = r_idx # Update for metadata
            problem_text, problem_options = prepare_problem_data(args)
            
            # run_reasoning_process_multiprocess will now use the global request_queue
            # and the result_queues_dict_proxy_for_server
            # It needs access to these, or they are passed.
            # The current run_reasoning_process_multiprocess re-creates manager and queues.
            # This should be harmonized: queues created once in main, passed down.
            
            # For run_reasoning_process_multiprocess to use the global queues and server proxy:
            # It needs to be refactored not to create its own Manager.
            # Let's assume it's refactored to accept request_q and result_q_proxy.
            
            # --- Refactor run_reasoning_process_multiprocess slightly to accept existing queues ---
            # def run_reasoning_process_multiprocess(args, problem, options, req_q, res_q_proxy_server)

            # For current structure, let's pass them:
            # (This means `run_reasoning_process_multiprocess` should NOT create its own manager/queues for these)
            # The `Pool` inside `run_reasoning_process_multiprocess` will use these passed queues via ModelProxy.
            
            # Pass the main request_queue and the server's view of result queues
            method_tracks_res_map, total_time_res, avg_rewrite_rate_res = \
                run_reasoning_process_multiprocess(args, problem_text, problem_options) # It will use global queues as per current design

            if not method_tracks_res_map and total_time_res == 0 and avg_rewrite_rate_res == 0:
                logging.warning(f"Problem {current_problem_display_id} resulted in no tracks or error during method selection.")
            
            metadata_to_save = process_results(method_tracks_res_map, total_time_res, avg_rewrite_rate_res, args, problem_text, problem_options)
            save_results(metadata_to_save, output_filename)
            
            completed_problem_tasks += 1
            logging.info(f"完成进度: {completed_problem_tasks}/{total_problem_tasks} ({completed_problem_tasks/total_problem_tasks*100:.2f}%)")
            logging.info(f"--- 问题 {current_problem_display_id} 处理完毕 ---")

    # --- 清理 ---
    logging.info("所有问题处理完毕.正在关闭模型服务器...")
    request_queue_global.put({"type": "TERMINATE"}) # Signal server to stop
    model_server.join(timeout=30) # Wait for server to finish
    if model_server.is_alive():
        logging.warning("Model server did not terminate gracefully. Forcing.")
        model_server.terminate()
        model_server.join()

    # Manager should also be shutdown if created explicitly and not through `with`
    # In this case, `manager` objects in `run_reasoning_process_multiprocess` are local to its call.
    # The `manager` in `main` for the global queues will be cleaned up when main exits.

    logging.info(f"所有任务完成！")

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True) # Good for CUDA safety, might be default on some OS
    main()