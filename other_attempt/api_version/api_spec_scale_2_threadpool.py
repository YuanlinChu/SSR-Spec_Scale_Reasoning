# python api_spec_scale_2_threadpool.py --dataset_name aime --problem_id 60 --repeat_id 3 --output_dir results/api_spec_scale_2 --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 3
# api版本 + 多线程
import os
import time
import pickle
import pprint
import logging
import argparse
import numpy as np
from collections import Counter
from datasets import load_dataset, load_from_disk
import re
import concurrent.futures
from openai import OpenAI
# 导入choose-prompts模块中的选择prompt
from prompts.choose_prompts import math_choose_prompt, gpqa_choose_prompt
from prompts.method_prompt import A_prompt, B_prompt, C_prompt, D_prompt, E_prompt, F_prompt, G_prompt, H_prompt, I_prompt, J_prompt, K_prompt, L_prompt, M_prompt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 模型服务地址
MODEL_TARGET_URL = "http://localhost:30000/v1"  # Qwen/QwQ-32B
MODEL_DRAFT_URL = "http://localhost:30001/v1"   # DeepSeek-1.5B
MODEL_TARGET_NAME = "Qwen/QwQ-32B"
MODEL_DRAFT_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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

def query_model(model_name, model_url, messages, temperature=0.7, top_p=0.95, max_tokens=100, stop=None, top_logprobs=None, extra_body=None):
    """使用OpenAI API查询模型"""
    client = OpenAI(
        api_key="EMPTY",
        base_url=model_url,
    )

    logprobs = None
    if top_logprobs is not None:
        logprobs = True

    response = client.chat.completions.create(
        model=model_name,  # 这个参数在vLLM服务中不重要，但需要提供
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        extra_body=extra_body,
    )
    
    return response

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
    parser.add_argument("--method_num", type=int, default=1,
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

def choose_solution_methods(problem, options, dataset_name, method_num=3):
    """选择最合适的解题方法"""
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
    prompt = f"{prompt}\n\nExamine the problem and select the **{method_num}** strategies (by their codes, e.g. {'B,E,F' if method_num == 3 else 'B,E,F,A' if method_num == 4 else 'B,E'}) that you believe are most promising for solving it. You only need to output {method_num} codes, without any other symbols or text."
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>I think the best {method_num} methods are: "},
    ]
    extra_body={
        "add_generation_prompt": False, 
        "continue_final_message": True,
        # "return_tokens_as_token_ids": True,
    }  
    # 使用目标模型生成方法选择
    response = query_model(
        MODEL_TARGET_NAME,
        MODEL_TARGET_URL,
        messages,
        temperature=0,
        max_tokens=method_num * 2-1,  # 每个方法可能需要2个token（字母+逗号）
        extra_body=extra_body,
    )
    
    method_text = response.choices[0].message.content.strip()
    # print(messages)
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

class ReasoningTrack:
    """推理轨道类，管理单个推理路径的完整生命周期"""
    
    def __init__(self, track_id, method_code, problem, options, dataset_name, score_threshold=7.0, score_method="greedy", token_budget=8192):
        self.track_id = track_id
        self.method_code = method_code  # 原始方法代码，不含索引
        self.problem = problem
        self.options = options
        self.dataset_name = dataset_name
        self.score_threshold = score_threshold
        self.score_method = score_method
        self.token_budget = token_budget  # 添加token预算
        
        # 推理状态
        self.steps = []
        self.metadata = []
        self.finished = False
        self.stop_reason = None  # 添加停止原因
        self.rewrite_count = 0
        self.total_steps = 0
        
        logging.info(f"[Track {track_id}] 初始化推理轨道，方法: {method_code}")
    
    def process_step(self):
        """处理一个完整的推理步骤：小模型推理->大模型打分->大模型改写（如需要）"""
        if self.finished:
            return False
        
        # 检查是否已经超过token预算
        total_tokens = sum(metadata.get("tokens", 0) for metadata in self.metadata)
        if total_tokens >= self.token_budget - 200: # 预留200token，防止最后一步的token被截断
            logging.info(f"[Track {self.track_id}] 达到token预算 ({total_tokens}/{self.token_budget})，停止推理")
            self.finished = True
            self.stop_reason = "budget"
            return False
        
        # 1. 小模型生成下一步推理
        step_text, is_finished, num_tokens = self._generate_step()
        
        # 2. 大模型评分
        score = self._score_step(step_text)
        
        # 记录元数据
        step_metadata = {
            "score": score,
            "tokens": num_tokens,
            "rewritten": False
        }
        
        # 3. 如果分数低于阈值，大模型改写
        if score is None or score < self.score_threshold:
            logging.info(f"[Track {self.track_id}] 分数 {score}，需要改写")
            rewritten_text, is_finished, num_tokens = self._rewrite_step(step_text)
            self.rewrite_count += 1
            step_metadata["rewritten"] = True
            step_metadata["original_text"] = step_text
            step_metadata["rewritten_tokens"] = num_tokens
            step_text = rewritten_text
        
        # 处理特殊情况
        if "</think>" in step_text and not any([x in step_text for x in ["boxed", "Answer:", "ANSWER:"]]):
            logging.warning(f"[Track {self.track_id}] 步骤包含 </think>，正在移除")
            step_text = step_text.replace("</think>", "")

        # 4. 更新状态
        self.steps.append(step_text)
        self.metadata.append(step_metadata)
        self.total_steps += 1
        
        # 5. 检查是否完成
        if is_finished:
            self.finished = True
            self.stop_reason = "finished"
            logging.info(f"[Track {self.track_id}] 推理完成")
            
            # 提取最终答案
            full_reasoning = "\n\n".join(self.steps)
            answer = extract_final_answer(full_reasoning, self.dataset_name)
            if answer is not None:
                logging.info(f"[Track {self.track_id}] 提取到答案: {answer}")
            else:
                logging.warning(f"[Track {self.track_id}] 无法提取答案")
            
            # 更新最后一个元数据
            self.metadata[-1]["answer"] = answer
        # 检查是否出现重复步骤
        elif len(self.steps) > 2 and self.steps[-1] == self.steps[-2]:
            self.finished = True
            self.stop_reason = "repeated"
            logging.info(f"[Track {self.track_id}] 检测到重复步骤，停止推理")
        
        return True
    
    def _generate_step(self):
        """使用小模型生成下一步推理"""
        if not self.steps:  # 第一步
            prompt = get_first_user_msg(self.problem, self.options, method_code=self.method_code)
            messages = [
                {"role": "user", "content": prompt},
            ]
            extra_body = {"add_generation_prompt": True}
        else:  # 继续之前的消息
            steps_so_far_str = "\n\n".join(self.steps) + "\n\n"
            prompt = f"{get_first_user_msg(self.problem, self.options, method_code=self.method_code)}"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
            ]
            extra_body = {"add_generation_prompt": False, "continue_final_message": True}

        response = query_model(
            MODEL_DRAFT_NAME,
            MODEL_DRAFT_URL,
            messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=512,
            stop=["\n\n"],
            extra_body=extra_body,
        )
        
        generated_text = response.choices[0].message.content
        # print(generated_text)

        num_tokens = len(generated_text.split())  # 简单估计token数
        
        # 检查是否完成
        finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
        
        return generated_text, finished, num_tokens
    
    def _score_step(self, step_text):
        """使用大模型评分"""

        steps_for_scoring = self.steps + [step_text]
        steps_so_far_str = "\n\n".join(steps_for_scoring) + "\n\n"
        
        messages = [
            {"role": "user", "content": get_first_user_msg(self.problem, self.options, method_code=self.method_code)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},  # a </think> cannot be added at the end, otherwise, none of the previous steps will be encoded
            {"role": "user", "content": "Evaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9."},
            {"role": "assistant", "content": "<think>I think the quality score is: "},
        ]
        extra_body={
            "add_generation_prompt": False, 
            "continue_final_message": True,
            # "return_tokens_as_token_ids": True,
        }  
        response = query_model(
            MODEL_TARGET_NAME,
            MODEL_TARGET_URL,
            messages,
            temperature=0.0,
            max_tokens=1,
            top_logprobs=10,
            extra_body=extra_body,
        )
        
        # token = response.choices[0].message.content
        # print(token)
        
        score = process_logprobs(response, method=self.score_method)
        return score  # 默认分数
    
    def _rewrite_step(self, step_text):
        """使用大模型改写步骤"""
        
        if not self.steps:  # 第一步
            prompt = get_first_user_msg(self.problem, self.options, method_code=self.method_code)
            messages = [
                {"role": "user", "content": prompt},
            ]
            extra_body = {"add_generation_prompt": True}
        else:  # 继续之前的消息
            steps_so_far_str = "\n\n".join(self.steps) + "\n\n"
            prompt = f"{get_first_user_msg(self.problem, self.options, method_code=self.method_code)}"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
            ]
            extra_body = {"add_generation_prompt": False, "continue_final_message": True}
        
        response = query_model(
            MODEL_TARGET_NAME,
            MODEL_TARGET_URL,
            messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=512,
            stop=["\n\n"],
            extra_body=extra_body,
        )
        
        rewritten_text = response.choices[0].message.content
        num_tokens = len(rewritten_text.split())  # 简单估计token数
        
        # 检查是否完成
        finished = any([x in rewritten_text for x in ["boxed", "Answer:", "ANSWER:"]])
        
        return rewritten_text, finished, num_tokens
    
    def get_result(self):
        """获取推理结果"""
        full_reasoning = "\n\n".join(self.steps)
        answer = extract_final_answer(full_reasoning, self.dataset_name) if self.finished else None
        
        return {
            "track_id": self.track_id,
            "method_code": self.method_code,
            "steps": self.steps,
            "metadata": self.metadata,
            "finished": self.finished,
            "rewrite_count": self.rewrite_count,
            "total_steps": self.total_steps,
            "reasoning": full_reasoning,
            "answer": answer,
            "stop_reason": "finished" if self.finished else "incomplete",
            "num_tokens": sum(m.get("tokens", 0) for m in self.metadata)
        }

def run_reasoning_process(args, problem, options):
    """运行推理过程，使用并行方式处理多条推理路径"""
    start_time_total = time.time()  # 记录整个过程的开始时间
    
    try:
        # 使用大模型选择解题方法
        solution_methods = choose_solution_methods(problem, options, args.dataset_name, args.method_num)

        # 为每个方法创建独立的推理轨道，使用索引区分相同方法的不同实例
        tracks = []
        method_count = {}  # 用于跟踪每个方法出现的次数
        
        for method_code in solution_methods:
            # 如果是重复的方法，给它一个唯一的标识符
            if method_code in method_count:
                method_count[method_code] += 1
                track_id = f"{method_code}{method_count[method_code]}"
            else:
                method_count[method_code] = 1
                track_id = f"{method_code}1"
            
            # 创建推理轨道
            track = ReasoningTrack(
                track_id=track_id,
                method_code=method_code,
                problem=problem,
                options=options,
                dataset_name=args.dataset_name,
                score_threshold=args.score_threshold,
                score_method=args.score_method
            )
            tracks.append(track)
        
        # 使用线程池并行处理所有推理轨道
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tracks)) as executor:
            # 创建一个字典，将每个轨道映射到其对应的future
            active_tracks = {executor.submit(track.process_step): track for track in tracks}
            
            # 持续处理，直到所有轨道完成或达到token预算
            while active_tracks:
                # 等待任意一个轨道完成当前步骤
                done, not_done = concurrent.futures.wait(
                    active_tracks.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # 处理完成的轨道
                for future in done:
                    track = active_tracks.pop(future)
                    
                    # 检查是否达到token预算
                    total_tokens = sum(metadata.get("tokens", 0) for metadata in track.metadata)
                    if total_tokens >= args.token_budget:
                        logging.info(f"[Track {track.track_id}] 达到token预算 ({total_tokens}/{args.token_budget})，停止推理")
                        track.finished = True
                    
                    # 如果轨道未完成，继续处理下一步
                    if not track.finished:
                        # 提交下一步处理
                        new_future = executor.submit(track.process_step)
                        active_tracks[new_future] = track
                
                # 短暂休眠，避免过度轮询
                # time.sleep(0.01)
        
        # 计算总时间
        total_time = time.time() - start_time_total
        
        # 计算改写率
        total_steps = sum(track.total_steps for track in tracks)
        total_rewrites = sum(track.rewrite_count for track in tracks)
        avg_rewrite_rate = (total_rewrites / total_steps) * 100 if total_steps > 0 else 0

        # 构建方法轨道结果
        method_tracks = {}
        for track in tracks:
            result = track.get_result()
            method_tracks[track.track_id] = {
                "steps": result["steps"],
                "finished": result["finished"],
                "metadata": result["metadata"],
                "rewrite_count": result["rewrite_count"],
                "total_steps": result["total_steps"],
                "method_code": result["method_code"]
            }
        
        return method_tracks, total_time, avg_rewrite_rate

    except Exception as e:
        logging.error(f"推理过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        method_tracks = {}
        total_time = time.time() - start_time_total
        avg_rewrite_rate = 0
        
        return method_tracks, total_time, avg_rewrite_rate

def process_results(method_tracks, total_time, avg_rewrite_rate, args, problem, options):
    """处理推理结果，提取答案和统计信息"""
    results = []
    
    for method, track in method_tracks.items():
        if track["steps"]:
            final_reasoning = "\n\n".join(track["steps"])
            answer = extract_final_answer(final_reasoning, args.dataset_name)
            
            # 计算token数量
            num_tokens = sum(m.get("tokens", 0) for m in track["metadata"])
            
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
                # "steps": track["steps"],
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

    return metadata_with_stats

if __name__ == "__main__":
    # 解析命令行参数
    args, _ = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    args.dataset = get_dataset(args.dataset_name)
    
    # 解析问题ID范围
    problem_ids = parse_problem_range(args.problem_id)
    
    # 处理每个问题
    for problem_id in problem_ids:
        for repeat_idx in range(args.repeat_id):

            args.problem_id = problem_id
            # 准备问题数据
            problem, options = prepare_problem_data(args)
            
            # 创建问题输出目录
            problem_output_dir = os.path.join(args.output_dir, args.dataset_name)
            os.makedirs(problem_output_dir, exist_ok=True)
            
            # 检查是否已经处理过该问题
            output_filename = os.path.join(problem_output_dir, f"{problem_id}-{repeat_idx}")
            if os.path.exists(f"{output_filename}.pickle"):
                logging.info(f"问题 {problem_id} 重复 {repeat_idx} 已解决，跳过")
                continue
            
            logging.info(f"处理问题 ID: {problem_id}，重复 {repeat_idx}")
            
            # 调用推理过程
            method_tracks, total_time, avg_rewrite_rate = run_reasoning_process(args, problem, options)
            
            # 处理结果
            metadata_with_stats = process_results(method_tracks, total_time, avg_rewrite_rate, args, problem, options)
            
            # 保存结果
            with open(f"{output_filename}.pickle", "wb") as f:
                pickle.dump(metadata_with_stats, f)
            with open(f"{output_filename}.txt", "w") as f:
                pprint.pprint(metadata_with_stats, stream=f)
                
            logging.info(f"问题 {problem_id} 重复 {repeat_idx} 已完成，结果已保存")

    logging.info(f"所有任务完成！共处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次")
