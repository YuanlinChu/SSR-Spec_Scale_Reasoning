# python spec_reason.py --dataset_name aime --problem_id 60 --repeat_id 2 --score_threshold 7.0 --token_budget 8192 --score_method greedy --output_dir results/spec_7

import os
import time
import pickle
import pprint
import logging
import argparse
import numpy as np
import statistics
from collections import Counter
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_avg_score(scores):
    return statistics.mean([x for x in scores if x is not None])

def get_frequency(scores):
    return dict(Counter(scores))

# 初始化模型
model_names = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "32b": "Qwen/QwQ-32B",
}

# 初始化全局模型变量
models = {}

def initialize_models():
    """初始化模型"""
    global models
    
    # 初始化小模型
    logging.info("正在加载草稿模型...")
    models["1.5b"] = LLM(
        model=model_names["1.5b"],
        tensor_parallel_size=4,
        gpu_memory_utilization=0.15,
        trust_remote_code=True
    )
    
    # 初始化大模型
    logging.info("正在加载目标模型...")
    models["32b"] = LLM(
        model=model_names["32b"],
        tensor_parallel_size=4,
        gpu_memory_utilization=0.75,
        trust_remote_code=True
    )
    logging.info("模型初始化完成")

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

def generate_new_step(problem, steps_so_far, model_size, options=None, stop_token="\n\n"):
    global models
    model = models[model_size]
    
    if steps_so_far == []:  # first step
        prompt = get_first_user_msg(problem, options)
    else:  # continuing on from a previous message
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        prompt = f"{get_first_user_msg(problem, options)}\n<think>{steps_so_far_str}"
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=512,
        stop=[stop_token]
    )
    
    # 生成回答
    outputs = model.generate([prompt], sampling_params)
    output = outputs[0]
    
    step_str = output.outputs[0].text
    num_output_tokens = len(output.outputs[0].token_ids)
    finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]])
    
    return step_str, finished, num_output_tokens

def get_score(args, problem, steps_so_far, model_size="32b", options=None):
    global models
    model = models[model_size]
    
    steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
    prompt = f"{get_first_user_msg(problem, options)}\n<think>{steps_so_far_str}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=10
    )
    
    # 生成评分
    outputs = model.generate([prompt], sampling_params)
    output = outputs[0]
    
    token = output.outputs[0].text
    logprobs = output.outputs[0].logprobs[0]
    
    score = process_vllm_logprobs(token, logprobs, method=args.score_method)
    
    return score, token, output

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

def get_dataset(dataset_name):
    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "live":
        dataset = load_dataset("opencompass/LiveMathBench", "v202412_AMC_en")["test"]
    elif dataset_name == "gpqa":
        if os.getenv("HF_HUB_OFFLINE", "0") == "1":
            dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
        else:    
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    else:
        raise NotImplementedError
    return dataset

def parse_problem_range(problem_id_str):
    """解析问题ID范围，支持单个ID或范围（如60-89）"""
    if '-' in problem_id_str:
        start, end = map(int, problem_id_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(problem_id_str)]

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

def run_speculative_reasoning(args, problem_id, repeat_idx):
    """运行推测推理过程"""
    # 准备问题数据
    dataset = args.dataset
    
    if args.dataset_name == "aime":
        problem = dataset["problem"][problem_id - 60]
        options = None
    elif args.dataset_name == "math":
        problem = dataset["problem"][problem_id]
        options = None
    elif args.dataset_name == "live":
        problem = dataset["question"][problem_id]
        options = None
    elif args.dataset_name == "gpqa":
        problem = dataset["Question"][problem_id]
        options = {
            "A": dataset["Correct Answer"][problem_id],
            "B": dataset["Incorrect Answer 1"][problem_id],
            "C": dataset["Incorrect Answer 2"][problem_id],
            "D": dataset["Incorrect Answer 3"][problem_id],
        }

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{problem_id}-{repeat_idx}")
    
    if os.path.exists(f"{output_filename}.pickle"):
        logging.info(f"问题 {problem_id} 重复 {repeat_idx} 已解决，跳过")
        return
        
    steps_so_far = []
    step_id = 0
    metadata_list = []

    start_time_total = time.time()  # 记录整个过程的开始时间

    try:
        while True:
            warning_flag = False
            
            # 1. generate a reasoning step using a small model
            step_str, finished, num_output_tokens = generate_new_step(problem, steps_so_far, "1.5b", options=options)
            small_model_step, num_output_tokens_small = step_str, num_output_tokens

            # 2. use the base model to score the step
            score, justification, response = get_score(args, problem, steps_so_far + [step_str], options=options)
            
            # 3. if over a threshold, accept. else, generate a reasoning step using the base model.
            if score is not None and score >= args.score_threshold:
                logging.info(f"[Step {step_id}] score {score}, accepted!")
                base_model_step, num_output_tokens_base = None, None
            else:
                logging.info(f"[Step {step_id}] score {score} rejected, falling back to base model")
                step_str, finished, num_output_tokens = generate_new_step(problem, steps_so_far, "32b", options=options)
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
            
            # 检查是否已经完成
            if finished:
                logging.info(f"[Step {step_id-1}] 推理完成!")
                break
                
            # 检查是否超出token预算
            total_tokens = sum(m["final_num_output_tokens"] for m in metadata_list)
            if total_tokens >= args.token_budget:
                logging.info(f"[Step {step_id-1}] 达到token预算上限 ({total_tokens} >= {args.token_budget})，停止生成")
                break
                
    except Exception as e:
        logging.error(f"在生成过程中捕获到错误: {e}")
        
    # 计算总时间
    total_time = time.time() - start_time_total
    
    # 提取最终答案
    final_reasoning = "\n\n".join(steps_so_far)
    final_answer = extract_final_answer(final_reasoning, args.dataset_name)
    
    # 计算统计信息
    total_tokens = sum(m["final_num_output_tokens"] for m in metadata_list)
    small_model_tokens = sum(m["num_output_tokens_small"] for m in metadata_list)
    base_model_tokens = sum(m["num_output_tokens_base"] for m in metadata_list if m["num_output_tokens_base"] is not None)
    
    # 计算小模型接受率
    total_steps = len(metadata_list)
    accepted_steps = sum(1 for m in metadata_list if m["base_model_step"] is None)
    acceptance_rate = (accepted_steps / total_steps) * 100 if total_steps > 0 else 0
    
    # 打印统计信息
    logging.info("=" * 50)
    logging.info("推理统计信息:")
    logging.info(f"问题ID: {problem_id}")
    logging.info(f"总步数: {total_steps}")
    logging.info(f"小模型接受步数: {accepted_steps} ({acceptance_rate:.2f}%)")
    logging.info(f"总token数: {total_tokens}")
    logging.info(f"小模型token数: {small_model_tokens}")
    logging.info(f"大模型token数: {base_model_tokens}")
    logging.info(f"总耗时: {total_time:.2f}秒")
    logging.info(f"最终答案: {final_answer}")
    logging.info("=" * 50)
    
    # 保存结果
    result = {
        "metadata": metadata_list,
        "steps": steps_so_far,
        "final_reasoning": final_reasoning,
        "answer": final_answer,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "small_model_tokens": small_model_tokens,
        "base_model_tokens": base_model_tokens,
        "acceptance_rate": acceptance_rate,
        "finished": finished if 'finished' in locals() else False,
        "stop_reason": "finished" if finished else "budget" if total_tokens >= args.token_budget else "error"
    }
    
    with open(f"{output_filename}.pickle", "wb") as f:
        pickle.dump(result, f)
    with open(f"{output_filename}.txt", "w") as f:
        pprint.pprint(result, stream=f)

    logging.info(f"结果已保存到: {output_filename}.pickle")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Runs speculative reasoning using a small model")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa","live"], default="aime",
                        help="Dataset")
    parser.add_argument("--score_threshold", type=float, default=7.0, 
                        help="Acceptance threshold")
    parser.add_argument("--token_budget", type=int, default=8192,
                        help="Max num of total output tokens in each step")
    # problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA
    parser.add_argument("--problem_id", type=str, default="60",
                        help="Query ID (60-89 for AIME), can be a single ID or a range like 60-89")
    parser.add_argument("--repeat_id", type=int, default=1,
                        help="Number of times to repeat each problem")
    parser.add_argument("--score_method", type=str, choices=["greedy", "average"], default="greedy",
                        help="Scoring method")
    parser.add_argument("--output_dir", type=str, default="results/spec_7", 
                        help="Where result pickle files will be written to")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for model initialization")
    return parser.parse_known_args()

def main():
    """主函数"""
    # 解析命令行参数
    args, _ = parse_arguments()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 初始化模型
    initialize_models()
    
    # 加载数据集
    args.dataset = get_dataset(args.dataset_name)
    
    # 解析问题ID范围
    problem_ids = parse_problem_range(args.problem_id)
    logging.info(f"将处理问题ID: {problem_ids}")
    
    # 处理每个问题ID，并对每个问题执行指定次数的重复
    total_tasks = len(problem_ids) * args.repeat_id
    completed_tasks = 0
    
    for problem_id in problem_ids:
        for repeat_idx in range(args.repeat_id):
            logging.info(f"处理问题 ID: {problem_id}，重复 {repeat_idx}/{args.repeat_id}")
            run_speculative_reasoning(args, problem_id, repeat_idx)
            completed_tasks += 1
            logging.info(f"完成进度: {completed_tasks}/{total_tasks} ({completed_tasks/total_tasks*100:.2f}%)")
    
    logging.info("所有任务已完成")

if __name__ == "__main__":
    main()

