import os
import logging
import argparse
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams

from prompts.choose_prompts import math_choose_prompt, gpqa_choose_prompt

target_model_name = "Qwen/QwQ-32B"
draft_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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

def format_chat(messages,
                model_name: str,
                add_generation_prompt: bool = True,
                continue_final_message: bool = False) -> str:
    """
    根据 model_name 选择格式化逻辑，将 messages 列表转换为 prompt 字符串。

    对于 Qwen/QwQ-32B:
      - ChatML 样式: <|im_start|>role\ncontent<|im_end|>\n
      - 末尾若 add_generation_prompt，则添加 <|im_start|>assistant\n

    对于 DeepSeek-R1-Distill-Qwen-1.5B:
      - 深搜自定义格式:
        开始: <｜begin▁of▁sentence｜>
        用户: <｜User｜>content
        助手: <｜Assistant｜>content<｜end▁of▁sentence｜>  （仅在 assistant 后加 end）
      - 末尾若 add_generation_prompt，则追加 <｜Assistant｜><think>
      - 若 continue_final_message=True，则最后一条 assistant 不插入 end

    Args:
      messages: List[{"role": "user"/"assistant", "content": str}]
      model_name: 模型名称，用于分支选择
      add_generation_prompt: 是否在末尾加入生成提示
      continue_final_message: 是否让模型在最后一条消息基础上续写
    """
    # Qwen/QwQ-32B 分支（不变）
    if model_name.lower().startswith("qwen/qwq-32b"):
        prompt = ""
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if continue_final_message and idx == len(messages) - 1:
                prompt += f"<|im_start|>{role}\n{content}"
            else:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n<think>"
        return prompt

    # DeepSeek-R1-Distill-Qwen-1.5B 分支（修正后）
    elif model_name.lower().startswith("deepseek-ai/deepseek-r1-distill-qwen-1.5b"):
        prompt = "<｜begin▁of▁sentence｜>"
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if role == "user":
                # 只加用户标签和内容，不加 end
                prompt += f"<｜User｜>{content}"
            else:  # assistant
                prompt += f"<｜Assistant｜>{content}"
                # 仅在 assistant 消息后添加 end，且不在最后一条需续写的情况下
                if not (continue_final_message and idx == len(messages) - 1):
                    prompt += "<｜end▁of▁sentence｜>"
        # 生成提示
        if add_generation_prompt:
            prompt += "<｜Assistant｜><think>"
        return prompt

    else:
        raise ValueError(f"Unsupported model: {model_name}")

def initialize_models(GPU_num):
    logging.info("正在初始化模型...")
    models = {}
    try:
        # 初始化大模型
        logging.info("正在加载目标模型...")
        models["target"] = LLM(
            model="Qwen/QwQ-32B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        logging.info("模型初始化完成")
        return models
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
        exit(1)

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
    prompt = f"{prompt}\n\nExamine the problem and select the **{method_num}** strategies (by their codes, e.g. {'B,E,F' if method_num == 3 else 'B,E,F,A' if method_num == 4 else 'B,E'}) that you believe are most promising for solving it. You only need to output {method_num} codes, without any other symbols or text."
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>I think the best {method_num} methods are: "},
    ]
    prompt = format_chat(messages, target_model_name, add_generation_prompt=False, continue_final_message=True)

    # 设置生成参数，根据method_num调整max_tokens
    scoring_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=10
    )
    
    output = models["target"].generate(prompt, scoring_params)
    
    logprobs = output.outputs[0].logprobs[0]

    return logprobs

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
    parser.add_argument("--fast", type=int, choices=[1, 2], default=None,
                        help="Fast mode: 1=最快模式(一个答案即停止), 2=次优模式(两个相同答案即停止)")
    return parser.parse_known_args()

def parse_problem_range(problem_id_str):
    """解析问题ID范围，支持单个ID或范围（如60-89）"""
    if '-' in problem_id_str:
        start, end = map(int, problem_id_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(problem_id_str)]

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


if __name__ == "__main__":
    args, _ = parse_arguments()
    
    # 初始化模型
    global models
    models = initialize_models(2)
    
    # 加载数据集
    args.dataset = get_dataset(args.dataset_name)
    
    # 解析问题ID范围
    problem_ids = parse_problem_range(args.problem_id)

    for problem_id in problem_ids:
        for repeat_idx in range(args.repeat_id):
            args.problem_id = problem_id  # 更新当前处理的问题ID
            problem, options = prepare_problem_data(args)
            choose_solution_methods(problem, options, args.dataset_name, models, args.method_num)

# CUDA_VISIBLE_DEVICES=4,5 python api_test/choose_test.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale_2_m5_t7_f1 --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 1