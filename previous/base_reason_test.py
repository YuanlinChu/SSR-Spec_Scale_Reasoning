#python base_reason_test.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --model_size 32b --output_dir results/baseline_test

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
    # "Llama8b": "meta-llama/Llama-3.1-8B",
    "32b": "Qwen/QwQ-32B",
}
ports = {
    "1.5b": "30001",
    "32b": "30000",
    # "Llama8b": "30002",
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
def generate_full_reasoning(problem, model_size, options=None, max_tokens=8192):
    """使用模型一次性生成完整的推理过程"""
    client = clients[model_size]
    
    start_time = time.time()  # 记录开始时间
    
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
    ]
    
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.6, 
        top_p=0.95,  # https://huggingface.co/Qwen/QwQ-32B#usage-guidelines
        max_tokens=max_tokens,
    )

    reasoning = response.choices[0].message.content
    num_output_tokens = response.usage.completion_tokens
    finished = any([x in reasoning for x in ["boxed", "Answer:", "ANSWER:"]])
    
    elapsed_time = time.time() - start_time  # 计算耗时
    
    # 计算每个token的平均时间（毫秒）
    time_per_token = (elapsed_time * 1000) / num_output_tokens if num_output_tokens > 0 else 0
    
    return reasoning, finished, num_output_tokens, elapsed_time, time_per_token

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

def run_baseline_test(args, problem_id, repeat_id):
    """运行基线测试"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = get_dataset(args.dataset_name)

    if args.dataset_name == "aime":
        problem = dataset["problem"][problem_id - 60]
        options = None
    elif args.dataset_name == "math":
        problem = dataset["problem"][problem_id]
        options = None
    elif args.dataset_name == "gpqa":
        problem = dataset["Question"][problem_id]
        options = {
            "A": dataset["Correct Answer"][problem_id],
            "B": dataset["Incorrect Answer 1"][problem_id],
            "C": dataset["Incorrect Answer 2"][problem_id],
            "D": dataset["Incorrect Answer 3"][problem_id],
        }

    # 规范输出文件路径
    output_dir = os.path.join(args.output_dir, f"{args.model_size}", f"{args.dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{problem_id}-{repeat_id}")
    
    if os.path.exists(f"{output_filename}.pickle"):
        logging.info(f"问题 {problem_id} 使用模型 {args.model_size} 重复 {repeat_id} 已解决，跳过")
        return

    start_time_total = time.time()  # 记录整个过程的开始时间

    try:
        # 使用指定模型一次性生成完整的推理过程
        reasoning, finished, num_tokens, elapsed_time, token_time = generate_full_reasoning(
            problem, 
            model_size=args.model_size, 
            options=options, 
            max_tokens=args.token_budget
        )
        
        # 记录结果
        metadata = {
            "reasoning": reasoning,
            "finished": finished,
            "num_tokens": num_tokens,
            "elapsed_time": elapsed_time,
            "token_time": token_time,  # 毫秒/token
            "model_size": args.model_size,  # 记录使用的模型大小
        }
        
        # 如果没有完成，记录原因
        if not finished:
            metadata["stop_reason"] = "budget"
        else:
            metadata["stop_reason"] = "finished"
            
    except ValueError as e:
        logging.error(f"在聊天模板应用中捕获到ValueError: {e}，继续")
        return

    # 计算总时间
    total_time = time.time() - start_time_total

    # 添加时间统计信息到元数据
    time_stats = {
        "total_time": total_time,
        "model_stats": {
            "model_size": args.model_size,
            "total_time": elapsed_time,
            "time_percentage": (elapsed_time / total_time) * 100 if total_time > 0 else 0,
            "token_time": token_time,  # 毫秒/token
        },
        "total_tokens": num_tokens
    }

    # 打印时间统计信息
    logging.info("=" * 50)
    logging.info("时间统计信息:")
    logging.info(f"问题ID: {problem_id}, 重复ID: {repeat_id}")
    logging.info(f"总时间: {total_time:.2f}秒")
    logging.info(f"模型大小: {args.model_size}")
    logging.info(f"模型生成:")
    logging.info(f"  - 总时间: {elapsed_time:.2f}秒 ({(elapsed_time/total_time)*100:.2f}%)")
    logging.info(f"  - 平均token时间: {token_time:.2f}毫秒/token")
    logging.info(f"  - 总token数: {num_tokens}")
    logging.info(f"  - 是否完成: {finished}")
    if not finished:
        logging.info(f"  - 停止原因: 达到token预算上限")
    else:
        logging.info(f"  - 停止原因: 推理完成")
    logging.info("=" * 50)

    # 将时间统计信息添加到元数据
    metadata_with_stats = {
        "reasoning": metadata,
        "time_stats": time_stats
    }

    with open(f"{output_filename}.pickle", "wb") as f:
        pickle.dump(metadata_with_stats, f)

    with open(f"{output_filename}.txt", "w") as f:
        pprint.pprint(metadata_with_stats, stream=f)

def parse_problem_range(problem_id_str):
    """解析问题ID范围，支持单个ID或范围（如60-89）"""
    if '-' in problem_id_str:
        start, end = map(int, problem_id_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(problem_id_str)]

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用单个模型运行推理")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="aime",
                        help="数据集")
    parser.add_argument("--token_budget", type=int, default=15900,  #8192
                        help="最大输出令牌数")
    # problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA
    parser.add_argument("--problem_id", type=str, default="60",
                        help="查询ID (AIME为60-89)，可以是单个ID或范围，例如：60-89")
    parser.add_argument("--repeat_id", type=int, default=1,
                        help="每个问题重复执行的次数")
    parser.add_argument("--model_size", type=str, choices=["1.5b", "32b", "Llama8b"], default="32b",
                        help="用于推理的模型大小")
    parser.add_argument("--output_dir", type=str, default="results/baseline_test", 
                        help="结果pickle文件的写入位置")
    args, _ = parser.parse_known_args()
    
    # 解析问题ID范围
    problem_ids = parse_problem_range(args.problem_id)
    
    # 处理每个问题ID，并对每个问题执行指定次数的重复
    total_tasks = len(problem_ids) * args.repeat_id
    completed_tasks = 0
    
    logging.info(f"开始处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次，共 {total_tasks} 个任务")
    
    for problem_id in problem_ids:
        for repeat_idx in range(args.repeat_id):
            logging.info(f"处理问题 ID: {problem_id}，重复 {repeat_idx}/{args.repeat_id}")
            run_baseline_test(args, problem_id, repeat_idx)
            completed_tasks += 1
            logging.info(f"完成进度: {completed_tasks}/{total_tasks} ({completed_tasks/total_tasks*100:.2f}%)")
    
    logging.info(f"所有任务完成！共处理 {len(problem_ids)} 个问题，每个问题重复 {args.repeat_id} 次")

if __name__ == "__main__":
    main()

