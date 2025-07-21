#python base_test.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/baseline_vllm_test

import os
import time
import pickle
import pprint
import logging
import argparse
import statistics
import re
from collections import Counter
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化全局模型变量
model = None

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

def generate_full_reasoning(problem, options=None, max_tokens=8192):
    """使用模型一次性生成完整的推理过程"""
    global model
    
    start_time = time.time()  # 记录开始时间
    
    prompt = get_first_user_msg(problem, options)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens
    )
    
    # 生成回答
    outputs = model.generate(prompt, sampling_params)
    response = outputs[0]  # 获取第一个（也是唯一的）回答
    
    reasoning = response.outputs[0].text
    num_output_tokens = len(response.outputs[0].token_ids)
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

    # 规范输出文件路径
    model_name_short = args.model_name.split('/')[-1]  # 提取模型名称的最后部分作为目录名
    output_dir = os.path.join(args.output_dir, f"{model_name_short}", f"{args.dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{problem_id}-{repeat_id}")
    
    if os.path.exists(f"{output_filename}.pickle"):
        logging.info(f"问题 {problem_id} 使用模型 {args.model_name} 重复 {repeat_id} 已解决，跳过")
        return

    start_time_total = time.time()  # 记录整个过程的开始时间

    try:
        # 使用指定模型一次性生成完整的推理过程
        reasoning, finished, num_tokens, elapsed_time, token_time = generate_full_reasoning(
            problem, 
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
            "model_name": args.model_name,  # 记录使用的模型名称
        }
        
        # 从推理文本中提取最终答案
        final_answer = extract_final_answer(reasoning, args.dataset_name)
        metadata["final_answer"] = final_answer
        
        # 如果没有完成，记录原因
        if not finished:
            metadata["stop_reason"] = "budget"
        else:
            metadata["stop_reason"] = "finished"
            
    except Exception as e:
        logging.error(f"在生成过程中捕获到错误: {e}，继续")
        return

    # 计算总时间
    total_time = time.time() - start_time_total

    # 添加时间统计信息到元数据
    time_stats = {
        "total_time": total_time,
        "model_stats": {
            "model_name": args.model_name,
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
    logging.info(f"模型名称: {args.model_name}")
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
            logging.warning("在推理文本中找到了\\boxed{,但没有找到匹配的}，可能是格式错误")
            return None
    else:
        # 备用方案：检查最后一步是否包含"Answer: X"（用于GPQA）
        if dataset_name == "gpqa":
            match_answer = re.search(r"[Aa]nswer:\s*([A-D])", reasoning)
            if match_answer:
                return match_answer.group(1).upper()

        logging.warning(f"在最终步骤中找不到\\boxed{{}}模式")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用vllm原生接口运行推理")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], default="aime",
                        help="数据集")
    parser.add_argument("--token_budget", type=int, default=8192,
                        help="最大输出令牌数")
    # problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA, 0-45 for Live
    parser.add_argument("--problem_id", type=str, default="60",
                        help="查询ID (AIME为60-89)，可以是单个ID或范围，例如：60-89")
    parser.add_argument("--repeat_id", type=int, default=1,
                        help="每个问题重复执行的次数")
    parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B",
                        help="用于推理的模型名称（可在huggingface上下载）")
    parser.add_argument("--output_dir", type=str, default="results/baseline_vllm_test", 
                        help="结果pickle文件的写入位置")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU内存利用率")
    args, _ = parser.parse_known_args()
    
    # 初始化全局模型
    global model
    logging.info(f"正在加载模型: {args.model_name}")
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto"
    )
    logging.info(f"模型加载完成")
    
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