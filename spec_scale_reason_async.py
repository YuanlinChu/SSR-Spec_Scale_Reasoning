# python spec_scale_reason_async.py --dataset_name aime --problem_id 60 --num_repeats 3 --score_threshold 7.0 --token_budget 8192 --score_method greedy
import os
import time
import pickle
import pprint
import uuid
import logging
import argparse
import numpy as np
from collections import Counter
from datasets import load_dataset, load_from_disk
import re
import asyncio
import traceback
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

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
    
async def initialize_models_async(GPU_num):
    """异步初始化小模型和大模型"""
    logging.info("正在初始化模型...")
    models = {}
    try:
        # 先初始化小模型，使用单GPU
        logging.info("正在加载草稿模型...")
        draft_engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            tensor_parallel_size=4,  # 只使用一个GPU
            gpu_memory_utilization=0.15,  # 降低内存使用率
            trust_remote_code=True,
            max_model_len=4096  # 减小上下文长度
        )
        models["draft"] = AsyncLLMEngine.from_engine_args(draft_engine_args)
        
        # 然后初始化大模型，使用所有GPU
        logging.info("正在加载目标模型...")
        target_engine_args = AsyncEngineArgs(
            model="Qwen/QwQ-32B",
            tensor_parallel_size=GPU_num,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=8192,
            dtype="bfloat16"
        )
        models["target"] = AsyncLLMEngine.from_engine_args(target_engine_args)
        
        # 验证模型状态
        logging.info("正在验证模型状态...")
        test_prompt = "Hello, world!"
        test_params = SamplingParams(temperature=0.0, max_tokens=8)
        
        try:
            # 测试草稿模型
            draft_test = await generate_async(models["draft"], [test_prompt], test_params)
            logging.info(f"草稿模型响应测试成功: {draft_test[0].outputs[0].text[:20]}...")
            
            # 测试目标模型
            target_test = await generate_async(models["target"], [test_prompt], test_params) 
            logging.info(f"目标模型响应测试成功: {target_test[0].outputs[0].text[:20]}...")
            
        except Exception as e:
            logging.error(f"模型响应测试失败: {str(e)}")
            logging.error(traceback.format_exc())
            raise
        
        logging.info("模型初始化完成")
        return models
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
        logging.error(traceback.format_exc())
        exit(1)

# 异步推理函数
async def generate_async(model, prompts, params):
    """异步生成文本，支持真正的并行处理"""
    if not prompts:
        return []
    
    # 创建一个任务列表
    async def process_single_prompt(prompt):
        request_id = str(uuid.uuid4())
        last_output = None
        
        # 使用generate方法
        async for output in model.generate(
            prompt=prompt, 
            sampling_params=params, 
            request_id=request_id
        ):
            last_output = output
            
        return last_output
    
    # 创建并行任务
    tasks = [process_single_prompt(prompt) for prompt in prompts]
    
    # 并行等待所有结果
    return await asyncio.gather(*tasks)

# 角色状态处理器
class RoleProcessor:
    def __init__(self, role, track, problem, options, args):
        self.role = role
        self.track = track
        self.problem = problem
        self.options = options
        self.args = args
        self.step_id = 0
    
    async def process(self):
        """处理角色的一个完整步骤"""
        if self.track["finished"]:
            return
        
        try:
            # 1. 小模型推理
            generated_text = await self.run_draft_model()
            logging.info(f"[Step {self.step_id}] 角色 {self.role} 草稿模型已生成 {len(generated_text)} 字符")
            
            # 2. 大模型评分
            score = await self.run_scoring()
            
            # 3. 如果需要，大模型重写
            if score is None or score < self.args.score_threshold:
                logging.info(f"[Step {self.step_id}] 角色 {self.role} 分数 {score}，需要重写")
                await self.run_rewriting()
            else:
                logging.info(f"[Step {self.step_id}] 角色 {self.role} 分数 {score}，已接受")
                self.update_track(False)
            
            self.step_id += 1
        except Exception as e:
            logging.error(f"角色 {self.role} 处理步骤 {self.step_id} 时出错: {str(e)}")
            logging.error(traceback.format_exc())
            # 标记此角色为完成状态，避免继续处理
            self.track["finished"] = True
            self.track["stop_reason"] = f"error: {str(e)}"
    
    async def run_draft_model(self):
        """运行小模型推理"""
        try:
            if not self.track["steps"]:  # 第一步
                prompt = get_first_user_msg(self.problem, self.options, role_type=self.role)
            else:  # 继续之前的消息
                steps_so_far_str = "\n\n".join(self.track["steps"]) + "\n\n"
                prompt = f"{get_first_user_msg(self.problem, self.options, role_type=self.role)}\n<think>{steps_so_far_str}"
            
            logging.debug(f"角色 {self.role} 准备小模型推理，提示长度: {len(prompt)}")
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=512,
                stop=["\n\n"]
            )
            
            # 异步生成
            start_time = time.time()
            outputs = await generate_async(models["draft"], [prompt], sampling_params)
            elapsed_time = time.time() - start_time
            logging.debug(f"角色 {self.role} 小模型推理完成，耗时: {elapsed_time:.2f}秒")
            
            output = outputs[0]
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
            
            # 存储小模型结果
            self.track["temp_data"]["small_result"] = {
                "text": generated_text,
                "tokens": num_tokens,
                "finished": finished
            }
            
            return generated_text
        except Exception as e:
            logging.error(f"角色 {self.role} 小模型推理出错: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    async def run_scoring(self):
        """运行大模型评分"""
        try:
            small_result = self.track["temp_data"]["small_result"]
            generated_text = small_result["text"]
            
            # 准备评分提示
            steps_for_scoring = self.track["steps"] + [generated_text]
            steps_so_far_str = "\n\n".join(steps_for_scoring) + "\n\n"
            
            score_prompt = f"{get_first_user_msg(self.problem, self.options, role_type=self.role)}\n<think>{steps_so_far_str}\nEvaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9.\n<think>I think the quality score is: "
            
            logging.debug(f"角色 {self.role} 准备大模型评分，提示长度: {len(score_prompt)}")
            
            # 设置评分参数
            scoring_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=10
            )
            
            # 异步评分
            start_time = time.time()
            outputs = await generate_async(models["target"], [score_prompt], scoring_params)
            elapsed_time = time.time() - start_time
            logging.debug(f"角色 {self.role} 大模型评分完成，耗时: {elapsed_time:.2f}秒")
            
            output = outputs[0]
            token = output.outputs[0].text
            logprobs = output.outputs[0].logprobs[0]
            
            score = process_vllm_logprobs(token, logprobs, method=self.args.score_method)
            self.track["temp_data"]["score"] = score
            
            return score
        except Exception as e:
            logging.error(f"角色 {self.role} 大模型评分出错: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    async def run_rewriting(self):
        """运行大模型重写"""
        try:
            # 准备重写提示
            if not self.track["steps"]:  # 第一步
                prompt = get_first_user_msg(self.problem, self.options, role_type=self.role)
            else:  # 继续之前的消息
                steps_so_far_str = "\n\n".join(self.track["steps"]) + "\n\n"
                prompt = f"{get_first_user_msg(self.problem, self.options, role_type=self.role)}\n<think>{steps_so_far_str}"
            
            logging.debug(f"角色 {self.role} 准备大模型重写，提示长度: {len(prompt)}")
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=512,
                stop=["\n\n"]
            )
            
            # 异步重写
            start_time = time.time()
            outputs = await generate_async(models["target"], [prompt], sampling_params)
            elapsed_time = time.time() - start_time
            logging.debug(f"角色 {self.role} 大模型重写完成，耗时: {elapsed_time:.2f}秒")
            
            output = outputs[0]
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            finished = any([x in generated_text for x in ["boxed", "Answer:", "ANSWER:"]])
            
            # 存储重写结果
            self.track["temp_data"]["rewrite_result"] = {
                "text": generated_text,
                "tokens": num_tokens,
                "finished": finished
            }
            
            # 更新轨道
            self.update_track(True)
        except Exception as e:
            logging.error(f"角色 {self.role} 大模型重写出错: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def update_track(self, used_base_model):
        """更新角色轨道"""
        # 确定最终使用的步骤
        if used_base_model:
            # 使用重写的结果
            step_text = self.track["temp_data"]["rewrite_result"]["text"]
            num_tokens = self.track["temp_data"]["rewrite_result"]["tokens"]
            finished = self.track["temp_data"]["rewrite_result"]["finished"]
            # 增加改写计数
            self.track["rewrite_count"] += 1
        else:
            # 使用小模型的结果
            step_text = self.track["temp_data"]["small_result"]["text"]
            num_tokens = self.track["temp_data"]["small_result"]["tokens"]
            finished = self.track["temp_data"]["small_result"]["finished"]
        
        # 增加总步骤计数
        self.track["total_steps"] += 1
        
        # 处理特殊情况
        if "</think>" in step_text and not any([x in step_text for x in ["boxed", "Answer:", "ANSWER:"]]):
            logging.warning(f"Warning: 角色 {self.role} 的步骤包含 </think>，正在移除")
            step_text = step_text.replace("</think>", "")
            warning_flag = True
        else:
            warning_flag = False
        
        # 添加到轨道的步骤中
        self.track["steps"].append(step_text)
        
        # 创建元数据
        metadata = {
            "step_id": self.step_id,
            "role_type": self.role,
            "step_str": step_text,
            "small_model_step": self.track["temp_data"]["small_result"]["text"],
            "num_output_tokens_small": self.track["temp_data"]["small_result"]["tokens"],
            "score": self.track["temp_data"].get("score"),
            "base_model_step": self.track["temp_data"].get("rewrite_result", {}).get("text") if used_base_model else None,
            "num_output_tokens_base": self.track["temp_data"].get("rewrite_result", {}).get("tokens") if used_base_model else None,
            "final_num_output_tokens": num_tokens,
            "used_base_model": used_base_model
        }
        
        if warning_flag:
            metadata["warning"] = "step_str had a </think>"
        
        self.track["metadata"].append(metadata)
        
        # 检查是否完成
        if finished:
            self.track["finished"] = True
            logging.info(f"角色 {self.role} 已完成推理")
        
        # 检查是否达到token预算
        total_tokens = sum(m.get("final_num_output_tokens", 0) for m in self.track["metadata"])
        if total_tokens >= self.args.token_budget:
            self.track["finished"] = True
            logging.warning(f"角色 {self.role} 达到token预算上限 {self.args.token_budget}，强制结束")
        
        # 清空临时数据
        self.track["temp_data"] = {}

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
            # answer = extract_answer("\n\n".join(track["steps"]))
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
        "num_repeats": args.num_repeats,
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

# 添加GPU监控函数
def get_gpu_memory_info():
    """获取GPU内存使用情况"""
    try:
        import torch
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)  # 转换为GB
            total_mem = torch.cuda.mem_get_info(i)[1] / (1024**3)
            used_mem = total_mem - free_mem
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "used_mem_gb": used_mem,
                "total_mem_gb": total_mem,
                "used_percent": used_mem / total_mem * 100
            })
        return gpu_info
    except Exception as e:
        logging.error(f"获取GPU信息失败: {str(e)}")
        return []

# 定期打印GPU使用情况的函数
async def monitor_gpu_memory(interval=30):
    """异步监控GPU内存使用情况"""
    while True:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu in gpu_info:
                logging.info(f"GPU {gpu['index']} ({gpu['name']}): 已用 {gpu['used_mem_gb']:.2f}/{gpu['total_mem_gb']:.2f} GB ({gpu['used_percent']:.2f}%)")
        await asyncio.sleep(interval)

# 在run_reasoning_process_async函数中添加GPU监控
async def run_reasoning_process_async(args, problem, options):
    """异步运行推理过程"""
    start_time_total = time.time()  # 记录整个过程的开始时间
    
    try:
        # 启动GPU监控任务
        monitor_task = None
        try:
            import torch
            if torch.cuda.is_available():
                monitor_task = asyncio.create_task(monitor_gpu_memory(interval=30))
                logging.info("GPU监控已启动")
        except ImportError:
            logging.info("无法导入torch库，跳过GPU监控")
        
        # 为每个角色创建独立的推理轨道
        role_tracks = {
            1: {"steps": [], "finished": False, "metadata": [], "rewrite_count": 0, "total_steps": 0, "temp_data": {}},
            2: {"steps": [], "finished": False, "metadata": [], "rewrite_count": 0, "total_steps": 0, "temp_data": {}},
            3: {"steps": [], "finished": False, "metadata": [], "rewrite_count": 0, "total_steps": 0, "temp_data": {}}
        }
        
        # 创建角色处理器
        processors = {
            role: RoleProcessor(role, track, problem, options, args)
            for role, track in role_tracks.items()
        }
        
        # 最大步骤数
        max_steps = args.token_budget // 512
        
        # 主循环
        for step_id in range(max_steps):
            # 检查是否所有轨道都已完成
            active_roles = [role for role, track in role_tracks.items() if not track["finished"]]
            if not active_roles:
                break
                
            logging.info(f"[Step {step_id}] 活跃角色: {active_roles}")
            
            # 并行处理所有活跃角色的一个步骤
            # 创建异步任务列表
            tasks = [processors[role].process() for role in active_roles]
            
            # 等待所有任务完成
            if tasks:
                await asyncio.gather(*tasks)
        
        # 停止GPU监控
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # 计算改写率
        rewrite_rates = {}
        total_rewrite_rate = 0.0
        valid_roles = 0
        
        for role, track in role_tracks.items():
            if track["total_steps"] > 0:
                rewrite_rate = track["rewrite_count"] / track["total_steps"] * 100
                rewrite_rates[role] = rewrite_rate
                total_rewrite_rate += rewrite_rate
                valid_roles += 1
                logging.info(f"角色 {role} 改写率: {rewrite_rate:.2f}% ({track['rewrite_count']}/{track['total_steps']})")
        
        # 计算平均改写率
        avg_rewrite_rate = total_rewrite_rate / valid_roles if valid_roles > 0 else 0
        logging.info(f"平均改写率: {avg_rewrite_rate:.2f}%")
        
        # 计算总时间
        total_time = time.time() - start_time_total
        
        # 使用已有的 process_results 函数处理结果
        metadata_with_stats = process_results(role_tracks, total_time, avg_rewrite_rate, args, problem, options)
        
        return metadata_with_stats
    
    except Exception as e:
        logging.error(f"推理过程中出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

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
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Number of times to repeat each problem")
    parser.add_argument("--score_method", type=str, choices=["greedy", "average"], default="greedy",
                        help="Scoring method")
    parser.add_argument("--output_dir", type=str, default="results/asyn_spec_scale", 
                        help="Where result pickle files will be written to")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallel size for model initialization")
    return parser.parse_known_args()

def parse_problem_range(problem_id_str):
    """解析问题ID范围，支持单个ID或范围（如60-89）"""
    if '-' in problem_id_str:
        start, end = map(int, problem_id_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(problem_id_str)]

def main():
    """主函数"""
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('spec_scale_reason.log')
        ]
    )
    
    # 打印系统信息
    import platform
    import psutil
    
    try:
        # 导入GPU信息库
        import torch
        logging.info(f"当前平台: {platform.platform()}")
        logging.info(f"CPU核心数: {psutil.cpu_count(logical=True)}")
        logging.info(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        logging.info(f"PyTorch版本: {torch.__version__}")
        logging.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logging.info("无法导入torch或psutil库，跳过系统信息打印")
    
    args, _ = parse_arguments()
    
    # 打印主要参数
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"分数阈值: {args.score_threshold}")
    logging.info(f"Token预算: {args.token_budget}")
    logging.info(f"评分方法: {args.score_method}")
    logging.info(f"GPU并行数: {args.tensor_parallel_size}")
    logging.info(f"每个问题重复次数: {args.num_repeats}")
    
    # 加载数据集
    args.dataset = get_dataset(args.dataset_name)
    
    # 解析问题ID范围
    problem_ids = parse_problem_range(args.problem_id)
    logging.info(f"将处理问题ID: {problem_ids}")
    
    # 初始化模型
    global models
    logging.info(f"使用 {args.tensor_parallel_size} 个GPU进行模型并行")
    models = asyncio.run(initialize_models_async(args.tensor_parallel_size))
    
    # 准备输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每个问题ID，并对每个问题执行指定次数的重复
    total_tasks = len(problem_ids) * args.num_repeats
    completed_tasks = 0
    
    for problem_id in problem_ids:
        for repeat_idx in range(args.num_repeats):

            # 更新当前任务的参数
            args.problem_id = problem_id
            
            logging.info(f"开始处理问题 [{completed_tasks+1}/{total_tasks}]: ID={problem_id}, 重复={repeat_idx}")
            
            # 检查是否已经处理过该组合
            output_filename = os.path.join(output_dir, f"{args.dataset_name}_{problem_id}_{repeat_idx}")
            if os.path.exists(f"{output_filename}.pickle"):
                logging.info(f"问题 {problem_id} (重复 {repeat_idx}) 已存在结果文件，跳过")
                completed_tasks += 1
                continue
            
            # 获取问题数据
            problem, options = prepare_problem_data(args)
            
            # 运行推理过程
            start_time = time.time()
            metadata_with_stats = asyncio.run(run_reasoning_process_async(args, problem, options))
            elapsed_time = time.time() - start_time
            
            if metadata_with_stats:
                # 保存结果
                save_results(metadata_with_stats, output_filename)
                
                # 打印统计信息
                logging.info(f"问题 {problem_id} (重复 {repeat_idx}) 已完成")
                logging.info(f"总耗时: {elapsed_time:.2f}秒")
                logging.info(f"平均改写率: {metadata_with_stats.get('avg_rewrite_rate', 0):.2f}%")
                
                # 打印每个角色的统计信息
                for role, result in metadata_with_stats.get("final_results", {}).items():
                    if "num_steps" in result:
                        logging.info(f"角色 {role}: {result['num_steps']} 步, {result['num_tokens']} tokens")
            else:
                logging.error(f"问题 {problem_id} (重复 {repeat_idx}) 处理失败")
            
            completed_tasks += 1
            progress_percent = completed_tasks / total_tasks * 100
            logging.info(f"总进度: {completed_tasks}/{total_tasks} ({progress_percent:.2f}%)")
            
    logging.info("所有任务已完成")

if __name__ == "__main__":
    main()


