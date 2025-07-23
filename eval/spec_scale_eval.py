# python spec_scale_eval.py --dataset_name aime --model_name QwQ-32B --results_dir results/spec_scale_m

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import argparse
import logging
import numpy as np
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from metric import pass_at_k
import sympy
from sympy.parsing.latex import parse_latex
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def get_ground_truth(dataset, problem_idx, dataset_name):
#     """
#     从数据集中获取正确答案
#     """
#     if dataset_name == "aime":
#         try:
#             # 根据观察到的数据集结构，直接使用 answer 字段
#             answer_str = dataset[problem_idx].get("answer", "")
#             try:
#                 # 尝试直接转换为整数
#                 return int(answer_str)
#             except ValueError:
#                 # 如果直接转换失败，尝试使用正则表达式提取数字
#                 match = re.search(r"(\d+)$", answer_str)
#                 if match:
#                     return int(match.group(1))
#                 logging.error(f"无法从AIME索引{problem_idx}的答案中提取数字: {answer_str}")
#                 return None
#         except (ValueError, TypeError, KeyError) as e:
#             logging.error(f"获取AIME索引{problem_idx}的正确答案时出错: {e}. 答案字符串: {dataset[problem_idx].get('answer', 'N/A')}")
#             return None
#     elif dataset_name == "math":
#         try:
#             solution_str = dataset[problem_idx].get("solution", "")
#             match = re.search(r"\\boxed\{(.+?)\}", solution_str)
#             if match:
#                 answer = match.group(1).strip()
#                 try:
#                     # 首先尝试转换为整数
#                     return int(answer)
#                 except ValueError:
#                     # 处理分数或其他格式
#                     logging.warning(f"MATH正确答案不是简单整数: {answer}")
#                     return answer  # 如果不是整数，则返回字符串
#             else:
#                 logging.error(f"无法从MATH索引{problem_idx}的解决方案中提取正确答案: {solution_str}")
#                 return None
#         except (KeyError, TypeError) as e:
#             logging.error(f"访问MATH索引{problem_idx}的解决方案时出错: {e}")
#             return None
#     elif dataset_name == "gpqa":
#         try:
#             # 找到与正确答案文本对应的字母
#             correct_answer_text = dataset[problem_idx].get("Correct Answer")
#             options_text = {
#                 "A": dataset[problem_idx].get("Answer A"),
#                 "B": dataset[problem_idx].get("Answer B"),
#                 "C": dataset[problem_idx].get("Answer C"),
#                 "D": dataset[problem_idx].get("Answer D"),
#             }
#             for letter, text in options_text.items():
#                 if text == correct_answer_text:
#                     return letter
#             logging.error(f"无法为GPQA索引{problem_idx}找到匹配的正确答案字母")
#             return None
#         except (KeyError, TypeError) as e:
#             logging.error(f"获取GPQA索引{problem_idx}的正确答案时出错: {e}")
#             return None
#     else:
#         raise ValueError(f"未知的数据集名称: {dataset_name}")

def get_ground_truth(dataset, problem_idx, dataset_name):
    """
    从数据集中获取正确答案
    """
    if dataset_name == "aime":
        try:
            answer_str = dataset[problem_idx].get("answer", "")
            try:
                # 尝试直接转换为整数
                return int(answer_str)
            except ValueError:
                # 如果直接转换失败，尝试使用正则表达式提取数字
                match = re.search(r"(\d+)$", answer_str)
                if match:
                    return int(match.group(1))
                logging.error(f"无法从AIME索引{problem_idx}的答案中提取数字: {answer_str}")
                return None
        except (ValueError, TypeError, KeyError) as e:
            logging.error(f"获取AIME索引{problem_idx}的正确答案时出错: {e}. 答案字符串: {dataset[problem_idx].get('answer', 'N/A')}")
            return None
    elif dataset_name == "live":
        try:
            answer_str = dataset[problem_idx].get("answer", "")
            try:
                # 尝试直接转换为整数
                return int(answer_str)
            except ValueError:
                # 如果不是整数，返回原始答案字符串
                return answer_str
        except (KeyError, TypeError) as e:
            logging.error(f"获取LiveMathBench索引{problem_idx}的答案时出错: {e}")
            return None
    elif dataset_name == "math":
        try:
            answer_str = dataset[problem_idx].get("answer", "")
            try:
                # 首先尝试转换为整数
                return int(answer_str)
            except ValueError:
                # 处理分数或其他格式
                # logging.warning(f"MATH正确答案不是简单整数: {answer}")
                return answer_str  # 如果不是整数，则返回字符串
        except (KeyError, TypeError) as e:
            logging.error(f"访问MATH索引{problem_idx}的解决方案时出错: {e}")
            return None
    elif dataset_name == "gpqa":
        try:
            # 找到与正确答案文本对应的字母
            correct_answer_text = dataset[problem_idx].get("Correct Answer")
            options_text = {
                "A": dataset[problem_idx].get("Answer A"),
                "B": dataset[problem_idx].get("Answer B"),
                "C": dataset[problem_idx].get("Answer C"),
                "D": dataset[problem_idx].get("Answer D"),
            }
            for letter, text in options_text.items():
                if text == correct_answer_text:
                    return letter
            logging.error(f"无法为GPQA索引{problem_idx}找到匹配的正确答案字母")
            return None
        except (KeyError, TypeError) as e:
            logging.error(f"获取GPQA索引{problem_idx}的正确答案时出错: {e}")
            return None
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

# def compare_answers(model_answer, ground_truth, dataset_name):
#     """
#     比较模型答案与正确答案
#     """
#     if model_answer is None or ground_truth is None:
#         return False
        
#     # 规范化答案进行比较（例如，转换为相同类型）
#     try:
#         if dataset_name in ["aime", "math"]:
#             # 尝试数值比较
#             return int(model_answer) == int(ground_truth)
#         elif dataset_name == "gpqa":
#             # 简单的字符串比较A、B、C、D
#             return str(model_answer).upper() == str(ground_truth).upper()
#         else:
#             return str(model_answer) == str(ground_truth)
#     except (ValueError, TypeError):
#         # 如果数值转换失败，则回退到字符串比较
#         logging.warning(f"{dataset_name}的数值比较失败。以字符串形式比较: '{model_answer}' vs '{ground_truth}'")
#         return str(model_answer).strip() == str(ground_truth).strip()
            
# 规范化处理：移除所有空格和LaTeX命令
def normalize_math_expr(expr):
    if not isinstance(expr, str):
        return str(expr)
    
    # 移除美元符号
    expr = re.sub(r'^\$(.*)\$$', r'\1', expr)
    expr = re.sub(r'^\$(.*)\$', r'\1', expr)
    
    # 移除千位分隔符（逗号）
    expr = re.sub(r'(\d),(\d)', r'\1\2', expr)
    
    # 移除所有空格
    expr = re.sub(r'\s+', '', expr)
    
    # 标准化分数表示: \dfrac 和 \frac 转换为相同形式
    expr = re.sub(r'\\dfrac', r'\\frac', expr)
    
    # 标准化文本表示: 移除 \text
    expr = re.sub(r'\\text\{(.*?)\}', r'\1', expr)
    
    # 标准化平方根表示: 处理不同的 \sqrt 写法
    expr = re.sub(r'\\\\sqrt', r'\\sqrt', expr)
    
    # 确保 \sqrt 后有花括号 - 修复 \sqrt2 变成 \sqrt{2}
    expr = re.sub(r'\\sqrt(\d+)', r'\\sqrt{\1}', expr)
    
    # 移除度数符号
    expr = re.sub(r'(\d+)\^\\circ', r'\1', expr)
    
    # 移除LaTeX命令如\left和\right
    expr = re.sub(r'\\left|\\\right', '', expr)
    
    # 标准化括号
    expr = re.sub(r'\\left\(', '(', expr)
    expr = re.sub(r'\\right\)', ')', expr)
    
    # 标准化矩阵表示
    expr = re.sub(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', lambda m: m.group(1).replace('\\\\', ','), expr)
    
    # 确保分数表示正确 (如 1/2 的形式)
    # 这步不要转换，保留LaTeX格式更容易被sympy正确解析
    # expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'\1/\2', expr)
    
    # 处理 \pm 符号（正负号）
    expr = re.sub(r'\\pm', r'\\pm ', expr)
    
    # 处理 x= 这样的前缀
    expr = re.sub(r'^[a-z]=', '', expr)
    
    return expr
                
def compare_answers(model_answer, ground_truth, dataset_name):
    """
    比较模型答案与正确答案
    """
    if model_answer is None or ground_truth is None:
        return False
        
    # 规范化答案进行比较（例如，转换为相同类型）
    try:
        if dataset_name in ["aime"]:
            # 尝试数值比较
            return int(model_answer) == int(ground_truth)
        elif dataset_name in ["live"]:
            # 首先尝试规范化后的直接字符串比较
            norm_model = normalize_math_expr(model_answer)
            norm_truth = normalize_math_expr(ground_truth)
            
            # 对于纯数字，提取并比较
            try:
                # 使用正则表达式提取数字部分
                model_digits = re.sub(r'[^\d]', '', norm_model)
                truth_digits = re.sub(r'[^\d]', '', norm_truth)
                
                if model_digits and truth_digits:
                    if int(model_digits) == int(truth_digits):
                        return True
            except:
                pass
            
            # 如果规范化后完全相同
            if norm_model == norm_truth:
                return True
            
            # 尝试使用更复杂的数学表达式比较
            if compare_math_expressions(norm_model, norm_truth):
                return True
            
            # 特别处理根式表达式
            # 对于您提供的例子 3\sqrt{15} 和 15\sqrt{7}
            sqrt_pattern = r'(\d*(?:\.\d+)?)\\sqrt\{(\d+(?:\.\d+)?)\}'
            
            model_sqrt_match = re.match(sqrt_pattern, norm_model)
            truth_sqrt_match = re.match(sqrt_pattern, norm_truth)
            
            if model_sqrt_match and truth_sqrt_match:
                try:
                    # 提取系数和根号内的数字
                    model_coef = float(model_sqrt_match.group(1) or 1)
                    model_rad = float(model_sqrt_match.group(2))
                    
                    truth_coef = float(truth_sqrt_match.group(1) or 1)
                    truth_rad = float(truth_sqrt_match.group(2))
                    
                    # 计算数值
                    model_value = model_coef * math.sqrt(model_rad)
                    truth_value = truth_coef * math.sqrt(truth_rad)
                    
                    # 比较数值
                    return abs(model_value - truth_value) < 1e-9
                except:
                    pass
            
            # 如果所有比较方法都失败，返回False
            logging.warning(f"无法匹配的数学表达式: '{model_answer}' vs '{ground_truth}'")
            logging.warning(f"规范化后: '{norm_model}' vs '{norm_truth}'")
            
            return False
            
        elif dataset_name in ["math"]:
            # 对于MATH数据集，需要处理复杂的数学表达式
            
            norm_model = normalize_math_expr(model_answer)
            norm_truth = normalize_math_expr(ground_truth)
            
            # 如果规范化后相等，则认为答案正确
            if norm_model == norm_truth:
                return True
            
            # 处理多个答案的情况（如 "1,-2" 或 "3, 5, 7"）
            # if ',' in norm_truth:
            #     truth_parts = [p.strip() for p in norm_truth.split(',')]
            #     # 检查模型答案是否是正确答案中的一个
            #     if norm_model in truth_parts:
            #         return True
                
            # 尝试数值比较（如果可能）
            try:
                # 对于简单数值，尝试直接比较
                return float(norm_model) == float(norm_truth)
            except:
                # 如果无法转换为数值，则使用字符串比较
                return norm_model == norm_truth
        elif dataset_name == "gpqa":
            # 简单的字符串比较A、B、C、D
            return str(model_answer).upper() == str(ground_truth).upper()
        else:
            return str(model_answer) == str(ground_truth)
    except (ValueError, TypeError):
        # 如果数值转换失败，则回退到字符串比较
        # logging.warning(f"{dataset_name}的数值比较失败。以字符串形式比较: '{model_answer}' vs '{ground_truth}'")
        return str(model_answer).strip() == str(ground_truth).strip()

def compare_math_expressions(expr1, expr2):
    """比较两个数学表达式是否等价，先尝试符号比较，再尝试数值比较"""
    try:
        # 1. 尝试使用sympy进行符号比较
        expr1_sympy = parse_latex(expr1)
        expr2_sympy = parse_latex(expr2)
        
        # 符号差异
        diff = sympy.simplify(expr1_sympy - expr2_sympy)
        if diff == 0:
            return True
            
        # 2. 对于除法表达式，尝试比较比值
        try:
            ratio = sympy.simplify(expr1_sympy / expr2_sympy)
            if ratio == 1:
                return True
        except:
            pass
            
        # 3. 尝试数值比较
        try:
            val1 = float(sympy.N(expr1_sympy))
            val2 = float(sympy.N(expr2_sympy))
            
            # 对于大数值，使用相对误差
            if max(abs(val1), abs(val2)) > 1.0:
                relative_error = abs(val1 - val2) / max(abs(val1), abs(val2))
                return relative_error < 1e-9
            else:
                # 对于小数值，使用绝对误差
                return abs(val1 - val2) < 1e-9
        except:
            pass
            
        return False
    except Exception as e:
        # 如果符号比较失败，记录错误但不抛出异常
        return False

def evaluate_model(args):
    """评估模型在数据集上的表现"""
    logging.info(f"开始评估数据集: {args.dataset_name}")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # --- 加载数据集 ---
    try:
        if args.dataset_name == "aime":
            dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
            id_offset = 60  # AIME问题ID从60开始
        elif args.dataset_name == "math":
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
            id_offset = 0
        elif args.dataset_name == "live":
            dataset = load_dataset("opencompass/LiveMathBench", "v202412_AMC_en")["test"]
            id_offset = 0
        elif args.dataset_name == "gpqa":
            if os.getenv("HF_HUB_OFFLINE", "0") == "1":
                dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
            else:    
                dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
            id_offset = 0
        else:
            raise ValueError(f"不支持的数据集: {args.dataset_name}")
        logging.info(f"数据集加载成功。总样本数: {len(dataset)}")
    except Exception as e:
        logging.error(f"加载数据集'{args.dataset_name}'失败: {e}")
        return

    # 检查结果目录是否存在
    model_dir = os.path.join(args.results_dir, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化统计数据
    problem_stats = {}  # 每个问题的统计信息
    problem_to_scores = defaultdict(list)  # 每个问题的正确/错误列表
    
    # 遍历结果文件
    for filename in os.listdir(model_dir):
        if not filename.endswith(".pickle"):
            continue
            
        # 解析文件名获取问题ID和重复ID
        match = re.match(r"(\d+)-(\d+)\.pickle", filename)
        if not match:
            logging.warning(f"跳过无法解析的文件名: {filename}")
            continue
            
        problem_id = int(match.group(1))
        repeat_id = int(match.group(2))
        
        # 计算数据集中的索引
        problem_idx = problem_id - id_offset
        if problem_idx < 0 or problem_idx >= len(dataset):
            logging.warning(f"问题ID {problem_id}（索引 {problem_idx}）超出数据集大小 {len(dataset)}。跳过。")
            continue
            
        # 加载结果文件
        file_path = os.path.join(model_dir, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            # 从best_result中直接获取答案，这是与eval.py的主要区别
            best_result = data.get("best_result", {})
            
            # 检查是否有答案
            if best_result is None or best_result.get("answer") is None:
                is_correct = False
                logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}: 没有找到答案，可能是由于达到token预算上限")
            else:
                # 直接从best_result中获取答案
                model_answer = best_result.get("answer")
                ground_truth = get_ground_truth(dataset, problem_idx, args.dataset_name)
                
                # # 检查答案是否正确
                is_correct = compare_answers(model_answer, ground_truth, args.dataset_name)
                # logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}: 模型='{model_answer}'，正确答案='{ground_truth}'，正确={is_correct}")
            
                # 记录详细结果
                if is_correct:
                    pass
                    # logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}: 模型='{model_answer}'，正确答案='{ground_truth}'，判断={is_correct}")
                else:
                    logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}: 模型='{model_answer}'，正确答案='{ground_truth}'，判断={is_correct}")
               

            # 更新问题统计信息
            if problem_id not in problem_stats:
                problem_stats[problem_id] = {"attempts": 0, "correct": 0}
                
            problem_stats[problem_id]["attempts"] += 1
            if is_correct:
                problem_stats[problem_id]["correct"] += 1
                
            # 为pass@k计算添加结果
            problem_to_scores[problem_id].append(is_correct)
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 计算基本统计信息
    total_problems = len(problem_stats)
    total_attempts = sum(stats["attempts"] for stats in problem_stats.values())
    total_correct = sum(stats["correct"] for stats in problem_stats.values())
    problems_with_correct = sum(1 for stats in problem_stats.values() if stats["correct"] > 0)

    # 计算准确率
    if total_problems > 0:
        accuracy_method1 = (problems_with_correct / total_problems) * 100
    else:
        accuracy_method1 = 0
        
    if total_attempts > 0:
        accuracy_method2 = (total_correct / total_attempts) * 100
    else:
        accuracy_method2 = 0

    # 打印基本结果
    logging.info("=" * 50)
    logging.info("评估摘要:")
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"模型: {args.model_name}")
    logging.info(f"结果目录: {model_dir}")
    logging.info(f"总问题数: {total_problems}")
    logging.info(f"总尝试次数: {total_attempts}")
    logging.info(f"总正确次数: {total_correct}")
    logging.info(f"至少有一次正确的问题数: {problems_with_correct}")
    logging.info(f"方法1准确率 (至少一次正确的问题/总问题数): {accuracy_method1:.2f}%")
    logging.info(f"方法2准确率 (总正确次数/总尝试次数): {accuracy_method2:.2f}%")
    logging.info("=" * 50)
    
    # 打印每个问题的详细统计信息
    logging.info("每个问题的详细统计信息:")
    for problem_id in sorted(problem_stats.keys()):
        stats = problem_stats[problem_id]
        logging.info(f"问题 {problem_id}: 尝试={stats['attempts']}, 正确={stats['correct']}, 正确率={stats['correct']/stats['attempts']*100:.2f}%")
    
    # 计算pass@k
    # 准备pass@k的输入数据
    N = max(len(scores) for scores in problem_to_scores.values())
    temp_to_scores = {"default": {}}
    
    # 确保每个问题都有相同数量的尝试
    for problem_id, scores in problem_to_scores.items():
        # 如果某个问题的尝试次数少于N，用False填充
        while len(scores) < N:
            scores.append(False)
        temp_to_scores["default"][problem_id] = scores
    
    # 计算pass@k
    pass_k_results = pass_at_k(N, temp_to_scores)
    
    # 打印pass@k结果
    logging.info("=" * 50)
    logging.info("Pass@k 结果:")
    for temp, results in pass_k_results.items():
        logging.info(f"温度: {temp}")
        for k, value in results.items():
            logging.info(f"  {k}: {value}%")
    logging.info("=" * 50)
    
    return {
        "total_problems": total_problems,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "problems_with_correct": problems_with_correct,
        "accuracy_method1": accuracy_method1,
        "accuracy_method2": accuracy_method2,
        "problem_stats": problem_stats,
        "pass_at_k": pass_k_results
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估模型在数据集上的表现")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], default="aime",
                        help="数据集名称")
    parser.add_argument("--model_name", type=str, default="QwQ-32B",
                        help="base模型名词")
    parser.add_argument("--results_dir", type=str, default="results/spec_scale_Inf",
                        help="结果文件的目录")
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()