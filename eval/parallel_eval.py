#python eval/parallel_eval.py --dataset_name live --model_name QwQ-32B --results_dir results/scale_reason_r3_noprompt --plot
#这个代码是评估准确率随着并行数量的变化（用pass@k来代表），分析到了一定数量后再增加就收效不大了
#python eval/parallel_eval.py --dataset_name aime --model_name QwQ-32B --results_dir results/scale_reason --plot
import os
import re
import pickle
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    elif dataset_name in ["math", "live"]:
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
        elif dataset_name in ["math", "live"]:
            # 对于MATH数据集，需要处理复杂的数学表达式
            # 规范化处理：移除所有空格和LaTeX命令
            def normalize_math_expr(expr):
                if not isinstance(expr, str):
                    return str(expr)
                
                # 移除所有空格
                expr = re.sub(r'\s+', '', expr)
                
                # 标准化分数表示: \dfrac 和 \frac 转换为相同形式
                expr = re.sub(r'\\dfrac', r'\\frac', expr)
                
                # 规范化分数表示
                expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'\1/\2', expr)
                
                # 标准化文本表示: 移除 \text
                expr = re.sub(r'\\text\{(.*?)\}', r'\1', expr)
                
                # 标准化平方根表示: 处理不同的 \sqrt 写法
                expr = re.sub(r'\\\\sqrt', r'\\sqrt', expr)
                
                # 移除度数符号
                expr = re.sub(r'(\d+)\^\\circ', r'\1', expr)
                
                # 移除LaTeX命令如\left和\right
                expr = re.sub(r'\\left|\\\right', '', expr)
                
                # 标准化括号
                expr = re.sub(r'\\left\(', '(', expr)
                expr = re.sub(r'\\right\)', ')', expr)
                
                # 标准化矩阵表示
                expr = re.sub(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', lambda m: m.group(1).replace('\\\\', ','), expr)
                
                # 标准化分数表示中的除号
                expr = re.sub(r'(\d+)/(\d+)', lambda m: f"{m.group(1)}/{m.group(2)}", expr)
                
                # 处理 \pm 符号（正负号）
                expr = re.sub(r'\\pm', 'pm', expr)
                
                # 处理 x= 这样的前缀
                expr = re.sub(r'^[a-z]=', '', expr)

                # 标准化平方根参数: \sqrt{x} 和 \sqrt x 转换为相同形式
                expr = re.sub(r'\\sqrt\{(.*?)\}', r'\\sqrt\1', expr)
                
                # 特别为LiveMathBench添加：移除美元符号
                expr = re.sub(r'^\$(.*)\$$', r'\1', expr)
                expr = re.sub(r'^\$(.*)\$', r'\1', expr)
                
                return expr
            
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

def _pass_at_k(n, c, k):
    """
    计算Pass@k的概率
    :param n: 总尝试次数
    :param c: 正确尝试的次数
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

def calculate_pass_at_k(scores, max_k=9):
    """
    计算不同k值(1到max_k)的Pass@k
    :param scores: 所有问题的评分列表（True/False）
    :param max_k: 最大的k值
    :return: 包含不同k值的Pass@k结果的字典
    """
    results = {}
    problem_count = len(scores)
    if problem_count == 0:
        return results
    
    n = max(len(s) for s in scores.values())  # 每个问题的尝试次数
    
    # 确保所有问题都有相同数量的尝试
    normalized_scores = {}
    for problem_id, problem_scores in scores.items():
        # 如果某个问题的尝试次数少于n，用False填充
        while len(problem_scores) < n:
            problem_scores.append(False)
        normalized_scores[problem_id] = problem_scores
    
    # 计算每个k值的Pass@k
    for k in range(1, min(max_k+1, n+1)):
        pass_k_values = []
        for problem_scores in normalized_scores.values():
            num_correct = sum(problem_scores)
            pass_k = _pass_at_k(n, num_correct, k)
            pass_k_values.append(pass_k)
        
        # 所有问题的平均Pass@k
        results[k] = np.mean(pass_k_values) * 100  # 转为百分比
    
    return results

def plot_pass_at_k(pass_k_results, output_path=None):
    """
    绘制Pass@k的折线图
    :param pass_k_results: 包含k值和对应Pass@k的字典
    :param output_path: 图表保存路径，若为None则显示图表
    """
    plt.figure(figsize=(10, 6))
    
    k_values = list(pass_k_results.keys())
    pass_k_values = list(pass_k_results.values())
    
    plt.plot(k_values, pass_k_values, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.title('Parallel Evaluation')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    
    # 在每个点上标注具体数值
    for i, value in enumerate(pass_k_values):
        plt.annotate(f"{value:.2f}%", (k_values[i], pass_k_values[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logging.info(f"折线图已保存至: {output_path}")
    else:
        plt.show()

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

    model_dir = os.path.join(args.results_dir, args.model_name, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化统计数据
    problem_stats = {}  # 每个问题的统计信息
    total_correct = 0   # 所有正确的次数
    total_attempts = 0  # 所有尝试的次数
    problems_with_correct = 0  # 至少有一次正确的问题数
    problem_to_scores = defaultdict(list)  # 每个问题的正确/错误列表
    
    # 初始化reasoning答案的统计数据
    reasoning_problem_stats = {}  # 每个问题的reasoning答案统计信息
    reasoning_total_correct = 0   # reasoning答案所有正确的次数
    reasoning_total_attempts = 0  # reasoning答案所有尝试的次数
    reasoning_problems_with_correct = 0  # reasoning答案至少有一次正确的问题数
    reasoning_problem_to_scores = defaultdict(list)  # 每个问题的reasoning答案正确/错误列表

    # 遍历结果文件
    for filename in os.listdir(model_dir):
        # 只处理pickle文件，跳过txt文件
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
            
        # 获取正确答案
        ground_truth = get_ground_truth(dataset, problem_idx, args.dataset_name)
        if ground_truth is None:
            logging.warning(f"问题ID {problem_id}（索引 {problem_idx}）的正确答案获取失败。跳过。")
            continue
            
        # 加载结果文件
        file_path = os.path.join(model_dir, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                    
            # 提取数据中的运行结果
            method_results = data.get("time_stats", {}).get("method_results", [])
            if not method_results:
                logging.warning(f"在文件 {file_path} 中未找到运行结果")
                continue
                
            # 处理reasoning字段中的答案（每个文件只处理一次）
            reasoning_data = data.get("reasoning")  # 从data顶层获取reasoning
            reasoning_answer = None
            reasoning_is_correct = False
            
            if reasoning_data and isinstance(reasoning_data, dict):
                reasoning_answer = reasoning_data.get("answer")
                if reasoning_answer is not None:
                    reasoning_is_correct = compare_answers(reasoning_answer, ground_truth, args.dataset_name)
            
            # 记录reasoning答案的详细结果
            if reasoning_answer is not None and not reasoning_is_correct:
                logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id} [Reasoning]: 模型='{reasoning_answer}'，正确答案='{ground_truth}'，判断={reasoning_is_correct}")
            
            # 更新reasoning答案的统计信息
            if reasoning_answer is not None:
                if problem_id not in reasoning_problem_stats:
                    reasoning_problem_stats[problem_id] = {"attempts": 0, "correct": 0}
                    
                reasoning_problem_stats[problem_id]["attempts"] += 1
                if reasoning_is_correct:
                    reasoning_problem_stats[problem_id]["correct"] += 1
                    reasoning_total_correct += 1
                    
                reasoning_total_attempts += 1
                
                # 为reasoning答案的pass@k计算添加结果
                reasoning_problem_to_scores[problem_id].append(reasoning_is_correct)
            
            # 处理每个运行结果
            for run_id, result in enumerate(method_results):
                model_answer = result.get("answer")
                
                # 如果model_answer不存在，直接判定为错误
                if model_answer is None:
                    is_correct = False
                else:
                    is_correct = compare_answers(model_answer, ground_truth, args.dataset_name)
                
                # 记录详细结果
                if not is_correct:
                    stop_reason = result.get("stop_reason", "未知")
                    logging.info(f"问题 {problem_id}（索引 {problem_idx}），重复 {repeat_id}，运行 {run_id}: 模型='{model_answer}'，正确答案='{ground_truth}'，判断={is_correct}，停止原因={stop_reason}")
                
                # 更新问题统计信息
                if problem_id not in problem_stats:
                    problem_stats[problem_id] = {"attempts": 0, "correct": 0}
                    
                problem_stats[problem_id]["attempts"] += 1
                if is_correct:
                    problem_stats[problem_id]["correct"] += 1
                    total_correct += 1
                    
                total_attempts += 1
                
                # 为pass@k计算添加结果
                problem_to_scores[problem_id].append(is_correct)
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 计算每个问题是否至少有一次正确
    for problem_id, stats in problem_stats.items():
        if stats["correct"] > 0:
            problems_with_correct += 1
            
    # 计算reasoning答案每个问题是否至少有一次正确
    for problem_id, stats in reasoning_problem_stats.items():
        if stats["correct"] > 0:
            reasoning_problems_with_correct += 1

    # 计算准确率
    total_problems = len(problem_stats)
    if total_problems > 0:
        accuracy_method1 = (problems_with_correct / total_problems) * 100
    else:
        accuracy_method1 = 0
        
    if total_attempts > 0:
        accuracy_method2 = (total_correct / total_attempts) * 100
    else:
        accuracy_method2 = 0
        
    # 计算reasoning答案的准确率
    reasoning_total_problems = len(reasoning_problem_stats)
    if reasoning_total_problems > 0:
        reasoning_accuracy_method1 = (reasoning_problems_with_correct / reasoning_total_problems) * 100
    else:
        reasoning_accuracy_method1 = 0
        
    if reasoning_total_attempts > 0:
        reasoning_accuracy_method2 = (reasoning_total_correct / reasoning_total_attempts) * 100
    else:
        reasoning_accuracy_method2 = 0

    # 打印结果
    logging.info("=" * 50)
    logging.info("评估摘要:")
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"模型: {args.model_name}")
    logging.info(f"结果目录: {model_dir}")
    logging.info("")
    logging.info("原始答案统计:")
    logging.info(f"总问题数: {total_problems}")
    logging.info(f"总尝试次数: {total_attempts}")
    logging.info(f"总正确次数: {total_correct}")
    logging.info(f"至少有一次正确的问题数: {problems_with_correct}")
    logging.info(f"(至少一次正确的问题/总问题数): {accuracy_method1:.2f}%")
    logging.info(f"(总正确次数/总尝试次数): {accuracy_method2:.2f}%")
    logging.info("")
    logging.info("Reasoning答案统计:")
    logging.info(f"总问题数: {reasoning_total_problems}")
    logging.info(f"总尝试次数: {reasoning_total_attempts}")
    logging.info(f"总正确次数: {reasoning_total_correct}")
    logging.info(f"至少有一次正确的问题数: {reasoning_problems_with_correct}")
    logging.info(f"(至少一次正确的问题/总问题数): {reasoning_accuracy_method1:.2f}%")
    logging.info(f"(总正确次数/总尝试次数): {reasoning_accuracy_method2:.2f}%")
    logging.info("=" * 50)
    
    # 打印每个问题的详细统计信息
    logging.info("每个问题的原始答案详细统计信息:")
    for problem_id in sorted(problem_stats.keys()):
        stats = problem_stats[problem_id]
        logging.info(f"问题 {problem_id}: 尝试={stats['attempts']}, 正确={stats['correct']}, 正确率={stats['correct']/stats['attempts']*100:.2f}%")
    
    # 打印reasoning答案的详细统计信息
    if reasoning_problem_stats:
        logging.info("")
        logging.info("每个问题的Reasoning答案详细统计信息:")
        for problem_id in sorted(reasoning_problem_stats.keys()):
            stats = reasoning_problem_stats[problem_id]
            logging.info(f"问题 {problem_id}: 尝试={stats['attempts']}, 正确={stats['correct']}, 正确率={stats['correct']/stats['attempts']*100:.2f}%")
    
    # 计算pass@k
    pass_k_results = calculate_pass_at_k(problem_to_scores, max_k=9)
    
    # 计算reasoning答案的pass@k
    reasoning_pass_k_results = {}
    if reasoning_problem_to_scores:
        reasoning_pass_k_results = calculate_pass_at_k(reasoning_problem_to_scores, max_k=9)
    
    # 打印pass@k结果
    logging.info("=" * 50)
    logging.info("原始答案 Pass@k 结果:")
    for k, value in pass_k_results.items():
        logging.info(f"Pass@{k}: {value:.2f}%")
    
    if reasoning_pass_k_results:
        logging.info("")
        logging.info("Reasoning答案 Pass@k 结果:")
        for k, value in reasoning_pass_k_results.items():
            logging.info(f"Pass@{k}: {value:.2f}%")
    logging.info("=" * 50)
    
    # 绘制pass@k折线图
    if args.plot:
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{args.model_name}_{args.dataset_name}_pass_at_k.png")
        plot_pass_at_k(pass_k_results, output_path=plot_path)
    
    return {
        "total_problems": total_problems,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "problems_with_correct": problems_with_correct,
        "accuracy_method1": accuracy_method1,
        "accuracy_method2": accuracy_method2,
        "problem_stats": problem_stats,
        "pass_at_k": pass_k_results,
        "reasoning_total_problems": reasoning_total_problems,
        "reasoning_total_attempts": reasoning_total_attempts,
        "reasoning_total_correct": reasoning_total_correct,
        "reasoning_problems_with_correct": reasoning_problems_with_correct,
        "reasoning_accuracy_method1": reasoning_accuracy_method1,
        "reasoning_accuracy_method2": reasoning_accuracy_method2,
        "reasoning_problem_stats": reasoning_problem_stats,
        "reasoning_pass_at_k": reasoning_pass_k_results
    }

def main():
    parser = argparse.ArgumentParser(description="评估模型在数据集上的表现")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], required=True,
                        help="要评估的数据集")
    parser.add_argument("--model_name", type=str, required=True,
                        help="要评估的模型名称")
    parser.add_argument("--results_dir", type=str, default="results/scale_reason_norole",
                        help="包含结果文件的目录")
    parser.add_argument("--plot", action="store_true",
                        help="是否生成pass@k的折线图")
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()