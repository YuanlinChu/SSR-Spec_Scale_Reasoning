# 统计策略池中每个方法被选中的次数
# python eval/method_count_eval.py --dataset_name aime --results_dir results/spec_scale_m3_t8.0
import os
import re
import pickle
import argparse
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_method_usage(args):
    """
    统计策略池中每个方法被选中的次数
    """
    logging.info(f"开始统计数据集: {args.dataset_name}")
    logging.info(f"从以下位置加载结果: {args.results_dir}")

    # 检查结果目录是否存在
    model_dir = os.path.join(args.results_dir, args.dataset_name)
    if not os.path.isdir(model_dir):
        logging.error(f"结果目录未找到: {model_dir}")
        return

    # 初始化统计数据
    method_counts = Counter()  # 统计每个方法被选中的次数
    total_files = 0  # 总文件数
    valid_files = 0  # 有效文件数（包含method_code的文件）
    
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
        total_files += 1
        
        # 加载结果文件
        file_path = os.path.join(model_dir, filename)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            # 从final_results中获取所有method_code
            final_results = data.get("final_results", {})
            
            if not final_results:
                logging.warning(f"问题 {problem_id}，重复 {repeat_id}: 没有找到final_results")
                continue
            
            # 遍历final_results中的所有子结果（如A1、B1、J1等）
            method_codes_in_file = []
            for result_key, result_data in final_results.items():
                if isinstance(result_data, dict) and "method_code" in result_data:
                    method_code = result_data["method_code"]
                    method_counts[method_code] += 1
                    method_codes_in_file.append(method_code)
                    logging.debug(f"问题 {problem_id}，重复 {repeat_id}，结果 {result_key}: 使用方法 {method_code}")
            
            if method_codes_in_file:
                valid_files += 1
                logging.debug(f"问题 {problem_id}，重复 {repeat_id}: 找到方法 {method_codes_in_file}")
            else:
                logging.warning(f"问题 {problem_id}，重复 {repeat_id}: 在final_results中没有找到有效的method_code")
                
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    # 打印统计结果
    logging.info("=" * 50)
    logging.info("方法使用统计摘要:")
    logging.info(f"数据集: {args.dataset_name}")
    logging.info(f"结果目录: {model_dir}")
    logging.info(f"总文件数: {total_files}")
    logging.info(f"有效文件数: {valid_files}")
    logging.info(f"无效文件数: {total_files - valid_files}")
    logging.info("=" * 50)
    
    if valid_files == 0:
        logging.warning("没有找到有效的结果文件")
        return
    
    # 按使用次数从高到低排序并打印详细统计
    logging.info("各方法使用次数统计 (按使用次数排序):")
    for method_code, count in method_counts.most_common():
        percentage = (count / valid_files) * 100
        logging.info(f"方法 {method_code}: {count} 次 ({percentage:.2f}%)")
    
    logging.info("=" * 50)
    
    # 打印最常用和最少用的方法
    if method_counts:
        most_common = method_counts.most_common(1)[0]
        least_common = method_counts.most_common()[-1]
        
        logging.info(f"最常用方法: {most_common[0]} ({most_common[1]} 次, {most_common[1]/valid_files*100:.2f}%)")
        logging.info(f"最少用方法: {least_common[0]} ({least_common[1]} 次, {least_common[1]/valid_files*100:.2f}%)")
        
        # 计算方法分布的均匀性
        num_methods = len(method_counts)
        expected_count = valid_files / num_methods
        variance = sum((count - expected_count) ** 2 for count in method_counts.values()) / num_methods
        std_dev = variance ** 0.5
        
        logging.info(f"方法总数: {num_methods}")
        logging.info(f"期望使用次数 (均匀分布): {expected_count:.2f}")
        logging.info(f"实际分布标准差: {std_dev:.2f}")
    
    logging.info("=" * 50)
    
    return {
        "total_files": total_files,
        "valid_files": valid_files,
        "method_counts": dict(method_counts),
        "method_percentages": {method: (count/valid_files)*100 for method, count in method_counts.items()}
    }

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="统计策略池中每个方法被选中的次数")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "live"], default="aime",
                        help="数据集名称")
    parser.add_argument("--results_dir", type=str, default="results/spec_scale_m3_t4.0",
                        help="结果文件的目录")
    parser.add_argument("--verbose", action="store_true",
                        help="启用详细日志输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 执行统计
    results = count_method_usage(args)
    
    if results:
        print("\n统计完成！")
        print(f"处理了 {results['valid_files']} 个有效文件")
        print("方法使用分布 (按使用次数排序):")
        # 按使用次数从高到低排序
        sorted_methods = sorted(results['method_percentages'].items(), key=lambda x: x[1], reverse=True)
        for method, percentage in sorted_methods:
            print(f"  {method}: {percentage:.2f}%")

if __name__ == "__main__":
    main()