#!/bin/bash
# chmod +x run_exp.sh
# ./run_exp.sh

# 设置日志文件
LOG_FILE="experiment_log.txt"

echo "开始执行$(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale --score_threshold 5.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale --score_threshold 6.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale --score_threshold 8.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale --score_threshold 9.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale --score_threshold 4.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "所有实验已完成 $(date)" | tee -a $LOG_FILE