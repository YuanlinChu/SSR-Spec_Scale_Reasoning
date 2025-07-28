#!/bin/bash
# chmod +x run_exp.sh
# ./run_exp.sh

# 设置日志文件
LOG_FILE="experiment_log.txt"

echo "开始执行$(date)" | tee -a $LOG_FILE

python ablation_exp/spec_scale_reason_mcount_abla.py --dataset_name math --problem_id 0-99 --method_num 3 --repeat_id 1 --output_dir results/spec_scale --strategy_pool_size 6

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python ablation_exp/spec_scale_reason_mcount_abla.py --dataset_name math --problem_id 0-99 --method_num 3 --repeat_id 1 --output_dir results/spec_scale --strategy_pool_size 9

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python ablation_exp/spec_scale_reason_mcount_abla.py --dataset_name math --problem_id 0-99 --method_num 3 --repeat_id 1 --output_dir results/spec_scale --strategy_pool_size 15

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "所有实验已完成 $(date)" | tee -a $LOG_FILE