#!/bin/bash
# chmod +x run_exp.sh
# ./run_exp.sh

# 设置日志文件
LOG_FILE="experiment_log.txt"

echo "开始执行$(date)" | tee -a $LOG_FILE

python scale_reason.py --dataset_name math --problem_id 1-99 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/scale_reason_r3_noprompt

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行$(date)" | tee -a $LOG_FILE

python scale_reason.py --dataset_name math --problem_id 0-99 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/scale_reason_r3_prompt --use_role_prompts

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "成功完成！" | tee -a $LOG_FILE
else
    echo "执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

# 运行GPU占用监控脚本
# echo "启动GPU占用监控..." | tee -a $LOG_FILE
# python GPUoccupy.py --gpu_ids 4 5 --utilization_threshold 80

# echo "所有实验已完成 $(date)" | tee -a $LOG_FILE