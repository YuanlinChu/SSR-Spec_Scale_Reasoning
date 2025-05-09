#!/bin/bash

# 设置日志文件
LOG_FILE="experiment_log.txt"

echo "开始执行实验 $(date)" | tee -a $LOG_FILE

python spec_scale_reason.py --dataset_name math --problem_id 11-99 --repeat_id 3 --output_dir results/spec_scale_Inf --score_threshold 7.0 --token_budget 8192 --score_method greedy
# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "实验成功完成！" | tee -a $LOG_FILE
else
    echo "实验执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

# 运行GPU占用监控脚本
echo "启动GPU占用监控..." | tee -a $LOG_FILE
python ../../GPUoccupy.py --gpu_ids 2 3 --utilization_threshold 60

echo "所有实验已完成 $(date)" | tee -a $LOG_FILE