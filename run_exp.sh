#!/bin/bash
# chmod +x run_exp2.sh
# ./run_exp2.sh

# 设置日志文件
LOG_FILE="experiment_log.txt"

echo "开始执行,math500 method_num = 3 $(date)" | tee -a $LOG_FILE

python spec_scale_reason_2.py --dataset_name math --problem_id 0-99 --repeat_id 3 --output_dir results/spec_scale_m3 --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 3

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "math500 method_num = 3  成功完成！" | tee -a $LOG_FILE
else
    echo "math500 method_num = 3  执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

echo "开始执行,math500 method_num = 5 $(date)" | tee -a $LOG_FILE

python spec_scale_reason_2.py --dataset_name math --problem_id 0-99 --repeat_id 3 --output_dir results/spec_scale_m5 --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 5

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "math500 method_num = 5  成功完成！" | tee -a $LOG_FILE
else
    echo "math500 method_num = 5  执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

# 运行GPU占用监控脚本
echo "启动GPU占用监控..." | tee -a $LOG_FILE
python ../../GPUoccupy.py --gpu_ids 2 3 --utilization_threshold 60

echo "所有实验已完成 $(date)" | tee -a $LOG_FILE