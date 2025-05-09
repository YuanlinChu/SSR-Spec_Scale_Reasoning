#!/bin/bash
# chmod +x run_experiments.sh
# ./run_experiments.sh
# 设置日志文件
LOG_FILE="experiment_log.txt"

echo "开始执行实验 $(date)" | tee -a $LOG_FILE

# 第一个命令：使用角色提示
echo "开始执行带角色提示的实验..." | tee -a $LOG_FILE
python scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/scale_reason --use_role_prompts

# 检查第一个命令是否成功
if [ $? -eq 0 ]; then
    echo "带角色提示的实验成功完成！" | tee -a $LOG_FILE
else
    echo "带角色提示的实验执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

# 第二个命令：不使用角色提示
echo "开始执行不带角色提示的实验..." | tee -a $LOG_FILE
python scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/scale_reason_norole

# 检查第二个命令是否成功
if [ $? -eq 0 ]; then
    echo "不带角色提示的实验成功完成！" | tee -a $LOG_FILE
else
    echo "不带角色提示的实验执行失败，退出代码: $?" | tee -a $LOG_FILE
    exit 1
fi

# 运行GPU占用监控脚本
echo "启动GPU占用监控..." | tee -a $LOG_FILE
python ../../GPUoccupy.py --gpu_ids 2 3 --utilization_threshold 60

echo "所有实验已完成 $(date)" | tee -a $LOG_FILE