#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=i64m1tga800ue
#SBATCH -J noprompt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=high

# 加载CUDA环境
module load cuda/12.2

# 激活conda环境
source /hpc2hdd/home/bwang423/miniconda3/bin/activate ssr

# 运行Python脚本
python /hpc2hdd/home/bwang423/test/SSR-Spec_Scale_Reasoning/scale_reason.py \
    --dataset_name live \
    --problem_id 0-45 \
    --repeat_id 3 \
    --model_name Qwen/QwQ-32B \
    --output_dir results/scale_reason_r3_noprompt \
