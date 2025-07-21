#!/bin/bash

echo "[INFO] 启动 Qwen/QwQ-32B 模型中..."

# 启动 Qwen/QwQ-32B 模型（占用 80% 显存，端口 30000）
VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B \
  --dtype auto \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --port 30000 > qwen32b.log 2>&1 &

echo "[INFO] 启动 DeepSeek-R1-Distill-Qwen-1.5B 模型中..."
# 启动 DeepSeek-R1-Distill-Qwen-1.5B 模型（占用 20% 显存，端口 30001）
VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --dtype auto \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.15 \
  --enable-prefix-caching \
  --port 30001 > deepseek15b.log 2>&1 &


echo "[INFO] 模型正在加载中，请使用以下命令查看部署状态："
echo "  tail -f qwen32b.log"
echo "  tail -f deepseek15b.log"