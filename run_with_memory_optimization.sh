#!/bin/bash

# GPU内存优化运行脚本
# 设置PyTorch内存分配优化参数

echo "设置GPU内存优化参数..."

# 设置PyTorch内存分配器参数
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 限制GPU内存使用比例 (80%)
export CUDA_VISIBLE_DEVICES=0

# 设置内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6

echo "环境变量设置完成:"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 运行你的Python脚本
echo "开始运行Python脚本..."
python3 "$@" 