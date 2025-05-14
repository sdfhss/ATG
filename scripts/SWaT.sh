#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，如果有多个GPU可以修改
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 优化CUDA内存分配

# 创建必要的目录
mkdir -p checkpoints
mkdir -p logs

# # 训练模型
echo "开始训练模型..."
python main.py \
    --mode train \
    --dataset SWaT \
    --model_type advanced_transgnn \
    --data_path ./dataset/SWaT \
    --model_save_path ./checkpoints \
    --win_size 50 \
    --batch_size 256 \
    --num_epochs 3 \
    --lr 1e-4 \
    --d_model 32 \
    --e_layers 1 \
    --n_heads 2 \
    --d_ff 32 \
    --gnn_type gatv2 \
    --gnn_hidden 32 \
    --gnn_layers 1 \
    --gnn_heads 1 \
    --k_neighbors 2 \
    --fusion_method add \
    --dynamic_graph True \
    --residual True \
    --anormly_ratio 4.00 \
    --input_c 51 \
    --output_c 51 \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

# 测试模型
echo "开始测试模型..."
python main.py \
    --mode test \
    --dataset SWaT \
    --model_type advanced_transgnn \
    --data_path ./dataset/SWaT \
    --model_save_path ./checkpoints \
    --win_size 50 \
    --batch_size 128 \
    --force_batch_size 256 \
    --d_model 32 \
    --e_layers 1 \
    --n_heads 2 \
    --d_ff 32 \
    --gnn_type gatv2 \
    --gnn_hidden 32 \
    --gnn_layers 1 \
    --gnn_heads 1 \
    --k_neighbors 2 \
    --fusion_method add \
    --dynamic_graph True \
    --residual True \
    --anormly_ratio 4.00 \
    --input_c 51 \
    --output_c 51 \
    2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log

echo "训练和测试完成！"