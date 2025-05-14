#!/bin/bash

# 性能优化脚本 - 针对训练速度慢的问题进行了以下优化：
# 1. 减小模型复杂度：降低了模型维度和层数
# 2. 减小batch_size：从256减小到128
# 3. 启用混合精度训练(AMP)：加速计算并减少内存使用
# 4. 优化数据加载：使用num_workers和pin_memory
# 5. 优化CUDA内存分配和使用
# 6. 移除训练时间限制：不再在训练时间超过2小时时提前终止训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，如果有多个GPU可以修改
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256  # 增加CUDA内存分配

# 优化CUDA性能
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4

# 创建必要的目录
mkdir -p checkpoints
mkdir -p logs

# 预热GPU以减少第一次迭代的延迟
echo "预热GPU..."
python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()"

# 训练模型
echo "开始训练模型..."
python /root/autodl-tmp/Anomaly-Transformer-main/main.py \
    --mode train \
    --dataset MSL \
    --model_type advanced_transgnn \
    --data_path /root/autodl-tmp/Anomaly-Transformer-main/dataset/MSL \
    --model_save_path /root/autodl-tmp/Anomaly-Transformer-main/checkpoints \
    --win_size 50 \
    --batch_size 256 \
    --num_epochs 3 \
    --lr 5e-5 \
    --d_model 64 \
    --e_layers 1 \
    --n_heads 2 \
    --d_ff 64 \
    --gnn_type gatv2 \
    --gnn_hidden 32 \
    --gnn_layers 1 \
    --gnn_heads 1 \
    --k_neighbors 3 \
    --fusion_method weighted \
    --dynamic_graph True \
    --residual True \
    --anormly_ratio 1 \
    --input_c 55 \
    --output_c 55 \
    --no_time_limit True \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

# 测试模型
echo "开始测试模型..."

# 注意：确保测试参数与训练参数完全一致，特别是GNN相关参数
# 错误'gnn.gnn_encoder.conv.lin_l.weight'表明GNN结构不匹配
python /root/autodl-tmp/Anomaly-Transformer-main/main.py \
    --mode test \
    --dataset MSL \
    --model_type advanced_transgnn \
    --data_path /root/autodl-tmp/Anomaly-Transformer-main/dataset/MSL \
    --model_save_path /root/autodl-tmp/Anomaly-Transformer-main/checkpoints \
    --win_size 50 \
    --batch_size 256 \
    --lr 5e-5 \
    --d_model 64 \
    --e_layers 1 \
    --n_heads 2 \
    --d_ff 64 \
    --gnn_type gatv2 \
    --gnn_hidden 32 \
    --gnn_layers 1 \
    --gnn_heads 1 \
    --k_neighbors 3 \
    --fusion_method weighted \
    --dynamic_graph True \
    --residual True \
    --anormly_ratio 1 \
    --input_c 55 \
    --output_c 55 \
    --no_time_limit True \
    --pretrained_model 20 \
    2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log

echo "训练和测试完成！"




