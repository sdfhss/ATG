import torch
import numpy as np
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AdvancedAnomalyTransGNN


def load_sample_data(batch_size=32, seq_len=100, feature_dim=10):
    """
    加载示例数据用于演示。
    在实际应用中，应该替换为真实的数据加载逻辑。
    """
    # 生成随机数据
    data = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
    # 添加一些异常点
    for i in range(batch_size):
        # 随机选择一些位置作为异常点
        anomaly_positions = np.random.choice(seq_len, size=int(seq_len * 0.05), replace=False)
        for pos in anomaly_positions:
            # 将异常点的值放大
            data[i, pos, :] *= 3
    
    return torch.tensor(data)


def main():
    """
    使用高级图神经网络模型进行异常检测的示例。
    """
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 模型参数
    win_size = 100  # 窗口大小
    enc_in = 10     # 输入特征维度
    c_out = 10      # 输出特征维度
    
    # 初始化GATv2模型
    print("=== 使用GATv2模型 ===")
    model_gatv2 = AdvancedAnomalyTransGNN(
        win_size=win_size,
        enc_in=enc_in,
        c_out=c_out,
        d_model=64,        # Transformer模型维度
        n_heads=4,         # Transformer多头注意力的头数
        e_layers=2,        # Transformer编码器层数
        d_ff=128,          # Transformer前馈网络的隐藏层维度
        dropout=0.1,       # Dropout概率
        activation='gelu',  # 激活函数类型
        output_attention=True,  # 输出注意力权重
        gnn_hidden=32,     # GNN隐藏层维度
        gnn_layers=2,      # GNN层数
        gnn_type='gatv2',  # 使用GATv2
        gnn_heads=4,       # GNN注意力头数
        k_neighbors=5,     # 构建图时的k近邻数量
        fusion_method='weighted',  # 特征融合方法
        dynamic_graph=True,  # 使用动态图构建
        residual=True      # 使用残差连接
    )
    
    # 加载示例数据
    data = load_sample_data(batch_size=8, seq_len=win_size, feature_dim=enc_in)
    
    # 模型前向传播
    fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = model_gatv2(data)
    
    # 计算损失
    loss = model_gatv2.calculate_loss(data, fused_features, anomaly_score, series, prior)
    
    # 打印结果
    print(f"输入数据形状: {data.shape}")
    print(f"融合特征形状: {fused_features.shape}")
    print(f"异常分数形状: {anomaly_score.shape}")
    print(f"边索引形状: {edge_index.shape}")
    print(f"边权重形状: {edge_weight.shape}")
    print(f"损失值: {loss.item()}")
    
    # 识别异常点
    anomaly_threshold = torch.mean(anomaly_score) + 2 * torch.std(anomaly_score)  # 简单阈值法
    anomalies = (anomaly_score > anomaly_threshold).squeeze().cpu().detach().numpy()
    
    print(f"检测到的异常点数量: {np.sum(anomalies)}")
    print(f"异常点比例: {np.sum(anomalies) / (data.shape[0] * data.shape[1]):.2%}")
    
    # 初始化Graph Transformer模型
    print("\n=== 使用Graph Transformer模型 ===")
    model_gt = AdvancedAnomalyTransGNN(
        win_size=win_size,
        enc_in=enc_in,
        c_out=c_out,
        d_model=64,        # Transformer模型维度
        n_heads=4,         # Transformer多头注意力的头数
        e_layers=2,        # Transformer编码器层数
        d_ff=128,          # Transformer前馈网络的隐藏层维度
        dropout=0.1,       # Dropout概率
        activation='gelu',  # 激活函数类型
        output_attention=True,  # 输出注意力权重
        gnn_hidden=32,     # GNN隐藏层维度
        gnn_layers=2,      # GNN层数
        gnn_type='graph_transformer',  # 使用Graph Transformer
        gnn_heads=4,       # GNN注意力头数
        k_neighbors=5,     # 构建图时的k近邻数量
        fusion_method='weighted',  # 特征融合方法
        dynamic_graph=True,  # 使用动态图构建
        residual=True      # 使用残差连接
    )
    
    # 模型前向传播
    fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = model_gt(data)
    
    # 计算损失
    loss = model_gt.calculate_loss(data, fused_features, anomaly_score, series, prior)
    
    # 打印结果
    print(f"输入数据形状: {data.shape}")
    print(f"融合特征形状: {fused_features.shape}")
    print(f"异常分数形状: {anomaly_score.shape}")
    print(f"边索引形状: {edge_index.shape}")
    print(f"边权重形状: {edge_weight.shape}")
    print(f"损失值: {loss.item()}")
    
    # 识别异常点
    anomaly_threshold = torch.mean(anomaly_score) + 2 * torch.std(anomaly_score)  # 简单阈值法
    anomalies = (anomaly_score > anomaly_threshold).squeeze().cpu().detach().numpy()
    
    print(f"检测到的异常点数量: {np.sum(anomalies)}")
    print(f"异常点比例: {np.sum(anomalies) / (data.shape[0] * data.shape[1]):.2%}")
    
    print("\n=== 模型比较 ===")
    print("两种模型都利用了先进的图神经网络技术，但各有特点：")
    print("- GATv2: 提供动态注意力机制，能更好地捕捉节点间的动态关系")
    print("- Graph Transformer: 结合了图结构和Transformer的自注意力，更适合捕捉复杂的长距离依赖关系")
    print("在实际应用中，建议根据数据特性选择合适的模型或进行集成")


if __name__ == "__main__":
    main()