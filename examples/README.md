# Anomaly-TransGNN 使用指南

本文档提供了关于如何使用新添加的图神经网络（GNN）增强型异常检测模型的指南。

## 模型架构

`AnomalyTransGNN` 是一个集成模型，它结合了 Transformer 的重构能力和图神经网络（GNN）的关系建模能力，用于时间序列异常检测。该模型包含以下主要组件：

1. **Transformer 模块**：利用自注意力机制捕捉时间序列中的长期依赖关系
2. **图构建模块**：基于节点相似度构建时间序列数据点之间的图结构
3. **GNN 模块**：使用图卷积或图注意力网络处理构建的图结构
4. **特征融合层**：将 Transformer 和 GNN 的输出进行融合
5. **异常分数计算层**：基于融合特征计算异常分数

## 安装依赖

首先，安装所需的依赖项：

```bash
pip install -r requirements.txt
```

注意：PyTorch Geometric (torch_geometric) 及其相关包的安装可能需要特定的 CUDA 版本匹配，请参考[官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

## 使用方法

### 基本用法

```python
from model import AnomalyTransGNN

# 初始化模型
model = AnomalyTransGNN(
    win_size=100,        # 窗口大小
    enc_in=10,           # 输入特征维度
    c_out=10,            # 输出特征维度
    d_model=64,          # Transformer模型维度
    n_heads=4,           # Transformer多头注意力的头数
    e_layers=2,          # Transformer编码器层数
    gnn_hidden=32,       # GNN隐藏层维度
    gnn_type='gcn',      # GNN类型 ('gcn' 或 'gat')
    fusion_method='weighted'  # 特征融合方法 ('concat', 'add', 或 'weighted')
)

# 前向传播
fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = model(data)

# 计算损失
loss = model.calculate_loss(data, fused_features, anomaly_score, series, prior)
```

### 运行示例

我们提供了一个示例脚本，展示如何使用 `AnomalyTransGNN` 模型：

```bash
python examples/use_transgnn.py
```

## 模型优势

1. **结构感知**：通过图结构捕捉时间序列数据点之间的关系
2. **多角度特征提取**：结合了 Transformer 的全局建模能力和 GNN 的局部关系建模能力
3. **灵活的融合策略**：支持多种特征融合方法，适应不同的数据特性
4. **可解释性**：可以通过注意力权重和图结构分析异常原因

## 参数调优建议

- **k_neighbors**：控制图构建时每个节点连接的邻居数量，较大的值会增加图的密度
- **gnn_type**：'gcn' 适合一般情况，'gat' 在节点关系复杂时可能表现更好
- **fusion_method**：'weighted' 通常提供最佳性能，因为它可以自适应调整两个模型的贡献
- **gnn_hidden**：调整 GNN 的表达能力，复杂数据集可能需要更大的值

## 注意事项

- 该模型需要较大的计算资源，特别是在处理大规模时间序列数据时
- 图构建过程在大批量数据上可能较慢，考虑使用较小的批量大小
- 对于高维特征，可能需要增加 GNN 的层数和隐藏单元数量