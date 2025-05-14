import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv


class GraphConstructor(nn.Module):
    """
    图构建模块，用于从时间序列数据构建图结构。
    """
    def __init__(self, win_size, k_neighbors=5, threshold=0.5, self_loop=True):
        """
        初始化图构建器。
        :param win_size: 窗口大小
        :param k_neighbors: k近邻数量
        :param threshold: 边权重阈值
        :param self_loop: 是否添加自环
        """
        super(GraphConstructor, self).__init__()
        self.win_size = win_size
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.self_loop = self_loop
        
    def forward(self, x):
        """
        前向传播，构建图结构。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 边索引和边权重
        """
        B, L, D = x.shape
        device = x.device
        
        # 计算节点间相似度矩阵
        x_flat = x.view(B * L, D)
        similarity = torch.mm(x_flat, x_flat.transpose(0, 1))
        norm = torch.mm(torch.norm(x_flat, dim=1).unsqueeze(1), torch.norm(x_flat, dim=1).unsqueeze(0))
        similarity = similarity / (norm + 1e-8)
        
        # 构建邻接矩阵
        adj = torch.zeros(B * L, B * L, device=device)
        
        # 对每个节点找到k个最相似的邻居
        _, indices = torch.topk(similarity, k=min(self.k_neighbors + 1, B * L), dim=1)
        
        # 构建边索引和权重
        edge_index = []
        edge_weight = []
        
        for i in range(B * L):
            for j in indices[i][1:]:  # 跳过自身
                if similarity[i, j] > self.threshold:
                    edge_index.append([i, j.item()])
                    edge_weight.append(similarity[i, j].item())
        
        # 添加自环
        if self.self_loop:
            for i in range(B * L):
                edge_index.append([i, i])
                edge_weight.append(1.0)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=device)
        
        return edge_index, edge_weight


class GNNLayer(nn.Module):
    """
    图神经网络层，支持GCN和GAT。
    """
    def __init__(self, in_channels, out_channels, gnn_type='gcn', heads=1, dropout=0.1):
        """
        初始化GNN层。
        :param in_channels: 输入特征维度
        :param out_channels: 输出特征维度
        :param gnn_type: GNN类型，'gcn'或'gat'
        :param heads: 注意力头数（仅用于GAT）
        :param dropout: Dropout概率
        """
        super(GNNLayer, self).__init__()
        self.gnn_type = gnn_type
        
        if gnn_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            self.conv = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
            
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播。
        :param x: 节点特征，形状为 [N, in_channels]
        :param edge_index: 边索引，形状为 [2, E]
        :param edge_weight: 边权重，形状为 [E]
        :return: 更新后的节点特征，形状为 [N, out_channels]
        """
        if self.gnn_type == 'gcn':
            out = self.conv(x, edge_index, edge_weight)
        else:  # gat
            out = self.conv(x, edge_index)
            
        out = self.activation(out)
        out = self.dropout(out)
        out = self.norm(out)
        
        return out


class GNNEncoder(nn.Module):
    """
    多层GNN编码器。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, gnn_type='gcn', heads=1, dropout=0.1):
        """
        初始化GNN编码器。
        :param in_channels: 输入特征维度
        :param hidden_channels: 隐藏层特征维度
        :param out_channels: 输出特征维度
        :param num_layers: GNN层数
        :param gnn_type: GNN类型，'gcn'或'gat'
        :param heads: 注意力头数（仅用于GAT）
        :param dropout: Dropout概率
        """
        super(GNNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(GNNLayer(in_channels, hidden_channels, gnn_type, heads, dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_channels, hidden_channels, gnn_type, heads, dropout))
            
        # 最后一层
        self.layers.append(GNNLayer(hidden_channels, out_channels, gnn_type, heads, dropout))
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播。
        :param x: 节点特征，形状为 [N, in_channels]
        :param edge_index: 边索引，形状为 [2, E]
        :param edge_weight: 边权重，形状为 [E]
        :return: 更新后的节点特征，形状为 [N, out_channels]
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            
        return x


class AnomalyGNN(nn.Module):
    """
    异常检测图神经网络模型。
    """
    def __init__(self, win_size, in_channels, hidden_channels, out_channels, 
                 num_layers=2, gnn_type='gcn', heads=1, dropout=0.1, k_neighbors=5):
        """
        初始化异常检测GNN模型。
        :param win_size: 窗口大小
        :param in_channels: 输入特征维度
        :param hidden_channels: 隐藏层特征维度
        :param out_channels: 输出特征维度
        :param num_layers: GNN层数
        :param gnn_type: GNN类型，'gcn'或'gat'
        :param heads: 注意力头数（仅用于GAT）
        :param dropout: Dropout概率
        :param k_neighbors: 构建图时的k近邻数量
        """
        super(AnomalyGNN, self).__init__()
        
        self.graph_constructor = GraphConstructor(win_size, k_neighbors=k_neighbors)
        self.gnn_encoder = GNNEncoder(in_channels, hidden_channels, out_channels, 
                                     num_layers, gnn_type, heads, dropout)
        self.projection = nn.Linear(out_channels, in_channels)
        
    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 重构后的张量，形状为 [B, L, D]
        """
        B, L, D = x.shape
        device = x.device
        
        # 构建图
        edge_index, edge_weight = self.graph_constructor(x)
        
        # 展平批次维度
        x_flat = x.reshape(B * L, D)
        
        # 应用GNN
        h = self.gnn_encoder(x_flat, edge_index, edge_weight)
        
        # 投影回原始维度
        out = self.projection(h)
        
        # 恢复原始形状
        out = out.reshape(B, L, D)
        
        return out, edge_index, edge_weight