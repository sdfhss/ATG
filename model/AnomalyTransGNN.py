import torch
import torch.nn as nn
import torch.nn.functional as F

from .AnomalyTransformer import AnomalyTransformer
from .gnn import AnomalyGNN


class AnomalyTransGNN(nn.Module):
    """
    集成模型，结合Transformer的重构能力和GNN的关系建模能力进行异常检测。
    """
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, gnn_hidden=256, 
                 gnn_layers=2, gnn_type='gcn', gnn_heads=1, k_neighbors=5, fusion_method='concat'):
        """
        初始化异常检测集成模型。
        :param win_size: 窗口大小
        :param enc_in: 输入特征维度
        :param c_out: 输出特征维度
        :param d_model: Transformer模型维度
        :param n_heads: Transformer多头注意力的头数
        :param e_layers: Transformer编码器层数
        :param d_ff: Transformer前馈网络的隐藏层维度
        :param dropout: Dropout概率
        :param activation: 激活函数类型
        :param output_attention: 是否输出注意力权重
        :param gnn_hidden: GNN隐藏层维度
        :param gnn_layers: GNN层数
        :param gnn_type: GNN类型，'gcn'或'gat'
        :param gnn_heads: GNN注意力头数（仅用于GAT）
        :param k_neighbors: 构建图时的k近邻数量
        :param fusion_method: 特征融合方法，'concat'、'add'或'weighted'
        """
        super(AnomalyTransGNN, self).__init__()
        
        # Transformer模型
        self.transformer = AnomalyTransformer(
            win_size=win_size,
            enc_in=enc_in,
            c_out=c_out,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention
        )
        
        # GNN模型
        self.gnn = AnomalyGNN(
            win_size=win_size,
            in_channels=enc_in,
            hidden_channels=gnn_hidden,
            out_channels=c_out,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            heads=gnn_heads,
            dropout=dropout,
            k_neighbors=k_neighbors
        )
        
        self.fusion_method = fusion_method
        self.output_attention = output_attention
        
        # 特征融合层
        if fusion_method == 'concat':
            self.fusion_layer = nn.Linear(c_out * 2, c_out)
        elif fusion_method == 'weighted':
            self.weight_transformer = nn.Parameter(torch.FloatTensor([0.5]))
            self.weight_gnn = nn.Parameter(torch.FloatTensor([0.5]))
        
        # 异常分数计算层
        self.anomaly_layer = nn.Sequential(
            nn.Linear(c_out, c_out // 2),
            nn.ReLU(),
            nn.Linear(c_out // 2, 1)
        )
        
    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 融合后的输出，异常分数，以及可能的注意力信息
        """
        # Transformer处理
        if self.output_attention:
            trans_out, series, prior, sigmas = self.transformer(x)
        else:
            trans_out = self.transformer(x)
        
        # GNN处理
        gnn_out, edge_index, edge_weight = self.gnn(x)
        
        # 特征融合
        if self.fusion_method == 'concat':
            # 拼接融合
            fused_features = torch.cat([trans_out, gnn_out], dim=-1)
            fused_features = self.fusion_layer(fused_features)
        elif self.fusion_method == 'add':
            # 加法融合
            fused_features = trans_out + gnn_out
        elif self.fusion_method == 'weighted':
            # 加权融合
            weight_t = torch.sigmoid(self.weight_transformer)
            weight_g = torch.sigmoid(self.weight_gnn)
            sum_weight = weight_t + weight_g
            weight_t = weight_t / sum_weight
            weight_g = weight_g / sum_weight
            fused_features = weight_t * trans_out + weight_g * gnn_out
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
        
        # 计算异常分数
        anomaly_score = self.anomaly_layer(fused_features)
        
        if self.output_attention:
            return fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight
        else:
            return fused_features, anomaly_score
    
    def calculate_loss(self, x, fused_features, anomaly_score=None, series=None, prior=None):
        """
        计算损失函数。
        :param x: 原始输入
        :param fused_features: 融合特征
        :param anomaly_score: 异常分数
        :param series: 注意力序列
        :param prior: 先验分布
        :return: 总损失
        """
        # 重构损失
        rec_loss = F.mse_loss(fused_features, x)
        
        # 如果有注意力信息，计算关联损失
        if series is not None and prior is not None:
            # 计算KL散度
            kl_loss = 0
            for i in range(len(series)):
                kl_div = F.kl_div(torch.log(series[i] + 1e-8), prior[i], reduction='batchmean')
                kl_loss += kl_div
            kl_loss = kl_loss / len(series)
            
            # 总损失 = 重构损失 - λ * KL散度
            # 负号是因为我们希望最大化KL散度（使注意力分布与先验分布不同）
            loss = rec_loss - 0.1 * kl_loss
        else:
            loss = rec_loss
        
        return loss