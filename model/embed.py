import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    """
    位置嵌入模块，用于为输入序列添加位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置嵌入。
        :param d_model: 特征维度
        :param max_len: 最大序列长度
        """
        super(PositionalEmbedding, self).__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 计算位置编码
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 添加批量维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播，返回与输入序列长度匹配的位置信息。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 位置嵌入，形状为 [1, L, D]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Token 嵌入模块，用于将输入特征映射到指定维度。
    """
    def __init__(self, c_in, d_model):
        """
        初始化 Token 嵌入。
        :param c_in: 输入特征维度
        :param d_model: 输出特征维度
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2  # 根据 PyTorch 版本设置填充
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化卷积权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 [B, L, C]
        :return: 嵌入后的张量，形状为 [B, L, D]
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """
    数据嵌入模块，结合 Token 嵌入和位置嵌入。
    """
    def __init__(self, c_in, d_model, dropout=0.0):
        """
        初始化数据嵌入。
        :param c_in: 输入特征维度
        :param d_model: 输出特征维度
        :param dropout: Dropout 概率
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)  # Token 嵌入
        self.position_embedding = PositionalEmbedding(d_model=d_model)  # 位置嵌入

        self.dropout = nn.Dropout(p=dropout)  # Dropout 层

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 [B, L, C]
        :return: 嵌入后的张量，形状为 [B, L, D]
        """
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)