import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    """
    三角因果掩码，用于防止序列中的信息泄露。
    """
    def __init__(self, B, L, device="cpu"):
        """
        初始化掩码。
        :param B: 批量大小
        :param L: 序列长度
        :param device: 设备类型
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        返回掩码张量。
        """
        return self._mask


class AnomalyAttention(nn.Module):
    """
    异常检测注意力机制。
    """
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        """
        初始化异常注意力。
        :param win_size: 窗口大小
        :param mask_flag: 是否使用掩码
        :param scale: 缩放因子
        :param attention_dropout: 注意力 Dropout 概率
        :param output_attention: 是否输出注意力权重
        """
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.win_size = win_size
        # 不预先计算距离矩阵，而是在forward中按需计算

    def forward(self, queries, keys, values, sigma, attn_mask):
        """
        前向传播。
        :param queries: 查询张量
        :param keys: 键张量
        :param values: 值张量
        :param sigma: 标准差张量
        :param attn_mask: 注意力掩码
        :return: 注意力输出
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        # 计算先验分布
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        
        # 动态计算距离矩阵，避免存储大型矩阵
        i_matrix = torch.arange(window_size, device=queries.device).view(1, 1, -1, 1).repeat(B, H, 1, window_size)
        j_matrix = torch.arange(window_size, device=queries.device).view(1, 1, 1, -1).repeat(B, H, window_size, 1)
        distances = torch.abs(i_matrix - j_matrix)
        
        # 计算先验分布
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-distances.float() ** 2 / 2 / (sigma ** 2))

        # 应用注意力机制
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    """
    注意力层，封装了多头注意力机制。
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        """
        初始化注意力层。
        :param attention: 注意力机制
        :param d_model: 模型维度
        :param n_heads: 多头数量
        :param d_keys: 键的维度
        :param d_values: 值的维度
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播。
        :param queries: 查询张量
        :param keys: 键张量
        :param values: 值张量
        :param attn_mask: 注意力掩码
        :return: 注意力输出
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma