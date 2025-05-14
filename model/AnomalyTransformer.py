import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer  # 导入自定义的注意力机制模块
from .embed import DataEmbedding, TokenEmbedding  # 导入嵌入模块


# 定义编码器中的单层结构
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        初始化编码器层。
        :param attention: 注意力机制模块
        :param d_model: 输入特征维度
        :param d_ff: 前馈网络的隐藏层维度，默认为4倍的d_model
        :param dropout: Dropout概率
        :param activation: 激活函数类型（"relu"或"gelu"）
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 如果未指定d_ff，则默认为4倍的d_model
        self.attention = attention  # 注意力机制
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 1D卷积层1
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 1D卷积层2
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化1
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化2
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.activation = F.relu if activation == "relu" else F.gelu  # 激活函数

    def forward(self, x, attn_mask=None):
        """
        前向传播。
        :param x: 输入张量，形状为[B, L, D]
        :param attn_mask: 注意力掩码
        :return: 编码后的张量，注意力权重，掩码，sigma值
        """
        # 应用注意力机制
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)  # 残差连接并应用Dropout
        y = x = self.norm1(x)  # 层归一化
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 1D卷积和激活函数
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 1D卷积和Dropout

        return self.norm2(x + y), attn, mask, sigma  # 返回结果


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        """
        初始化编码器。
        :param attn_layers: 注意力层列表
        :param norm_layer: 最后的归一化层
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # 将注意力层封装为ModuleList
        self.norm = norm_layer  # 最后的归一化层

    def forward(self, x, attn_mask=None):
        """
        前向传播。
        :param x: 输入张量，形状为[B, L, D]
        :param attn_mask: 注意力掩码
        :return: 编码后的张量，时间序列列表，先验列表，sigma列表
        """
        series_list = []  # 存储时间序列信息
        prior_list = []  # 存储先验信息
        sigma_list = []  # 存储sigma值
        for attn_layer in self.attn_layers:  # 遍历每一层注意力层
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:  # 如果定义了归一化层，则应用
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list  # 返回结果


# 定义异常检测Transformer模型
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=256, n_heads=4, e_layers=3, d_ff=256,
                 dropout=0.0, activation='gelu', output_attention=True):
        """
        初始化异常检测Transformer模型。
        :param win_size: 窗口大小
        :param enc_in: 输入特征维度
        :param c_out: 输出特征维度
        :param d_model: 模型维度
        :param n_heads: 多头注意力的头数
        :param e_layers: 编码器层数
        :param d_ff: 前馈网络的隐藏层维度
        :param dropout: Dropout概率
        :param activation: 激活函数类型
        :param output_attention: 是否输出注意力权重
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention  # 是否输出注意力权重

        # 数据嵌入层
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)  # 根据层数创建多个编码器层
            ],
            norm_layer=torch.nn.LayerNorm(d_model)  # 最后的归一化层
        )

        # 投影层，将编码器输出映射到目标维度
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为[B, L, D]
        :return: 模型输出，可能包括注意力权重等
        """
        enc_out = self.embedding(x)  # 数据嵌入
        enc_out, series, prior, sigmas = self.encoder(enc_out)  # 编码器处理
        enc_out = self.projection(enc_out)  # 投影到输出维度

        if self.output_attention:  # 如果需要输出注意力信息
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # 仅返回编码结果