import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .AnomalyTransformer import AnomalyTransformer
from .advanced_gnn import AdvancedAnomalyGNN


class AdvancedAnomalyTransGNN(nn.Module):
    """
    高级集成模型，结合Transformer的重构能力和先进GNN（GATv2或GraphTransformer）的关系建模能力进行异常检测。
    
    论文参考：
    - GATv2: "How Attentive are Graph Attention Networks?" (ICLR 2022)
    - Graph Transformer: "Graph Transformer Networks" (NeurIPS 2019)
    """
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, gnn_hidden=256, 
                 gnn_layers=2, gnn_type='gatv2', gnn_heads=4, k_neighbors=5, 
                 fusion_method='concat', dynamic_graph=True, residual=True, debug=False):
        """
        初始化高级异常检测集成模型。
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
        :param gnn_type: GNN类型，'gatv2'或'graph_transformer'
        :param gnn_heads: GNN注意力头数
        :param k_neighbors: 构建图时的k近邻数量
        :param fusion_method: 特征融合方法，'concat'、'add'或'weighted'
        :param dynamic_graph: 是否使用动态图构建（考虑时间信息）
        :param residual: 是否在GNN中使用残差连接
        :param debug: 是否启用调试输出
        """
        super(AdvancedAnomalyTransGNN, self).__init__()
        self.debug = debug
        self.debug_counter = 0  # 用于控制调试输出频率
        
        # 仅在调试模式下打印初始化参数
        if self.debug:
            print(f"初始化AdvancedAnomalyTransGNN - win_size:{win_size}, enc_in:{enc_in}, c_out:{c_out}, d_model:{d_model}, n_heads:{n_heads}")
        
        # 针对RTX 3090优化参数
        # 检测是否有大显存GPU
        if torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if total_mem > 20:  # 确认是大显存GPU (如RTX 3090)
                    # 针对RTX 3090优化模型参数
                    d_model = max(d_model, 768)  # 增大模型维度
                    d_ff = max(d_ff, 768)  # 增大前馈网络维度
                    gnn_hidden = max(gnn_hidden, 384)  # 增大GNN隐藏层维度
                    if not self.debug:
                        print(f"检测到大显存GPU ({total_mem:.1f}GB)，自动优化模型参数")
            except Exception as e:
                if self.debug:
                    print(f"GPU检测失败: {str(e)}")
        
        # 确保参数有效
        win_size = max(5, min(win_size, 100))  # 限制窗口大小在5-100之间
        d_model = max(16, min(d_model, 1024))   # 针对RTX 3090，允许更大的模型维度
        
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
        
        # 高级GNN模型 - 传递debug参数
        self.gnn = AdvancedAnomalyGNN(
            win_size=win_size,
            in_channels=enc_in,
            hidden_channels=gnn_hidden,
            out_channels=c_out,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            heads=gnn_heads,
            dropout=dropout,
            k_neighbors=k_neighbors,
            dynamic_graph=dynamic_graph,
            residual=residual,
            debug=debug
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
        
        # 启用混合精度训练以提高性能和GPU利用率
        self.use_amp = torch.cuda.is_available()
        
    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 融合后的输出，异常分数，以及可能的注意力信息
        """
        # 性能监控 - 仅在调试模式下启用
        start_time = time.time() if self.debug else None
        
        # 验证输入
        if x.dim() != 3:
            raise ValueError(f"输入维度应为3，但得到了{x.dim()}")
            
        # 确保输入不包含NaN或Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            if self.debug:
                print("警告: 输入包含NaN或Inf值，已替换为0")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 针对RTX 3090优化的混合精度训练设置
        # 强制启用混合精度以提高性能和GPU利用率
        use_amp = torch.cuda.is_available()
        
        # 输入太大时，使用分批处理 - 针对RTX 3090优化的批处理大小
        B, L, D = x.shape
        # 增大批处理阈值，充分利用RTX 3090的24GB显存
        if B > 64 and L * D > 10000:  # 提高阈值以减少分批处理，充分利用RTX 3090
            try:
                # 为了减少内存使用，对输入执行批次分割
                max_batch_size = 32  # 增大最大批次大小，充分利用RTX 3090的24GB显存
                
                if not torch.is_grad_enabled():  # 在评估模式下
                    # 在评估模式下可以分批处理
                    outputs = []
                    for i in range(0, B, max_batch_size):
                        sub_batch = x[i:i+max_batch_size]
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                if self.output_attention:
                                    sub_trans_out, sub_series, sub_prior, sub_sigmas = self.transformer(sub_batch)
                                else:
                                    sub_trans_out = self.transformer(sub_batch)
                                
                                # 高级GNN处理
                                sub_gnn_out, sub_edge_index, sub_edge_weight, sub_gnn_attn = self.gnn(sub_batch)
                                
                                # 特征融合
                                if self.fusion_method == 'concat':
                                    # 拼接融合
                                    sub_fused_features = torch.cat([sub_trans_out, sub_gnn_out], dim=-1)
                                    sub_fused_features = self.fusion_layer(sub_fused_features)
                                elif self.fusion_method == 'add':
                                    # 加法融合
                                    sub_fused_features = sub_trans_out + sub_gnn_out
                                elif self.fusion_method == 'weighted':
                                    # 加权融合
                                    weight_t = torch.sigmoid(self.weight_transformer)
                                    weight_g = torch.sigmoid(self.weight_gnn)
                                    sum_weight = weight_t + weight_g
                                    weight_t = weight_t / sum_weight
                                    weight_g = weight_g / sum_weight
                                    sub_fused_features = weight_t * sub_trans_out + weight_g * sub_gnn_out
                                else:
                                    raise ValueError(f"不支持的融合方法: {self.fusion_method}")
                                
                                # 计算异常分数 - 结合GNN的注意力权重
                                sub_anomaly_base = self.anomaly_layer(sub_fused_features)
                                sub_anomaly_score = sub_anomaly_base * sub_gnn_attn
                                
                                if self.output_attention:
                                    outputs.append((sub_fused_features, sub_anomaly_score, sub_series, sub_prior, sub_sigmas, sub_edge_index, sub_edge_weight))
                                else:
                                    outputs.append((sub_fused_features, sub_anomaly_score))
                    
                    # 合并结果
                    if self.output_attention:
                        fused_features = torch.cat([o[0] for o in outputs], dim=0)
                        anomaly_score = torch.cat([o[1] for o in outputs], dim=0)
                        series = [torch.cat([o[2][i] for o in outputs], dim=0) for i in range(len(outputs[0][2]))]
                        prior = [torch.cat([o[3][i] for o in outputs], dim=0) for i in range(len(outputs[0][3]))]
                        sigmas = torch.cat([o[4] for o in outputs], dim=0)
                        # edge_index和edge_weight不能简单合并
                        edge_index, edge_weight = outputs[0][5], outputs[0][6]
                        return fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight
                    else:
                        fused_features = torch.cat([o[0] for o in outputs], dim=0)
                        anomaly_score = torch.cat([o[1] for o in outputs], dim=0)
                        return fused_features, anomaly_score
            except Exception as e:
                if self.debug:
                    print(f"分批处理时出错：{str(e)}，尝试常规处理")
                # 继续执行常规处理
        
        # 常规处理方式 - 针对RTX 3090优化
        try:
            # 使用混合精度训练提高性能 - 针对RTX 3090优化
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 预先分配GPU内存，减少内存碎片
                if torch.cuda.is_available() and self.debug and self.debug_counter == 0:
                    # 每隔一段时间检查GPU内存使用情况
                    mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    print(f"GPU内存使用: 已分配={mem_allocated:.2f}GB, 已预留={mem_reserved:.2f}GB")
                
                # Transformer处理
                transformer_start = time.time() if self.debug else None
                if self.output_attention:
                    trans_out, series, prior, sigmas = self.transformer(x)
                else:
                    trans_out = self.transformer(x)
                
                if self.debug and transformer_start and self.debug_counter == 0:
                    print(f"Transformer处理耗时: {(time.time() - transformer_start)*1000:.2f}ms")
                
                # 高级GNN处理
                gnn_start = time.time() if self.debug else None
                gnn_out, edge_index, edge_weight, gnn_attn = self.gnn(x)
                
                if self.debug and gnn_start and self.debug_counter == 0:
                    print(f"GNN处理耗时: {(time.time() - gnn_start)*1000:.2f}ms")
                
                # 特征融合 - 针对RTX 3090优化内存使用
                fusion_start = time.time() if self.debug else None
                if self.fusion_method == 'concat':
                    # 拼接融合
                    fused_features = torch.cat([trans_out, gnn_out], dim=-1)
                    fused_features = self.fusion_layer(fused_features)
                elif self.fusion_method == 'add':
                    # 加法融合 - 内存效率更高
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
                
                if self.debug and fusion_start and self.debug_counter == 0:
                    print(f"特征融合耗时: {(time.time() - fusion_start)*1000:.2f}ms")
                
                # 计算异常分数 - 结合GNN的注意力权重
                anomaly_start = time.time() if self.debug else None
                anomaly_base = self.anomaly_layer(fused_features)
                anomaly_score = anomaly_base * gnn_attn
                
                if self.debug and anomaly_start and self.debug_counter == 0:
                    print(f"异常分数计算耗时: {(time.time() - anomaly_start)*1000:.2f}ms")
            
            # 更新调试计数器，减少调试输出频率以提高性能
            if self.debug:
                self.debug_counter = (self.debug_counter + 1) % 50  # 降低调试输出频率，提高性能
                if start_time and self.debug_counter == 0:
                    print(f"总前向传播耗时: {(time.time() - start_time)*1000:.2f}ms, 输入形状: {x.shape}")
            
            # 清理不需要的中间变量，减少内存使用
            if not self.training and torch.cuda.is_available():
                # 在评估模式下，主动释放不需要的张量
                del trans_out, gnn_out
                if 'anomaly_base' in locals():
                    del anomaly_base
                # 不要在这里调用torch.cuda.empty_cache()，会影响性能
            
            if self.output_attention:
                return fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight
            else:
                return fused_features, anomaly_score
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if self.debug:
                    print("CUDA内存不足，尝试释放内存")
                if torch.cuda.is_available():
                    try:
                        # 清空缓存
                        torch.cuda.empty_cache()
                        # 打印内存使用情况
                        if self.debug:
                            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                            print(f"清理后GPU内存: 已分配={mem_allocated:.2f}GB, 已预留={mem_reserved:.2f}GB")
                    except Exception as cache_error:
                        if self.debug:
                            print(f"清空缓存失败: {str(cache_error)}")
                # 返回错误
                raise e
            else:
                # 其他错误
                if self.debug:
                    print(f"前向传播时出错：{str(e)}")
                raise e
    
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