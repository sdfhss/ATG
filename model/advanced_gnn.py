import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv


class GraphConstructor(nn.Module):
    """
    图构建模块，用于从时间序列数据构建图结构。
    """
    def __init__(self, win_size, k_neighbors=5, threshold=0.5, self_loop=True, dynamic_graph=False, debug=False):
        """
        初始化图构建器。
        :param win_size: 窗口大小
        :param k_neighbors: k近邻数量
        :param threshold: 边权重阈值
        :param self_loop: 是否添加自环
        :param dynamic_graph: 是否使用动态图构建（考虑时间信息）
        :param debug: 是否启用调试输出
        """
        super(GraphConstructor, self).__init__()
        self.win_size = win_size
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.self_loop = self_loop
        self.dynamic_graph = dynamic_graph
        self.debug = debug
        
    def forward(self, x):
        """
        前向传播，构建图结构。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 边索引和边权重
        """
        B, L, D = x.shape
        device = x.device
        
        # 针对RTX 3090优化的内存使用
        # 检查输入大小，对于大型输入使用分块处理
        use_amp = torch.cuda.is_available()  # 启用混合精度计算
        use_chunked = B * L > 5000  # 对于大型输入使用分块处理
        
        # 使用混合精度计算以提高性能
        with torch.cuda.amp.autocast(enabled=use_amp):
            # 计算节点间相似度矩阵 - 针对RTX 3090优化
            x_flat = x.view(B * L, D)
            
            if use_chunked and not torch.is_grad_enabled():
                # 分块计算相似度矩阵以减少内存使用
                chunk_size = 1024  # 针对RTX 3090优化的块大小
                num_nodes = B * L
                similarity = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float16 if use_amp else torch.float32)
                
                for i in range(0, num_nodes, chunk_size):
                    end_i = min(i + chunk_size, num_nodes)
                    chunk_i = x_flat[i:end_i]
                    
                    # 计算当前块与所有节点的相似度
                    chunk_sim = torch.mm(chunk_i, x_flat.transpose(0, 1))
                    chunk_norm_i = torch.norm(chunk_i, dim=1).unsqueeze(1)
                    chunk_norm_all = torch.norm(x_flat, dim=1).unsqueeze(0)
                    chunk_sim = chunk_sim / (torch.mm(chunk_norm_i, chunk_norm_all) + 1e-8)
                    
                    # 更新相似度矩阵
                    similarity[i:end_i, :] = chunk_sim
            else:
                # 标准计算方式 - 对于较小的输入
                similarity = torch.mm(x_flat, x_flat.transpose(0, 1))
                norm = torch.mm(torch.norm(x_flat, dim=1).unsqueeze(1), torch.norm(x_flat, dim=1).unsqueeze(0))
                similarity = similarity / (norm + 1e-8)
            
            # 如果使用动态图构建，考虑时间信息 - 优化实现
            if self.dynamic_graph:
                # 优化时间惩罚矩阵计算 - 使用向量化操作
                if use_chunked and not torch.is_grad_enabled():
                    # 分块计算时间惩罚
                    time_penalty = torch.zeros_like(similarity)
                    
                    for b in range(B):
                        # 为每个批次创建时间索引矩阵
                        time_indices = torch.arange(L, device=device).float()
                        time_diff_matrix = torch.abs(time_indices.unsqueeze(1) - time_indices.unsqueeze(0)) / L
                        
                        # 计算时间惩罚
                        batch_penalty = torch.exp(-3 * time_diff_matrix)
                        
                        # 将批次惩罚应用到相应位置
                        start_idx = b * L
                        end_idx = (b + 1) * L
                        time_penalty[start_idx:end_idx, start_idx:end_idx] = batch_penalty
                else:
                    # 更高效的时间惩罚计算
                    time_penalty = torch.zeros_like(similarity)
                    
                    # 为每个批次创建时间惩罚矩阵
                    for b in range(B):
                        # 创建时间索引
                        time_indices = torch.arange(L, device=device).float()
                        # 计算时间差异矩阵
                        time_diff_matrix = torch.abs(time_indices.unsqueeze(1) - time_indices.unsqueeze(0)) / L
                        # 计算惩罚
                        batch_penalty = torch.exp(-3 * time_diff_matrix)
                        
                        # 应用到相应位置
                        start_idx = b * L
                        end_idx = (b + 1) * L
                        time_penalty[start_idx:end_idx, start_idx:end_idx] = batch_penalty
                
                # 结合相似度和时间信息
                similarity = similarity * time_penalty
            
            # 对每个节点找到k个最相似的邻居 - 内存优化版本
            k = min(self.k_neighbors + 1, B * L)
            
            # 使用topk获取最相似的邻居
            if use_chunked and not torch.is_grad_enabled():
                # 分块处理topk以减少内存使用
                edge_index = []
                edge_weight = []
                
                for i in range(0, B * L, chunk_size):
                    end_i = min(i + chunk_size, B * L)
                    # 获取当前块的topk
                    _, indices_chunk = torch.topk(similarity[i:end_i], k=k, dim=1)
                    
                    # 处理当前块的边
                    for local_i, global_i in enumerate(range(i, end_i)):
                        for j_idx in range(1, k):  # 跳过自身
                            j = indices_chunk[local_i, j_idx].item()
                            if similarity[global_i, j] > self.threshold:
                                edge_index.append([global_i, j])
                                edge_weight.append(similarity[global_i, j].item())
            else:
                # 标准处理方式
                _, indices = torch.topk(similarity, k=k, dim=1)
                
                # 构建边索引和权重 - 使用列表推导式优化
                edge_index = [
                    [i, j.item()]
                    for i in range(B * L)
                    for j in indices[i][1:]
                    if similarity[i, j] > self.threshold
                ]
                
                edge_weight = [
                    similarity[i, j].item()
                    for i in range(B * L)
                    for j in indices[i][1:]
                    if similarity[i, j] > self.threshold
                ]
            
            # 添加自环 - 优化版本
            if self.self_loop:
                self_loops = [[i, i] for i in range(B * L)]
                self_weights = [1.0] * (B * L)
                
                edge_index.extend(self_loops)
                edge_weight.extend(self_weights)
            
            # 转换为张量
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
                edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=device)
            else:
                # 处理边缘情况：没有边
                edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
                edge_weight = torch.ones(1, dtype=torch.float, device=device)
                if self.debug:
                    print("警告: 未找到任何边，创建默认边")
        
        # 清理不需要的大型张量以减少内存使用
        del similarity
        if self.dynamic_graph:
            del time_penalty
        torch.cuda.empty_cache()  # 针对RTX 3090优化内存使用
        
        return edge_index, edge_weight


class GATv2Layer(nn.Module):
    """
    GATv2层，实现动态图注意力机制。
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1, concat=True, debug=False):
        """
        初始化GATv2层。
        :param in_channels: 输入特征维度
        :param out_channels: 输出特征维度
        :param heads: 注意力头数
        :param dropout: Dropout概率
        :param concat: 是否拼接多头注意力的结果
        :param debug: 是否启用调试输出
        """
        super(GATv2Layer, self).__init__()
        self.debug = debug
        if self.debug:
            print(f"GATv2Layer初始化 - in_channels: {in_channels}, out_channels: {out_channels}, heads: {heads}")
        
        self.conv = GATv2Conv(
            in_channels=in_channels, 
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=1  # 添加edge_dim参数，确保可以处理边特征
        )
        
        # 计算输出维度
        final_out_dim = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(final_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.elu
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播。
        :param x: 节点特征，形状为 [N, in_channels]
        :param edge_index: 边索引，形状为 [2, E]
        :param edge_weight: 边权重，形状为 [E]
        :return: 更新后的节点特征
        """
        if self.debug:
            print(f"GATv2Layer输入 - x: {x.shape}, edge_index: {edge_index.shape}")
        
        # 确保edge_weight是正确的形状
        if edge_weight is not None:
            # 将一维的edge_weight转换为二维，以符合edge_attr的要求
            edge_weight = edge_weight.view(-1, 1)
            
        out = self.conv(x, edge_index, edge_attr=edge_weight)
        
        if self.debug:
            print(f"GATv2Layer卷积后 - out: {out.shape}")
        
        out = self.activation(out)
        out = self.dropout(out)
        out = self.norm(out)
        
        return out


class GraphTransformerLayer(nn.Module):
    """
    图Transformer层，结合了图结构和Transformer的自注意力机制。
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1, concat=True, beta=False):
        """
        初始化图Transformer层。
        :param in_channels: 输入特征维度
        :param out_channels: 输出特征维度
        :param heads: 注意力头数
        :param dropout: Dropout概率
        :param concat: 是否拼接多头注意力的结果
        :param beta: 是否使用beta变换（增强边特征）
        """
        super(GraphTransformerLayer, self).__init__()
        self.conv = TransformerConv(
            in_channels=in_channels, 
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            beta=beta
        )
        
        self.norm = nn.LayerNorm(out_channels if not concat else out_channels * heads)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播。
        :param x: 节点特征，形状为 [N, in_channels]
        :param edge_index: 边索引，形状为 [2, E]
        :param edge_weight: 边权重，形状为 [E]
        :return: 更新后的节点特征
        """
        out = self.conv(x, edge_index, edge_attr=edge_weight)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.norm(out)
        
        return out


class AdvancedGNNEncoder(nn.Module):
    """
    高级GNN编码器，支持GATv2和GraphTransformer。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 gnn_type='gatv2', heads=4, dropout=0.1, residual=True):
        """
        初始化高级GNN编码器。
        :param in_channels: 输入特征维度
        :param hidden_channels: 隐藏层特征维度
        :param out_channels: 输出特征维度
        :param num_layers: GNN层数
        :param gnn_type: GNN类型，'gatv2'或'graph_transformer'
        :param heads: 注意力头数
        :param dropout: Dropout概率
        :param residual: 是否使用残差连接
        """
        super(AdvancedGNNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        self.residual = residual
        self.gnn_type = gnn_type
        self.out_channels = out_channels
        
        # 第一层
        if gnn_type == 'gatv2':
            self.layers.append(GATv2Layer(in_channels, hidden_channels, heads, dropout, concat=True))
            hidden_dim = hidden_channels * heads
        elif gnn_type == 'graph_transformer':
            self.layers.append(GraphTransformerLayer(in_channels, hidden_channels, heads, dropout, concat=True))
            hidden_dim = hidden_channels * heads
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        # 中间层
        for _ in range(num_layers - 2):
            if gnn_type == 'gatv2':
                self.layers.append(GATv2Layer(hidden_dim, hidden_channels, heads, dropout, concat=True))
            elif gnn_type == 'graph_transformer':
                self.layers.append(GraphTransformerLayer(hidden_dim, hidden_channels, heads, dropout, concat=True))
        
        # 最后一层 - 输出层不拼接多头结果
        if num_layers > 1:
            if gnn_type == 'gatv2':
                self.layers.append(GATv2Layer(hidden_dim, out_channels, heads, dropout, concat=False))
            elif gnn_type == 'graph_transformer':
                self.layers.append(GraphTransformerLayer(hidden_dim, out_channels, heads, dropout, concat=False))
        
        # 如果使用残差连接，添加投影层
        if residual:
            self.res_projs = nn.ModuleList()
            # 第一层的残差投影
            self.res_projs.append(nn.Linear(in_channels, hidden_dim))
            # 中间层的残差投影
            for _ in range(num_layers - 2):
                self.res_projs.append(nn.Identity())
            # 最后一层的残差投影
            if num_layers > 1:
                self.res_projs.append(nn.Linear(hidden_dim, out_channels))
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播。
        :param x: 节点特征，形状为 [N, in_channels]
        :param edge_index: 边索引，形状为 [2, E]
        :param edge_weight: 边权重，形状为 [E]
        :return: 更新后的节点特征，形状为 [N, out_channels]
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            # 应用GNN层
            h_new = layer(h, edge_index, edge_weight)
            
            # 应用残差连接
            if self.residual:
                res = self.res_projs[i](h)
                h = h_new + res
            else:
                h = h_new
            
        return h


class AdvancedAnomalyGNN(nn.Module):
    """
    高级异常检测图神经网络模型。
    """
    def __init__(self, win_size, in_channels, hidden_channels, out_channels, 
                 num_layers=2, gnn_type='gatv2', heads=4, dropout=0.1, 
                 k_neighbors=5, dynamic_graph=True, residual=True, debug=False):
        """
        初始化高级异常检测GNN模型。
        :param win_size: 窗口大小
        :param in_channels: 输入特征维度
        :param hidden_channels: 隐藏层特征维度
        :param out_channels: 输出特征维度
        :param num_layers: GNN层数
        :param gnn_type: GNN类型，'gatv2'或'graph_transformer'
        :param heads: 注意力头数
        :param dropout: Dropout概率
        :param k_neighbors: 构建图时的k近邻数量
        :param dynamic_graph: 是否使用动态图构建
        :param residual: 是否使用残差连接
        :param debug: 是否启用调试输出
        """
        super(AdvancedAnomalyGNN, self).__init__()
        
        self.win_size = win_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.debug = debug
        self.debug_counter = 0  # 用于控制调试输出频率
        
        # 仅在调试模式下打印模型参数
        if self.debug:
            print(f"AdvancedAnomalyGNN初始化 - in_channels: {in_channels}, hidden_channels: {hidden_channels}, out_channels: {out_channels}")
        
        self.graph_constructor = GraphConstructor(
            win_size, 
            k_neighbors=k_neighbors, 
            dynamic_graph=dynamic_graph
        )
        
        # 强制设置GNN输出维度与输入维度相同
        actual_out_channels = in_channels
        
        # 为简化模型，我们直接使用一个基本的GNN编码器
        if num_layers == 1:
            # 单层GNN
            if gnn_type == 'gatv2':
                self.gnn_encoder = GATv2Layer(in_channels, actual_out_channels, heads, dropout, concat=False, debug=debug)
            else:
                self.gnn_encoder = GraphTransformerLayer(in_channels, actual_out_channels, heads, dropout, concat=False)
        else:
            # 多层GNN
            self.gnn_encoder = AdvancedGNNEncoder(
                in_channels, 
                hidden_channels, 
                actual_out_channels, 
                num_layers, 
                gnn_type, 
                heads, 
                dropout,
                residual
            )
        
        # 确保注意力层使用正确的输入维度
        self.attention = nn.Sequential(
            nn.Linear(actual_out_channels, actual_out_channels // 2),
            nn.Tanh(),
            nn.Linear(actual_out_channels // 2, 1)
        )
        
        # 启用混合精度训练以提高性能
        self.use_amp = torch.cuda.is_available()
        
    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 [B, L, D]
        :return: 重构后的张量，形状为 [B, L, D]，以及其他信息
        """
        B, L, D = x.shape
        device = x.device
        
        # 仅在调试模式下且计数器为0时打印信息
        if self.debug and self.debug_counter == 0:
            print(f"AdvancedAnomalyGNN输入 - 形状: {x.shape}, 通道数: {D}")
        
        # 针对RTX 3090优化的批处理策略
        # 对于大批次，我们使用分批处理以避免OOM
        max_batch_size = 2048  # 针对RTX 3090的24GB显存优化
        if B * L > max_batch_size and not torch.is_grad_enabled():
            # 仅在评估模式下使用分批处理
            if self.debug and self.debug_counter == 0:
                print(f"使用分批处理: 输入大小 {B*L} > 阈值 {max_batch_size}")
            
            # 分批处理结果
            all_outs = []
            all_attn_weights = []
            
            # 保存第一批的edge_index和edge_weight用于返回
            saved_edge_index = None
            saved_edge_weight = None
            
            # 计算每批的大小
            batch_size = max(1, max_batch_size // L)
            
            for i in range(0, B, batch_size):
                end_idx = min(i + batch_size, B)
                x_batch = x[i:end_idx]
                
                # 处理当前批次
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # 构建图
                    edge_index, edge_weight = self.graph_constructor(x_batch)
                    
                    if saved_edge_index is None:
                        saved_edge_index = edge_index
                        saved_edge_weight = edge_weight
                    
                    # 展平批次维度
                    batch_size_curr = end_idx - i
                    x_flat = x_batch.reshape(batch_size_curr * L, D)
                    
                    # 应用GNN
                    if isinstance(self.gnn_encoder, (GATv2Layer, GraphTransformerLayer)):
                        h = self.gnn_encoder(x_flat, edge_index, edge_weight)
                    else:
                        h = self.gnn_encoder(x_flat, edge_index, edge_weight)
                    
                    # 计算注意力分数
                    attn_scores = self.attention(h)
                    attn_weights_batch = F.softmax(attn_scores, dim=0)
                    
                    # 恢复原始形状
                    out_batch = h.reshape(batch_size_curr, L, D)
                    attn_weights_batch = attn_weights_batch.reshape(batch_size_curr, L, 1)
                    
                    all_outs.append(out_batch)
                    all_attn_weights.append(attn_weights_batch)
            
            # 合并结果
            out = torch.cat(all_outs, dim=0)
            attn_weights = torch.cat(all_attn_weights, dim=0)
            
            # 更新调试计数器
            if self.debug:
                self.debug_counter = (self.debug_counter + 1) % 20
            
            return out, saved_edge_index, saved_edge_weight, attn_weights
        
        # 标准处理流程 - 针对RTX 3090优化
        try:
            # 使用混合精度计算以提高性能
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 构建图 - 使用优化的图构建器
                edge_index, edge_weight = self.graph_constructor(x)
                
                # 展平批次维度
                x_flat = x.reshape(B * L, D)
                
                # 应用内存优化 - 针对RTX 3090
                if torch.cuda.is_available() and x_flat.requires_grad and B * L * D > 1000000:
                    # 对于大型输入，使用检查点以减少内存使用
                    # 这对于RTX 3090特别有用，可以处理更大的批次
                    if hasattr(torch.utils.checkpoint, 'checkpoint') and not self.debug:
                        # 定义检查点函数
                        def checkpoint_fn(module, inputs, edge_idx, edge_wt):
                            return module(inputs, edge_idx, edge_wt)
                        
                        # 使用检查点运行GNN编码器
                        if isinstance(self.gnn_encoder, (GATv2Layer, GraphTransformerLayer)):
                            h = torch.utils.checkpoint.checkpoint(
                                checkpoint_fn, self.gnn_encoder, x_flat, edge_index, edge_weight)
                        else:
                            h = torch.utils.checkpoint.checkpoint(
                                checkpoint_fn, self.gnn_encoder, x_flat, edge_index, edge_weight)
                    else:
                        # 标准处理
                        if isinstance(self.gnn_encoder, (GATv2Layer, GraphTransformerLayer)):
                            h = self.gnn_encoder(x_flat, edge_index, edge_weight)
                        else:
                            h = self.gnn_encoder(x_flat, edge_index, edge_weight)
                else:
                    # 标准处理 - 对于较小的输入
                    if isinstance(self.gnn_encoder, (GATv2Layer, GraphTransformerLayer)):
                        h = self.gnn_encoder(x_flat, edge_index, edge_weight)
                    else:
                        h = self.gnn_encoder(x_flat, edge_index, edge_weight)
                
                if self.debug and self.debug_counter == 0:
                    print(f"GNN编码器输出 - 形状: {h.shape}")
                
                # 计算注意力分数
                attn_scores = self.attention(h)  # [B*L, 1]
                
                # 使用更稳定的softmax实现
                attn_weights = F.softmax(attn_scores, dim=0)
                
                # 直接使用GNN输出作为重构结果
                out = h
                
                # 恢复原始形状
                out = out.reshape(B, L, D)
                attn_weights = attn_weights.reshape(B, L, 1)
        
        except RuntimeError as e:
            # 处理OOM错误 - 针对RTX 3090优化
            if 'out of memory' in str(e) and torch.cuda.is_available():
                # 清理缓存并重试，但使用较小的批次
                torch.cuda.empty_cache()
                if self.debug:
                    print(f"GPU内存不足，尝试使用较小的批次处理")
                
                # 使用较小的批次重试
                half_batch = B // 2
                if half_batch > 0:
                    # 处理第一半
                    out1, edge_index, edge_weight, attn_weights1 = self.forward(x[:half_batch])
                    # 处理第二半
                    out2, _, _, attn_weights2 = self.forward(x[half_batch:])
                    # 合并结果
                    out = torch.cat([out1, out2], dim=0)
                    attn_weights = torch.cat([attn_weights1, attn_weights2], dim=0)
                    return out, edge_index, edge_weight, attn_weights
                else:
                    # 无法进一步分割，重新抛出异常
                    raise
            else:
                # 其他类型的错误，重新抛出
                raise
        
        # 更新调试计数器，减少打印频率以提高性能
        if self.debug:
            self.debug_counter = (self.debug_counter + 1) % 100  # 降低调试输出频率
        
        return out, edge_index, edge_weight, attn_weights