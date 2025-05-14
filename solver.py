import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model import AnomalyTransformer, AnomalyTransGNN, AdvancedAnomalyTransGNN
from data_factory.data_loader import get_loader_segment
from tqdm import tqdm


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        
        # 自动调整batch_size以提高GPU利用率 - 针对RTX 3090 24GB显存优化
        if torch.cuda.is_available():
            try:
                # 获取GPU总内存和当前使用量
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                free_mem = total_mem - allocated_mem
                
                # 针对RTX 3090的24GB显存，更激进地使用显存
                # 估算每个样本需要的内存
                sample_size_estimate = 0.005  # 降低每个样本的估计内存需求
                if hasattr(self, 'input_c') and hasattr(self, 'win_size'):
                    sample_size_estimate = self.input_c * self.win_size * 4 * 2 / (1024**2)  # MB，降低系数
                
                # 计算可用的最大batch_size (使用85%的可用内存)
                max_batch_size = int((free_mem * 0.85 * 1024) / sample_size_estimate)
                
                # 设置新的batch_size，确保至少为32，最大为1024
                new_batch_size = max(32, min(1024, max_batch_size))
                
                # 强制使用更大的batch_size
                if hasattr(self, 'force_batch_size') and self.force_batch_size:
                    new_batch_size = self.force_batch_size
                elif new_batch_size > self.batch_size:
                    print(f"自动调整batch_size: {self.batch_size} -> {new_batch_size} (GPU内存: {free_mem:.2f}GB可用)")
                    self.batch_size = new_batch_size
                else:
                    # 即使计算出的batch_size较小，也尝试使用更大的值
                    min_batch_size = 512  # 针对RTX 3090设置较大的最小batch_size
                    if self.batch_size < min_batch_size:
                        print(f"强制增大batch_size: {self.batch_size} -> {min_batch_size} (针对RTX 3090 24GB优化)")
                        self.batch_size = min_batch_size
                
                print(f"当前GPU利用率: {allocated_mem/total_mem*100:.2f}%，目标提高到60-80%")
            except Exception as e:
                print(f"自动调整batch_size失败: {str(e)}")

        # 优化数据加载
        self.train_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        self.vali_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='val',
            dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        
        # 优化CUDA性能 - 针对RTX 3090优化
        if torch.cuda.is_available():
            try:
                # 清空缓存
                torch.cuda.empty_cache()
                
                # 启用cuDNN自动调优
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能
                
                # 设置CUDA工作模式为异步执行
                torch.cuda.set_device(0)
                torch.cuda.current_stream().synchronize()
                
                # 设置更激进的内存分配策略
                if hasattr(torch.cuda, 'memory_stats'):
                    # 打印初始内存状态
                    mem_stats = torch.cuda.memory_stats()
                    print(f"初始GPU内存分配: {mem_stats.get('allocated_bytes.all.current', 0) / (1024**3):.2f}GB")
                    print(f"初始GPU内存预留: {mem_stats.get('reserved_bytes.all.current', 0) / (1024**3):.2f}GB")
                
                # 设置CUDA缓存分配器
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    # 允许使用更多GPU内存
                    torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
                    print("已设置GPU内存使用上限为95%")
                
                print("CUDA性能优化设置完成，已针对RTX 3090进行优化")
            except Exception as e:
                print(f"CUDA优化设置失败: {str(e)}")

    def build_model(self):
        # 针对RTX 3090优化模型参数
        # 对于24GB显存，可以使用更大的模型
        win_size = self.win_size  # 不限制窗口大小，充分利用GPU
        
        # 使用命令行参数中的d_model和e_layers，如果未指定则使用默认值
        # 针对RTX 3090，可以使用更大的模型参数
        d_model = getattr(self, 'd_model', 512)  # 默认值为512
        e_layers = getattr(self, 'e_layers', 3)  # 默认值为3
        n_heads = getattr(self, 'n_heads', 8)  # 默认值为8
        d_ff = getattr(self, 'd_ff', 512)  # 默认值为512
        
        # 是否启用调试输出
        debug = getattr(self, 'debug', False)  # 默认关闭调试输出
        
        # 根据RTX 3090的24GB显存调整模型大小
        if torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_mem = total_mem - torch.cuda.memory_allocated(0) / (1024**3)
                
                # 针对RTX 3090，可以使用更大的模型
                if total_mem > 20:  # 确认是大显存GPU (如RTX 3090)
                    # 如果有足够的显存，可以增大模型参数
                    if free_mem > 16:  # 有大量空闲显存
                        d_model = max(d_model, 768)  # 增大模型维度
                        d_ff = max(d_ff, 768)  # 增大前馈网络维度
                        n_heads = max(n_heads, 12)  # 增加注意力头数
                        if debug:
                            print(f"检测到RTX 3090大显存 ({free_mem:.2f}GB可用)，增大模型参数: d_model={d_model}, d_ff={d_ff}, n_heads={n_heads}")
                        
                        # 针对RTX 3090，大幅增加batch_size以提高GPU利用率
                        if hasattr(self, 'batch_size') and self.batch_size < 128:
                            old_batch_size = self.batch_size
                            self.batch_size = max(128, min(512, self.batch_size * 4))
                            print(f"针对RTX 3090优化：增大batch_size: {old_batch_size} -> {self.batch_size}")
                elif free_mem < 4:  # 小于4GB时缩小模型
                    d_model = min(d_model, 256)
                    d_ff = min(d_ff, 256)
                    if debug:
                        print(f"GPU内存不足 ({free_mem:.2f}GB)，自动缩小模型大小: d_model={d_model}, d_ff={d_ff}")
            except Exception as e:
                if debug:
                    print(f"调整模型大小时出错: {str(e)}")
        
        # 根据模型类型选择不同的模型
        if hasattr(self, 'model_type') and self.model_type == 'advanced_transgnn':
            self.model = AdvancedAnomalyTransGNN(
                win_size=win_size,
                enc_in=self.input_c,
                c_out=self.output_c,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                dropout=0.1,
                activation='gelu',
                output_attention=True,
                gnn_hidden=self.gnn_hidden,
                gnn_layers=self.gnn_layers,
                gnn_type=self.gnn_type,
                gnn_heads=self.gnn_heads,
                k_neighbors=self.k_neighbors,
                fusion_method=self.fusion_method,
                dynamic_graph=self.dynamic_graph,
                residual=self.residual,
                debug=debug  # 传递调试参数
            )
        elif hasattr(self, 'model_type') and self.model_type == 'transgnn':
            self.model = AnomalyTransGNN(
                win_size=win_size,
                enc_in=self.input_c,
                c_out=self.output_c,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                dropout=0.1,
                activation='gelu',
                output_attention=True,
                gnn_hidden=self.gnn_hidden,
                gnn_layers=self.gnn_layers,
                gnn_type=self.gnn_type,
                gnn_heads=self.gnn_heads,
                k_neighbors=self.k_neighbors,
                fusion_method=self.fusion_method
            )
        else:  # 默认使用AnomalyTransformer
            self.model = AnomalyTransformer(
                win_size=win_size,
                enc_in=self.input_c,
                c_out=self.output_c,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                dropout=0.1,
                activation='gelu',
                output_attention=True
            )
        
        # 针对RTX 3090优化的优化器设置
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            eps=1e-8,  # 提高数值稳定性
            weight_decay=getattr(self, 'weight_decay', 1e-4),  # 添加权重衰减以减少过拟合
            betas=(0.9, 0.999)  # 使用默认动量参数
        )
        
        # 强制启用混合精度训练以提高性能和GPU利用率
        self.use_amp = True
        if torch.cuda.is_available():
            try:
                # 检查是否支持自动混合精度
                if hasattr(torch.cuda, 'amp'):
                    self.scaler = torch.cuda.amp.GradScaler()
                    print("已强制启用自动混合精度训练，提高GPU利用率")
                else:
                    self.use_amp = False
                    if debug:
                        print("当前PyTorch版本不支持自动混合精度训练")
            except Exception as e:
                if debug:
                    print(f"设置混合精度训练失败: {str(e)}")
                self.use_amp = False

        # 将模型移至GPU并优化内存使用
        if torch.cuda.is_available():
            # 使用float16精度来减少内存使用
            if self.use_amp:
                # 在启用AMP的情况下，模型参数仍然是float32，但前向传播会使用float16
                self.model.cuda()
            else:
                self.model.cuda()
            
            # 尝试使用并行处理
            if torch.cuda.device_count() > 1 and hasattr(self, 'use_parallel') and self.use_parallel:
                print(f"使用 {torch.cuda.device_count()} 个GPU进行并行训练")
                self.model = nn.DataParallel(self.model)
            else:
                print("使用单GPU训练 - RTX 3090")
                
            # 打印模型参数量
            model_size = sum(p.numel() for p in self.model.parameters())
            print(f"模型总参数量: {model_size/1e6:.2f}M 参数")
            
            # 预热GPU
            if debug:
                print("预热GPU...")
            dummy_input = torch.randn(self.batch_size, win_size, self.input_c).cuda()
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.cuda.synchronize()
            if debug:
                print("GPU预热完成，准备开始训练")

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            # 根据模型类型处理不同的返回值
            if hasattr(self, 'model_type') and self.model_type == 'advanced_transgnn':
                fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                output = fused_features
            else:
                output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        # 是否启用调试输出
        debug = getattr(self, 'debug', False)  # 默认关闭调试输出
        
        print("======================TRAIN MODE======================")

        # 降低batch_size以解决OOM问题
        original_batch_size = self.batch_size
        # 如果batch_size大于256，则降低到256
        if self.batch_size > 256:
            print(f"降低batch_size以解决OOM问题: {self.batch_size} -> 256")
            self.batch_size = 256
            
            # 重新加载数据，使用新的batch_size
            self.train_loader = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='train',
                dataset=self.dataset
            )
            self.vali_loader = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='val',
                dataset=self.dataset
            )
            self.test_loader = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='test',
                dataset=self.dataset
            )
            self.thre_loader = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='thre',
                dataset=self.dataset
            )
        
        # 如果使用的是高级GNN模型，设置dynamic_graph=False以减少内存使用
        if hasattr(self, 'model_type') and self.model_type == 'advanced_transgnn':
            if hasattr(self.model, 'graph_constructor') and hasattr(self.model.graph_constructor, 'dynamic_graph'):
                if self.model.graph_constructor.dynamic_graph:
                    print("设置dynamic_graph=False以减少内存使用")
                    self.model.graph_constructor.dynamic_graph = False
            
            # 减少k_neighbors值以减少边数量
            if hasattr(self.model, 'graph_constructor') and hasattr(self.model.graph_constructor, 'k_neighbors'):
                if self.model.graph_constructor.k_neighbors > 3:
                    old_k = self.model.graph_constructor.k_neighbors
                    self.model.graph_constructor.k_neighbors = 3
                    print(f"减少k_neighbors值以减少边数量: {old_k} -> 3")
        
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        
        # 强制清理CUDA缓存
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print("已清空CUDA缓存，准备开始训练")
            except Exception as e:
                if debug:
                    print(f"清空CUDA缓存失败: {str(e)}")
        
        # 添加最大训练时间限制
        max_training_hours = 240
        start_time = time.time()
        # 检查是否禁用训练时间限制
        no_time_limit = getattr(self, 'no_time_limit', False)

        for epoch in range(self.num_epochs):
            # 检查是否超过最大训练时间（仅当未禁用时间限制时）
            if not no_time_limit and (time.time() - start_time) / 3600 > max_training_hours:
                print(f"训练时间超过{max_training_hours}小时，提前终止训练")
                break
            
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            
            print(f"\n开始第 {epoch + 1}/{self.num_epochs} 轮训练")
            
            # 创建进度条
            pbar = tqdm(range(train_steps), desc=f'Epoch {epoch + 1}/{self.num_epochs}',
                       bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            
            for i in pbar:
                # 定期清理CUDA缓存以减少内存碎片 - 减少清理频率以提高性能
                if i % 50 == 0 and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        if debug:
                            print(f"清空CUDA缓存失败: {str(e)}")
                
                iter_count += 1
                
                try:
                    batch_x, batch_y = self.train_loader.__iter__().__next__()
                except StopIteration:
                    batch_x, batch_y = next(iter(self.train_loader))
                
                input = batch_x.float().to(self.device)
                
                # 使用梯度累积减少内存使用 - 对于RTX 3090，减少累积步数以提高吞吐量
                accumulation_steps = 1  # 对于RTX 3090，不需要梯度累积，直接更新
                loss1_batch = 0
                
                # 使用混合精度训练以提高性能
                try:
                    if hasattr(self, 'use_amp') and self.use_amp:
                        with torch.cuda.amp.autocast():
                            # 根据模型类型处理不同的返回值
                            if hasattr(self, 'model_type') and self.model_type == 'advanced_transgnn':
                                try:
                                    # 在每个批次处理前清理CUDA缓存
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    
                                    fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                                    output = fused_features
                                except RuntimeError as e:
                                    if 'out of memory' in str(e):
                                        # 清理缓存
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                        print(f"\nOOM错误发生在图构建过程中，尝试进一步降低batch_size...")
                                        # 如果batch_size大于128，则进一步降低
                                        if self.batch_size > 128:
                                            self.batch_size = max(64, self.batch_size // 2)
                                            print(f"进一步降低batch_size: {self.batch_size*2} -> {self.batch_size}")
                                            # 重新加载数据
                                            self.train_loader = get_loader_segment(
                                                data_path=self.data_path,
                                                batch_size=self.batch_size,
                                                win_size=self.win_size,
                                                mode='train',
                                                dataset=self.dataset
                                            )
                                            train_steps = len(self.train_loader)
                                            pbar.close()
                                            return self.train()  # 重新开始训练
                                        else:
                                            # 如果batch_size已经很小，则尝试关闭动态图构建
                                            if hasattr(self.model, 'graph_constructor'):
                                                if hasattr(self.model.graph_constructor, 'dynamic_graph') and self.model.graph_constructor.dynamic_graph:
                                                    print("关闭动态图构建以减少内存使用")
                                                    self.model.graph_constructor.dynamic_graph = False
                                                    if torch.cuda.is_available():
                                                        torch.cuda.empty_cache()
                                                    continue  # 重试当前批次
                                                
                                                # 进一步减少k_neighbors
                                                if hasattr(self.model.graph_constructor, 'k_neighbors') and self.model.graph_constructor.k_neighbors > 2:
                                                    old_k = self.model.graph_constructor.k_neighbors
                                                    self.model.graph_constructor.k_neighbors = 2
                                                    print(f"进一步减少k_neighbors: {old_k} -> 2")
                                                    if torch.cuda.is_available():
                                                        torch.cuda.empty_cache()
                                                    continue  # 重试当前批次
                                    raise e  # 如果不是OOM错误或无法解决，则重新抛出异常
                            else:
                                output, series, prior, _ = self.model(input)
                            
                            # 计算关联差异
                            series_loss = 0.0
                            prior_loss = 0.0
                            for u in range(len(prior)):
                                series_loss += (torch.mean(my_kl_loss(series[u], (
                                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               self.win_size)).detach())) + torch.mean(
                                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                       self.win_size)).detach(),
                                               series[u])))
                                prior_loss += (torch.mean(my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            self.win_size)),
                                    series[u].detach())) + torch.mean(
                                    my_kl_loss(series[u].detach(), (
                                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                   self.win_size)))))                        
                            series_loss = series_loss / len(prior)
                            prior_loss = prior_loss / len(prior)
                            
                            rec_loss = self.criterion(output, input)
                            
                            loss1_batch = (rec_loss - self.k * series_loss).item()
                            loss1_list.append(loss1_batch)
                            loss1 = rec_loss - self.k * series_loss
                            loss2 = rec_loss + self.k * prior_loss
                        
                        # 缩放梯度以匹配混合精度训练
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss1).backward(retain_graph=True)
                        self.scaler.scale(loss2).backward()
                        
                        if (i + 1) % accumulation_steps == 0 or (i + 1) == train_steps:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            
                        # 每个批次处理后清理CUDA缓存
                        if (i + 1) % 5 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"\nCUDA内存不足，尝试清理缓存并降低batch_size...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # 降低batch_size
                        if self.batch_size > 64:
                            self.batch_size = max(64, self.batch_size // 2)
                            print(f"降低batch_size: {self.batch_size*2} -> {self.batch_size}")
                            # 重新加载数据
                            self.train_loader = get_loader_segment(
                                data_path=self.data_path,
                                batch_size=self.batch_size,
                                win_size=self.win_size,
                                mode='train',
                                dataset=self.dataset
                            )
                            train_steps = len(self.train_loader)
                            pbar.close()
                            return self.train()  # 重新开始训练
                    else:
                        raise e  # 如果不是OOM错误，则重新抛出异常
                else:
                    # 常规训练（无混合精度）
                    try:
                        self.optimizer.zero_grad()
                        
                        # 在每个批次处理前清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 根据模型类型处理不同的返回值
                        if hasattr(self, 'model_type') and self.model_type == 'advanced_transgnn':
                            try:
                                fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                                output = fused_features
                            except RuntimeError as e:
                                if 'out of memory' in str(e):
                                    # 清理缓存
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    print(f"\nOOM错误发生在图构建过程中，尝试进一步降低batch_size...")
                                    # 如果batch_size大于128，则进一步降低
                                    if self.batch_size > 128:
                                        self.batch_size = max(64, self.batch_size // 2)
                                        print(f"进一步降低batch_size: {self.batch_size*2} -> {self.batch_size}")
                                        # 重新加载数据
                                        self.train_loader = get_loader_segment(
                                            data_path=self.data_path,
                                            batch_size=self.batch_size,
                                            win_size=self.win_size,
                                            mode='train',
                                            dataset=self.dataset
                                        )
                                        train_steps = len(self.train_loader)
                                        pbar.close()
                                        return self.train()  # 重新开始训练
                                    else:
                                        # 如果batch_size已经很小，则尝试关闭动态图构建
                                        if hasattr(self.model, 'graph_constructor'):
                                            if hasattr(self.model.graph_constructor, 'dynamic_graph') and self.model.graph_constructor.dynamic_graph:
                                                print("关闭动态图构建以减少内存使用")
                                                self.model.graph_constructor.dynamic_graph = False
                                                if torch.cuda.is_available():
                                                    torch.cuda.empty_cache()
                                                continue  # 重试当前批次
                                            
                                            # 进一步减少k_neighbors
                                            if hasattr(self.model.graph_constructor, 'k_neighbors') and self.model.graph_constructor.k_neighbors > 2:
                                                old_k = self.model.graph_constructor.k_neighbors
                                                self.model.graph_constructor.k_neighbors = 2
                                                print(f"进一步减少k_neighbors: {old_k} -> 2")
                                                if torch.cuda.is_available():
                                                    torch.cuda.empty_cache()
                                                continue  # 重试当前批次
                                raise e  # 如果不是OOM错误或无法解决，则重新抛出异常
                        else:
                            output, series, prior, _ = self.model(input)
                        
                        # 计算关联差异
                        series_loss = 0.0
                        prior_loss = 0.0
                        for u in range(len(prior)):
                            series_loss += (torch.mean(my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach())) + torch.mean(
                                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                                   self.win_size)).detach(),
                                           series[u])))
                            prior_loss += (torch.mean(my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach())) + torch.mean(
                                my_kl_loss(series[u].detach(), (
                                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                               self.win_size)))))                    
                        series_loss = series_loss / len(prior)
                        prior_loss = prior_loss / len(prior)
                        
                        rec_loss = self.criterion(output, input)
                        
                        loss1_batch = (rec_loss - self.k * series_loss).item()
                        loss1_list.append(loss1_batch)
                        loss1 = rec_loss - self.k * series_loss
                        loss2 = rec_loss + self.k * prior_loss
                        
                        # Minimax策略
                        loss1.backward(retain_graph=True)
                        loss2.backward()
                        self.optimizer.step()
                        
                        # 每个批次处理后清理CUDA缓存
                        if (i + 1) % 5 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"\nCUDA内存不足，尝试清理缓存并降低batch_size...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # 降低batch_size
                            if self.batch_size > 64:
                                self.batch_size = max(64, self.batch_size // 2)
                                print(f"降低batch_size: {self.batch_size*2} -> {self.batch_size}")
                                # 重新加载数据
                                self.train_loader = get_loader_segment(
                                    data_path=self.data_path,
                                    batch_size=self.batch_size,
                                    win_size=self.win_size,
                                    mode='train',
                                    dataset=self.dataset
                                )
                                train_steps = len(self.train_loader)
                                pbar.close()
                                return self.train()  # 重新开始训练
                        else:
                            raise e  # 如果不是OOM错误，则重新抛出异常
                
                # 更新参数 - 针对RTX 3090优化
                if iter_count % accumulation_steps == 0 or iter_count == len(self.train_loader):
                    if hasattr(self, 'use_amp') and self.use_amp:
                        # 使用混合精度训练的优化器步骤
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # 监控GPU利用率 - 仅在调试模式下每100次迭代显示一次
                        if debug and iter_count % 100 == 0 and torch.cuda.is_available():
                            try:
                                mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                                mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                                print(f"\tGPU内存: 已分配={mem_allocated:.2f}GB, 已预留={mem_reserved:.2f}GB, 利用率={(mem_allocated/total_mem)*100:.1f}%")
                                
                                # 针对RTX 3090的优化设置
                                if total_mem > 20:  # 超过20GB显存，可能是RTX 3090
                                    print(f"检测到RTX 3090 GPU，启用高性能训练模式")
                                    # 设置更激进的cuDNN基准模式
                                    torch.backends.cudnn.benchmark = True
                                    # 禁用确定性模式以提高性能
                                    torch.backends.cudnn.deterministic = False
                            except Exception as e:
                                print(f"GPU监控失败: {str(e)}")
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # 更新进度条 - 针对RTX 3090优化显示更详细的内存信息
                mem_info = ""
                if torch.cuda.is_available():
                    try:
                        # 显示更详细的内存信息
                        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        mem_info = f'{mem_allocated:.1f}GB/{total_mem:.1f}GB ({(mem_allocated/total_mem)*100:.0f}%)'
                        
                        # 监控GPU利用率 - 仅在调试模式下每100次迭代显示一次
                        if debug and iter_count % 100 == 0:
                            print(f"\tGPU内存: 已分配={mem_allocated:.2f}GB, 总内存={total_mem:.1f}GB, 利用率={(mem_allocated/total_mem)*100:.1f}%")
                    except Exception as e:
                        mem_info = "无法获取内存信息"
                
                pbar.set_postfix({
                    'loss': f'{loss1_batch:.4f}',
                    'speed': f'{(time.time() - time_now) / iter_count:.2f}s/iter',
                    'memory': mem_info,
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })

                if (i + 1) % 10 == 0:
                    mem_usage = ""
                    if torch.cuda.is_available():
                        try:
                            current_mem = torch.cuda.memory_allocated() / 1024**2
                            max_mem = torch.cuda.max_memory_allocated() / 1024**2
                            mem_usage = f"{current_mem:.0f}MB / {max_mem:.0f}MB"
                        except Exception as e:
                            mem_usage = "无法获取内存信息"
                        
                    print(f"批次: {i+1}/{train_steps}, 损失: {loss1_batch:.4f}, "
                          f"内存: {mem_usage}")
                    time_now = time.time()

            print(f"Epoch {epoch + 1} 完成，耗时: {time.time() - epoch_time:.2f}秒")
            
            # 手动进行垃圾回收
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"清空CUDA缓存失败: {str(e)}")
                
            import gc
            gc.collect()
            
            # 计算平均损失
            train_loss = np.average(loss1_list)

            try:
                vali_loss1, vali_loss2 = self.vali(self.test_loader)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                        epoch + 1, train_steps, train_loss, vali_loss1))
                early_stopping(vali_loss1, vali_loss2, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            except Exception as e:
                print(f"验证过程中出错: {str(e)}")
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        try:
            # 确保在函数开始时导入所需的库
            import torch
            import gc
            
            model_path = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
            if not os.path.exists(model_path):
                print(f"错误：找不到模型文件 {model_path}")
                print("请先运行训练模式生成模型文件。")
                return
            
            # 加载模型参数
            try:
                # 尝试直接加载
                state_dict = torch.load(model_path)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                print("尝试使用兼容模式加载模型...")
                
                # 兼容模式：只加载匹配的参数
                state_dict = torch.load(model_path)
                model_dict = self.model.state_dict()
                
                # 过滤出匹配的参数
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                missing_keys = [k for k in model_dict.keys() if k not in filtered_state_dict]
                unexpected_keys = [k for k in state_dict.keys() if k not in model_dict]
                
                if missing_keys:
                    print(f"警告：以下参数在预训练模型中缺失: {missing_keys}")
                if unexpected_keys:
                    print(f"警告：预训练模型包含不需要的参数: {unexpected_keys}")
                
                # 加载匹配的参数
                model_dict.update(filtered_state_dict)
                self.model.load_state_dict(model_dict)
                print("模型已使用兼容模式加载，部分参数可能未加载。")
            
            self.model.eval()
            temperature = 50

            print("======================TEST MODE======================")
            print("清除缓存")

            gc.collect()
            torch.cuda.empty_cache()

            criterion = nn.MSELoss(reduce=False)

            # (1) stastic on the train set
            attens_energy = []
            print("计算训练集统计信息...")
            for i, (input_data, labels) in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                              desc='处理训练集'):
                input = input_data.float().to(self.device)
                
                # 根据模型类型处理返回值
                if self.model_type == 'advanced_transgnn':
                    fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                    output = fused_features
                else:
                    output, series, prior, _ = self.model(input)
                    
                loss = torch.mean(criterion(input, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) find the threshold
            attens_energy = []
            print("计算阈值...")
            for i, (input_data, labels) in tqdm(enumerate(self.thre_loader), total=len(self.thre_loader),
                                              desc='计算阈值'):
                input = input_data.float().to(self.device)
                
                # 根据模型类型处理返回值
                if self.model_type == 'advanced_transgnn':
                    fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                    output = fused_features
                else:
                    output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
            print("Threshold :", thresh)

            # (3) evaluation on the test set
            test_labels = []
            attens_energy = []
            print("评估测试集...")
            for i, (input_data, labels) in tqdm(enumerate(self.thre_loader), total=len(self.thre_loader),
                                              desc='评估测试集'):
                input = input_data.float().to(self.device)
                
                # 根据模型类型处理返回值
                if self.model_type == 'advanced_transgnn':
                    fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                    output = fused_features
                else:
                    output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)

            pred = (test_energy > thresh).astype(int)

            gt = test_labels.astype(int)

            print("pred:   ", pred.shape)
            print("gt:     ", gt.shape)

            # detection adjustment
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)

            from sklearn.metrics import precision_recall_fscore_support
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                                  average='binary')
            print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision,
                    recall, f_score))

            return accuracy, precision, recall, f_score
            
        except Exception as e:
            print(f"测试过程中发生错误：{str(e)}")
            return None
