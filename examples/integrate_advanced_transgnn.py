import os
import sys
import torch
import argparse
import numpy as np
import time
from torch.backends import cudnn

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.AdvancedAnomalyTransGNN import AdvancedAnomalyTransGNN
from utils.utils import *
from data_factory.data_loader import get_loader_segment


class EarlyStopping:
    """早停机制"""
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


class AdvancedTransGNNSolver(object):
    """用于训练和测试高级AnomalyTransGNN模型的求解器"""
    
    def __init__(self, config):
        self.__dict__.update(config)
        
        # 加载数据
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                           mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                          mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                          mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                          mode='thre', dataset=self.dataset)
        
        # 构建模型
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
    
    def build_model(self):
        """构建高级AnomalyTransGNN模型"""
        self.model = AdvancedAnomalyTransGNN(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,        # Transformer模型维度
            n_heads=self.n_heads,         # Transformer多头注意力的头数
            e_layers=self.e_layers,        # Transformer编码器层数
            d_ff=self.d_ff,          # Transformer前馈网络的隐藏层维度
            dropout=self.dropout,       # Dropout概率
            activation=self.activation,  # 激活函数类型
            output_attention=True,  # 输出注意力权重
            gnn_hidden=self.gnn_hidden,     # GNN隐藏层维度
            gnn_layers=self.gnn_layers,      # GNN层数
            gnn_type=self.gnn_type,    # GNN类型
            gnn_heads=self.gnn_heads,       # GNN注意力头数
            k_neighbors=self.k_neighbors,     # 构建图时的k近邻数量
            fusion_method=self.fusion_method,  # 特征融合方法
            dynamic_graph=self.dynamic_graph,  # 是否使用动态图构建
            residual=self.residual      # 是否使用残差连接
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        if torch.cuda.is_available():
            self.model.cuda()
    
    def train(self):
        """训练模型"""
        print("======================TRAIN MODE=======================")
        time_now = time.time()
        
        # 早停机制
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = []
            
            for i, (input_data, _) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                
                # 前向传播
                fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                
                # 计算损失
                loss = self.model.calculate_loss(input, fused_features, anomaly_score, series, prior)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss.append(loss.item())
            
            # 验证
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} {4:.7f}".format(
                epoch + 1, len(self.train_loader), np.mean(train_loss), vali_loss1, vali_loss2))
            
            # 早停检查
            early_stopping(vali_loss1, vali_loss2, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 调整学习率
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        
        print("Training time:", time.time() - time_now)
        return self.model

    def vali(self, vali_loader):
        """验证模型"""
        self.model.eval()
        loss_1 = []
        loss_2 = []
        
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                
                # 前向传播
                fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
                
                # 计算损失
                loss = self.model.calculate_loss(input, fused_features, anomaly_score, series, prior)
                
                # 重构损失
                loss_1.append(torch.mean(self.criterion(fused_features, input)).item())
                # 异常分数损失
                loss_2.append(torch.mean(anomaly_score).item())
        
        return np.mean(loss_1), np.mean(loss_2)

    def test(self):
        """测试模型"""
        print("======================TEST MODE=======================")
        
        # 加载预训练模型
        if self.pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint.pth')))
        
        self.model.eval()
        temperature = 50
        
        # 计算阈值
        print("Calculating threshold...")
        attens_energy = []
        for i, (input_data, _) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            
            # 前向传播
            fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
            
            # 计算异常分数
            attens_energy.append(anomaly_score.detach().cpu().numpy())
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        threshold = np.percentile(attens_energy, 100 - self.anormly_ratio)
        print("Threshold :", threshold)
        
        # 测试
        print("Testing...")
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            
            # 前向传播
            fused_features, anomaly_score, series, prior, sigmas, edge_index, edge_weight = self.model(input)
            
            # 计算异常分数
            attens_energy.append(anomaly_score.detach().cpu().numpy())
            test_labels.append(labels)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        
        # 计算评估指标
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_scores = test_energy
        
        # 根据阈值判断异常
        pred = (test_scores > threshold).astype(int)
        gt = test_labels.astype(int)
        
        # 计算评估指标
        print("Threshold :", threshold)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        
        # 计算精确度、召回率、F1分数
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        
        print("\nTest Results:")
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f_score))
        
        # 可视化结果（如果需要）
        if self.visualize:
            self.visualize_results(test_scores, gt, threshold)
    
    def visualize_results(self, scores, labels, threshold):
        """可视化测试结果"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建结果目录
            if not os.path.exists('./results'):
                os.makedirs('./results')
            
            plt.figure(figsize=(15, 8))
            plt.plot(scores, label='Anomaly Score')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            
            # 标记真实异常点
            anomaly_indices = np.where(labels == 1)[0]
            plt.scatter(anomaly_indices, scores[anomaly_indices], color='red', marker='x', s=50, label='True Anomaly')
            
            # 标记预测异常点
            pred_anomaly_indices = np.where(scores > threshold)[0]
            plt.scatter(pred_anomaly_indices, scores[pred_anomaly_indices], color='green', marker='o', s=30, alpha=0.5, label='Predicted Anomaly')
            
            plt.title(f'Anomaly Detection Results - {self.dataset}')
            plt.xlabel('Time')
            plt.ylabel('Anomaly Score')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'./results/{self.dataset}_advanced_transgnn_results.png')
            print(f"结果已保存到 ./results/{self.dataset}_advanced_transgnn_results.png")
        except Exception as e:
            print(f"可视化失败: {e}")


def adjust_learning_rate(optimizer, epoch, lr_):
    """调整学习率"""
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('学习率调整为：', lr)


def main():
    """主函数"""
    # 参数解析
    parser = argparse.ArgumentParser(description='高级图神经网络异常检测模型')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='SMD', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='./dataset/SMD', help='数据集路径')
    parser.add_argument('--win_size', type=int, default=100, help='窗口大小')
    parser.add_argument('--input_c', type=int, default=38, help='输入特征维度')
    parser.add_argument('--output_c', type=int, default=38, help='输出特征维度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    # Transformer参数
    parser.add_argument('--d_model', type=int, default=64, help='Transformer模型维度')
    parser.add_argument('--n_heads', type=int, default=4, help='Transformer多头注意力的头数')
    parser.add_argument('--e_layers', type=int, default=3, help='Transformer编码器层数')
    parser.add_argument('--d_ff', type=int, default=128, help='Transformer前馈网络的隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数类型')
    
    # GNN参数
    parser.add_argument('--gnn_type', type=str, default='gatv2', choices=['gatv2', 'graph_transformer'], help='GNN类型')
    parser.add_argument('--gnn_hidden', type=int, default=32, help='GNN隐藏层维度')
    parser.add_argument('--gnn_layers', type=int, default=2, help='GNN层数')
    parser.add_argument('--gnn_heads', type=int, default=4, help='GNN注意力头数')
    parser.add_argument('--k_neighbors', type=int, default=5, help='构建图时的k近邻数量')
    parser.add_argument('--dynamic_graph', action='store_true', help='是否使用动态图构建')
    parser.add_argument('--residual', action='store_true', help='是否使用残差连接')
    
    # 融合参数
    parser.add_argument('--fusion_method', type=str, default='weighted', choices=['concat', 'add', 'weighted'], help='特征融合方法')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--anormly_ratio', type=float, default=4.0, help='异常比例')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='模型保存路径')
    
    # 测试参数
    parser.add_argument('--pretrained_model', action='store_true', help='是否使用预训练模型')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    
    args = parser.parse_args()
    
    # 创建模型保存目录
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        cudnn.benchmark = True
    
    # 创建求解器
    solver = AdvancedTransGNNSolver(vars(args))
    
    # 根据模式运行
    if args.mode == 'train':
        solver.train()
        solver.test()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    main()