import os
import argparse  # 用于解析命令行参数

from torch.backends import cudnn  # 提高训练速度的工具
from utils.utils import *  # 导入自定义工具函数

from solver import Solver  # 导入 Solver 类，用于训练和测试模型


def str2bool(v):
    """
    将字符串转换为布尔值。
    :param v: 输入字符串
    :return: 转换后的布尔值
    """
    return v.lower() in ('true')


def main(config):
    """
    主函数，负责根据配置执行训练或测试。
    :param config: 配置参数
    """
    cudnn.benchmark = True  # 启用 cuDNN 的自动优化，提升 GPU 训练速度

    # 如果模型保存路径不存在，则创建
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)

    # 初始化 Solver 对象，传入配置参数
    solver = Solver(vars(config))

    # 根据模式选择训练或测试
    if config.mode == 'train':
        solver.train()  # 执行训练
    elif config.mode == 'test':
        solver.test()  # 执行测试

    return solver  # 返回 Solver 对象


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数及其默认值
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率")
    parser.add_argument('--num_epochs', type=int, default=10, help="训练的总轮数")
    parser.add_argument('--k', type=int, default=3, help="模型的超参数 k")
    parser.add_argument('--win_size', type=int, default=100, help="时间窗口大小")
    parser.add_argument('--input_c', type=int, default=38, help="输入特征维度")
    parser.add_argument('--output_c', type=int, default=38, help="输出特征维度")
    parser.add_argument('--batch_size', type=int, default=1024, help="批量大小")
    parser.add_argument('--force_batch_size', type=int, default=None, help="强制使用指定的批量大小，禁用自动调整")
    parser.add_argument('--pretrained_model', type=str, default=None, help="预训练模型的路径")
    parser.add_argument('--dataset', type=str, default='SWaT', help="数据集名称")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="运行模式：训练或测试")
    parser.add_argument('--data_path', type=str, default='./dataset/SWaT', help="数据集路径")
    parser.add_argument('--model_save_path', type=str, default='checkpoints', help="模型保存路径")
    parser.add_argument('--anormly_ratio', type=float, default=4.00, help="异常比例")
    
    # 添加模型结构参数
    parser.add_argument('--d_model', type=int, default=512, help="模型隐藏层维度")
    parser.add_argument('--e_layers', type=int, default=3, help="编码器层数")
    parser.add_argument('--n_heads', type=int, default=8, help="注意力头数")
    parser.add_argument('--d_ff', type=int, default=512, help="前馈网络维度")
    
    # 添加AdvancedAnomalyTransGNN模型相关参数
    parser.add_argument('--model_type', type=str, default='advanced_transgnn', help="模型类型：transformer、transgnn或advanced_transgnn")
    parser.add_argument('--gnn_type', type=str, default='gatv2', help="GNN类型：gcn、gat、gatv2或graph_transformer")
    parser.add_argument('--gnn_hidden', type=int, default=256, help="GNN隐藏层维度")
    parser.add_argument('--gnn_layers', type=int, default=2, help="GNN层数")
    parser.add_argument('--gnn_heads', type=int, default=4, help="GNN注意力头数")
    parser.add_argument('--k_neighbors', type=int, default=5, help="构建图时的邻居数量")
    parser.add_argument('--fusion_method', type=str, default='weighted', help="特征融合方法：concat、add或weighted")
    parser.add_argument('--dynamic_graph', type=str2bool, default=True, help="是否使用动态图构建")
    parser.add_argument('--residual', type=str2bool, default=True, help="是否使用残差连接")
    parser.add_argument('--no_time_limit', type=str2bool, default=False, help="是否禁用训练时间限制（默认2小时）")

    # 解析命令行参数
    config = parser.parse_args()

    # 打印所有配置参数
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # 调用主函数
    main(config)