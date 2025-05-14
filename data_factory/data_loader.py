import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        # 获取训练数据的特征数量
        n_features = data.shape[1]
        print(f"训练数据特征数量: {n_features}")
        
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        
        # 检查测试数据的特征数量
        print(f"测试数据特征数量: {test_data.shape[1]}")
        
        # 如果测试数据特征数量与训练数据不匹配，进行填充
        if test_data.shape[1] < n_features:
            # 添加零列以匹配训练数据的特征数量
            padding = np.zeros((test_data.shape[0], n_features - test_data.shape[1]))
            test_data = np.hstack((test_data, padding))
            print(f"已填充测试数据至 {test_data.shape[1]} 个特征")

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            # 确保测试标签的长度与窗口大小一致，避免批处理时张量大小不匹配
            test_label = self.test_labels[index:index + self.win_size]
            # 如果标签长度不足，使用零填充
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[index:index + self.win_size]), np.float32(test_label)
        else:
            # 同样确保thre模式下标签长度一致
            idx = index // self.step * self.win_size
            test_label = self.test_labels[idx:idx + self.win_size]
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(test_label)


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            # 确保测试标签的长度与窗口大小一致，避免批处理时张量大小不匹配
            test_label = self.test_labels[index:index + self.win_size]
            # 如果标签长度不足，使用零填充
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[index:index + self.win_size]), np.float32(test_label)
        else:
            # 同样确保thre模式下标签长度一致
            idx = index // self.step * self.win_size
            test_label = self.test_labels[idx:idx + self.win_size]
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(test_label)


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            # 确保测试标签的长度与窗口大小一致，避免批处理时张量大小不匹配
            test_label = self.test_labels[index:index + self.win_size]
            # 如果标签长度不足，使用零填充
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[index:index + self.win_size]), np.float32(test_label)
        else:
            # 同样确保thre模式下标签长度一致
            idx = index // self.step * self.win_size
            test_label = self.test_labels[idx:idx + self.win_size]
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(test_label)

class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        # 获取训练数据的特征数量
        n_features = data.shape[1]
        print(f"训练数据特征数量: {n_features}")
        
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        
        # 检查测试数据的特征数量
        print(f"测试数据特征数量: {test_data.shape[1]}")
        
        # 如果测试数据特征数量与训练数据不匹配，进行填充
        if test_data.shape[1] < n_features:
            # 添加零列以匹配训练数据的特征数量
            padding = np.zeros((test_data.shape[0], n_features - test_data.shape[1]))
            test_data = np.hstack((test_data, padding))
            print(f"已填充测试数据至 {test_data.shape[1]} 个特征")

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            # 确保测试标签的长度与窗口大小一致，避免批处理时张量大小不匹配
            test_label = self.test_labels[index:index + self.win_size]
            # 如果标签长度不足，使用零填充
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[index:index + self.win_size]), np.float32(test_label)
        else:
            # 同样确保thre模式下标签长度一致
            idx = index // self.step * self.win_size
            test_label = self.test_labels[idx:idx + self.win_size]
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(test_label)




class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            # 确保测试标签的长度与窗口大小一致，避免批处理时张量大小不匹配
            test_label = self.test_labels[index:index + self.win_size]
            # 如果标签长度不足，使用零填充
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[index:index + self.win_size]), np.float32(test_label)
        else:
            # 同样确保thre模式下标签长度一致
            idx = index // self.step * self.win_size
            test_label = self.test_labels[idx:idx + self.win_size]
            if len(test_label) < self.win_size:
                padding = np.zeros((self.win_size - len(test_label), test_label.shape[1]))
                test_label = np.vstack((test_label, padding))
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(test_label)


def get_loader_segment(data_path, batch_size, win_size, mode='train', dataset='SWaT', num_workers=4, pin_memory=True, persistent_workers=True):
    # 设置默认步长
    step = 1  # 默认步长为1
    
    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR':
        dataset = UCRSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL':
        dataset = UCR_MSLSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP':
        dataset = UCR_SMAPSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_PSM':
        dataset = UCR_PSMSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SWaT':
        dataset = UCR_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMD':
        dataset = UCR_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP':
        dataset = UCR_MSL_SMAPSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_PSM':
        dataset = UCR_MSL_PSMSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SWaT':
        dataset = UCR_MSL_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMD':
        dataset = UCR_MSL_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_PSM':
        dataset = UCR_SMAP_PSMSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_SWaT':
        dataset = UCR_SMAP_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_SMD':
        dataset = UCR_SMAP_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_PSM_SWaT':
        dataset = UCR_PSM_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_PSM_SMD':
        dataset = UCR_PSM_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SWaT_SMD':
        dataset = UCR_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_PSM':
        dataset = UCR_MSL_SMAP_PSMSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_SWaT':
        dataset = UCR_MSL_SMAP_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_SMD':
        dataset = UCR_MSL_SMAP_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_PSM_SWaT':
        dataset = UCR_MSL_PSM_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_PSM_SMD':
        dataset = UCR_MSL_PSM_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SWaT_SMD':
        dataset = UCR_MSL_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_PSM_SWaT':
        dataset = UCR_SMAP_PSM_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_PSM_SMD':
        dataset = UCR_SMAP_PSM_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_SWaT_SMD':
        dataset = UCR_SMAP_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_PSM_SWaT_SMD':
        dataset = UCR_PSM_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_PSM_SWaT':
        dataset = UCR_MSL_SMAP_PSM_SWaTSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_PSM_SMD':
        dataset = UCR_MSL_SMAP_PSM_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_SWaT_SMD':
        dataset = UCR_MSL_SMAP_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_PSM_SWaT_SMD':
        dataset = UCR_MSL_PSM_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_SMAP_PSM_SWaT_SMD':
        dataset = UCR_SMAP_PSM_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    elif dataset == 'UCR_MSL_SMAP_PSM_SWaT_SMD':
        dataset = UCR_MSL_SMAP_PSM_SWaT_SMDSegLoader(data_path, win_size=win_size, step=step, mode=mode)
    else:
        raise ValueError(f"未知的数据集: {dataset}")

    shuffle = True if mode == 'train' else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                     pin_memory=pin_memory, persistent_workers=persistent_workers)
