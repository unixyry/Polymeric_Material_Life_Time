import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

class High_Polymer_Material(Dataset):
    def __init__(self, file_datas, transform=None, target_transform=None):
        # self.datas = pd.read_csv(file_name)
        self.datas = file_datas.values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        data = self.datas[index, :5]      # 前五个为特征
        label = self.datas[index, 5]      # 第五个为标签
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

