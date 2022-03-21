import torch
from torch import nn
from collections import OrderedDict
import numpy as np



"D1是5x5的全连接层，激活函数设置为ReLU"
class D1(nn.Module):
    def __init__(self):
        super(D1, self).__init__()

        self.d1 = nn.Sequential(OrderedDict([
            ('dense_1', nn.Linear(5, 5)),
            ('relu', nn.ReLU())
        ]))

    def forward(self, input):
        output = self.d1(input)
        return output


"D2是5x1的全连接层"
class D2(nn.Module):
    def __init__(self):
        super(D2, self).__init__()

        self.d2 = nn.Sequential(OrderedDict([
            ('dense_2', nn.Linear(5, 1))
        ]))

    def forward(self, input):
        output = self.d2(input)
        return output


"""
输入：5维
输出：1维
"""
class YanNet(nn.Module):
    def __init__(self):
        super(YanNet, self).__init__()

        self.d1 = D1()
        self.d2 = D1()
        self.d3 = D2()


    def inisialize(self):
        """
        先实例化，再调用该方法
        model = YanNet()
        model.inisialize()
        :return:
        """
        for w in self.modules():
            # isinstance(a, b) 函数来判断一个对象a是否是一个已知的类型b,a可以是b的子类
            if isinstance(w, nn.Linear):
                # normal_(),用标准正态分布填充tensor
                nn.init.normal_(w.weight.data, mean=0.0, std=1.0)



    def forward(self, inputs):
        outputs = self.d1(inputs)
        outputs = self.d2(outputs)
        outputs = self.d3(outputs)
        return outputs



# model = YanNet()
# print("随机初始化之前：----------------------")
# for p in model.parameters():
#     print(p)
# model.inisialize()
# print("随机初始化之后：----------------------")
# for p in model.parameters():
#     print(p)
