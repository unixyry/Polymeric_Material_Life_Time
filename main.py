import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from Meta import Meta
from Leaner import YanNet

"""
所有输入数据文件存在'Nomal_Data'文件夹下
地址使用Linux形式,地址划分使用'/'
os.listdir()会将该文件夹下的所有的文件名放入到列表files中
files存储了所有15个任务的绝对地址
task_num是用于元学习训练阶段的任务数，设为12
每个epoch打乱files,取前12个作为元学习的任务task_files
"""
data_path = 'D:/Code/Polymeric_Material_Life_Time/Data'
input_file_path = os.path.join(data_path, 'Nomal_Data')
files = os.listdir(input_file_path)
for i in range(0, len(files)):
    files[i] = os.path.join(input_file_path, files[i])
total_task_num = 15
task_num = 12




"""
1、有GPU则优先使用GPU               
2、损失函数compute_loss使用MSE      
3、优化器optimizer使用SGD        
"""



"""
配置一些相关参数
1、学习率learning_rate定为8*1e-2，即0.08   //学习率是指内层循环的α，用于每个具体任务的训练更新
2、更新率update_rate定为1*1e-3，即0.001   //更新率是指外层循环的β，在整体上用于更新下一次权重的初值
3、训练轮数epochs定为40次
4、学习轮数learning_step定为1次     //学习轮数是指针对每个具体任务，模型权重更新learing_step次
5、更新轮数update_step定为1次       //更新轮数是指在整体上得到下一次权重初值学习update_step次，只能是1次
6、每个任务内训练数据和测试数据比例ratio_train_test定为2(2:1)
7、微调轮数finetune_step尚未定义    //微调轮数finetune_step是指，模型在某任务上通过已学习到的初始权重学习的次数
8、微调率finetune_rate定为5*1e-3，即0.0001   //微调率finetune_rate是指，模型在某任务上通过已学习到的初始权重学习时的步长

参数统一存进字典arg中，传入Meta
"""

learning_rate = 8*1e-2
update_rate = 1*1e-3
finetune_rate = 1*1e-4
epochs = 1000
learning_step = 1
update_step = 1    #注意，外层循环的更新只能迭代一步
finetune_step = 10
ratio_train_test = 2

arg = {'learning_rate': learning_rate, 'update_rate': update_rate, 'epochs': epochs, 'task_num': task_num,
       'learning_step': learning_step, 'update_step': update_step, 'ratio_train_test': ratio_train_test,
       'finetune_step': finetune_step, 'finetune_rate': finetune_rate}

if __name__ == '__main__':
    print("hi")
    """
    确保在MAML的最开始时，模型的初始权重在各任务上有统一的权重，存入Init_Weight文件夹中
    init_leaner只有一个作用，就是统一初始化权重，并将权重initialization保存到Init_Weight文件夹中
    """
    init_leaner = YanNet()
    init_leaner.inisialize()
    init_wight_path = os.path.join(data_path, 'Init_Weight')
    # 仅保存模型权重，加载时需要先实例化，详见Define_Own_Dataloader
    torch.save(init_leaner.state_dict(), os.path.join(init_wight_path, 'initialized_weights.pth'))
    # 删除该对象
    del init_leaner


    """
    MAML阶段
    """
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # 打乱打乱files,取前12个作为元学习的任务task_files
        random.shuffle(files)
        task_files = files[ :12]
        maml = Meta(arg)
        maml.learning(task_files)

    """
    Finetune阶段
    """
    maml.FineTune("Data/Nomal_Data/GZ.CSV")


    print("Done!")


