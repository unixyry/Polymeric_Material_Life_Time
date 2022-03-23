import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

from Maml import Meta
from Leaner import YanNet

warnings.filterwarnings("ignore")
"""
input_file指向'Total.csv'，存储了全部的数据
nature_files列表存储户外数据文件
"""
data_path = 'D:/Code/Polymeric_Material_Life_Time/Data'
input_file_path = os.path.join(data_path, 'Nomal_Data')
input_file = os.path.join(input_file_path, 'Total.csv')

nature_files = ["GZ.csv", "HLR.csv", "LS.csv", "QD.csv", "QH.csv", "RQ.csv"]
for i in range(0, len(nature_files)):
    nature_files[i] = os.path.join(input_file_path, nature_files[i])

"""
配置一些相关参数
0、元任务数task_num定为10
1、学习率learning_rate定为2*1e-4，即0.0002   //学习率是指内层循环的α，用于每个具体任务的训练更新
2、更新率update_rate定为5*1e-4，即0.0005   //更新率是指外层循环的β，在整体上用于更新下一次权重的初值
3、训练轮数epochs定为200次
4、学习轮数learning_step定为4次     //学习轮数是指针对每个具体任务，模型权重更新learing_step次
5、更新轮数update_step定为1次       //更新轮数是指在整体上得到下一次权重初值学习update_step次，只能是1次
6、每个任务内训练数据和测试数据比例ratio_train_test定为1(1:1)
7、微调轮数finetune_step定为6次    //微调轮数finetune_step是指，模型在某任务上通过已学习到的初始权重学习的次数
8、微调率finetune_rate定为8*1e-2，即0.08   //微调率finetune_rate是指，模型在某任务上通过已学习到的初始权重学习时的步长

参数统一存进字典arg中，传入Meta
"""
task_num = 10
learning_rate = 2*1e-4
update_rate = 5*1e-4
finetune_rate = 8*1e-2
epochs = 3940
learning_step = 4
update_step = 1    # 注意，外层循环的更新只能迭代一步
finetune_step = 6
ratio_train_test = 2

arg = {'learning_rate': learning_rate, 'update_rate': update_rate, 'epochs': epochs, 'task_num': task_num,
       'learning_step': learning_step, 'update_step': update_step, 'ratio_train_test': ratio_train_test,
       'finetune_step': finetune_step, 'finetune_rate': finetune_rate}


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
train_p = np.zeros(epochs)
test_p = np.zeros(epochs)
for t in range(epochs):
    if (((t+1)%100)==0):
        print(f"Epoch {t + 1}\n-------------------------------")
    # 打乱打乱files,取前12个作为元学习的任务task_files
    maml = Meta(arg, t)
    train_loss, test_loss, stop = maml.learning(input_file)
    train_p[t] = train_loss
    test_p[t] = test_loss
    if (stop):
        break

"""
绘制meta的学习曲线
"""
fig = plt.figure(figsize = (8,8.5))
ax_train = fig.add_subplot(1, 1, 1)
ax_train.plot(train_p, color = 'r', label = 'Train_Loss')
ax_test = fig.add_subplot(1, 1, 1)
ax_test.plot(test_p, color = 'g', label = 'Test_Loss')
plt.legend(loc = 'upper right')
#保存图片
plt.savefig('Data/Draw/Picture/Learning_Curve/Meta_Learning_Curve.png', dpi = 400)



"""
微调阶段请运行Finetune
"""

"""
预测阶段请运行Prediction
"""

print("Done!")