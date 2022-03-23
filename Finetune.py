import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from Leaner import YanNet
from Dataset import High_Polymer_Material
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute_loss = nn.MSELoss()


def FineTune(arg, file_name, type):
    """
    :param file_name:需要微调的文件
    :return:
        train_p:
        test_p:
    """
    rtt = arg['ratio_train_test']
    fr = arg['finetune_rate']
    ft = arg['finetune_step']
    datas = pd.read_csv(file_name)
    # 包含初始样本一起打乱
    datas = shuffle(datas)
    # 定义训练集和测试集大小并进行划分
    total_sample_num = datas.shape[0]
    train_sample_num = int((total_sample_num / (rtt + 1)) * rtt)
    test_sample_num = total_sample_num - train_sample_num
    train_data = High_Polymer_Material(datas[: train_sample_num])
    test_data = High_Polymer_Material(datas[train_sample_num:])

    # 定义对应的DataLoader
    train_dataloader = DataLoader(train_data, batch_size=train_sample_num, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_sample_num, shuffle=True)

    # 获取任务名
    file_name = os.path.split(file_name)[1]
    file_name = os.path.splitext(file_name)[0]

    print(f"{file_name}\n-------------------------------")

    # 实例化网络对象
    finetune_model = YanNet()
    finetune_model.load_state_dict(torch.load('Data/Init_Weight/initialized_weights.pth'))
    finetune_model = finetune_model.to(device)
    finetune_model = finetune_model.double()

    # 定义优化器
    finetune_optim = torch.optim.SGD(finetune_model.parameters(), lr=fr)

    train_p = np.zeros(ft)
    test_p = np.zeros(ft)
    # 在任务上微调finetune_step次
    for t in np.arange(0, ft, 1):
        """
        训练
        """
        for batch, (X, y) in enumerate(train_dataloader):
            # 数据读到GPU
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            prediction = finetune_model(X)
            prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
            loss = compute_loss(prediction, y)

            finetune_optim.zero_grad()
            loss.backward()
            finetune_optim.step()

            loss_out = loss.item()
        print(f"{file_name}-train: {loss_out:>6f}")
        train_p[t] = loss_out
        """
        测试
        """
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                prediction = finetune_model(X)
                prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
                test_loss = compute_loss(prediction, y).item()

        print(f"{file_name}-test: {test_loss:>6f}")
        test_p[t] = test_loss

        """
        存储权重
        户外数据(共六个地点)的微调后权重存储在Data/Nature_Weight中
        """
        if (type == "nature"):
            weight_file_name = 'Data/Nature_Weight/NN_{}.pth'.format(file_name)
        else:
            weight_file_name = 'Data/Home_Weight/NN_{}.pth'.format(file_name)
        torch.save(finetune_model.state_dict(), weight_file_name)

    return train_p, test_p




print("Finetune-------------------------------")
ratio_train_test = 2
finetune_rate = 8*1e-4
finetune_step = 12

argments = {'ratio_train_test': ratio_train_test, 'finetune_step': finetune_step, 'finetune_rate': finetune_rate}

data_path = 'D:/Code/Polymeric_Material_Life_Time/Data'
input_file_path = os.path.join(data_path, 'Nomal_Data')
nature_files = ["GZ.csv", "HLR.csv", "LS.csv", "QD.csv", "QH.csv", "RQ.csv"]
# home_files = ["One.csv", "Two.csv", "Three.csv", "Four.csv", "Five.csv", "Six.csv", "Siven.csv", "Eight.csv", "Nine.csv"]
for i in range(0, len(nature_files)):
    nature_files[i] = os.path.join(input_file_path, nature_files[i])
# for i in range(0, len(home_files)):
#     home_files[i] = os.path.join(input_file_path, home_files[i])


for file in nature_files:
    train_painter, test_painter = FineTune(argments, file, "nature")
    """
    绘制meta的学习曲线
    """
    fig = plt.figure(figsize=(8, 8.5))
    ax_train = fig.add_subplot(1, 1, 1)
    ax_train.plot(train_painter, color='r', label='Train_Loss')
    ax_test = fig.add_subplot(1, 1, 1)
    ax_test.plot(test_painter, color='g', label='Test_Loss')
    plt.legend(loc='upper right')
    # 保存图片
    # 获取任务名
    task_name = os.path.split(file)[1]
    task_name = os.path.splitext(task_name)[0]
    plt.savefig('Data/Draw/Picture/Learning_Curve/{}_Learning_Curve.png'.format(task_name), dpi=400)

# for file in home_files:
#     FineTune(argments, file, "home")