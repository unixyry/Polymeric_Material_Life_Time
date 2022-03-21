import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from copy import deepcopy
from Leaner import YanNet
from Dataset import High_Polymer_Material



class Meta(nn.Module):
    def __init__(self, arg):
        super(Meta, self).__init__()

        self.learning_rate = arg['learning_rate']
        self.update_rate = arg['update_rate']
        self.epochs = arg['epochs']
        self.task_num = arg['task_num']
        self.learning_step = arg['learning_step']
        # self.update_step = arg['update_step']
        self.ratio_train_test = arg['ratio_train_test']

        self.net = YanNet()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task_optimizer = torch.optim.SGD(self.net.parameters(), lr = self.learning_rate)
        self.meta_optimizer = torch.optim.SGD(self.net.parameters(), lr = self.update_rate)
        self.compute_loss = nn.MSELoss()





    def learning(self, task_files):
        """
        1.learning是先进行内层循环模型在各个任务的训练集上运行，计算并记录各测试集的loss，再进行外层更新初始权重      //注意：最初的模型权重应当是统一随机化的，应当在main函数当中初始化
        2.在leaning阶段划分任务的训练和测试？在learing阶段打乱
        :param task_files:存储了各任务（数据文件）的绝对地址
        :return:

        finetune函数用于测试，即模型在某任务上通过学习得到的初始权重上微调数次
        """
        task_num = len(task_files)
        count = 1
        print('Using {} device'.format(self.device))

        """
        我认为不需要，因为外层循环只能更新一次，即，通过所有的测试集更新了初始权重之后，这些若干权重就没有使用意义了
        空的DataFrame,将各任务的test加入其中用于外层循环
        ！需要配套的索引
        """
        # test_datas = pd.DataFrame()

        """
        test_losses:numpy向量,初始为空，在每个任务的测试集计算出loss后记录下来
        """
        test_losses = np.array([])


        for file_name in task_files:
            """
            按ratio_train_test(2:1)的比例划分出该任务的训练和测试
            total_sample_num:该任务的总样本数
            train_sample_num:该任务的训练样本数,用于内层循环
            test_sample_num:该任务的测试样本数,用于外层循环
            """
            datas = pd.read_csv(file_name)
            # 包含初始样本一起打乱
            datas = shuffle(datas)
            total_sample_num = datas.shape[0]
            train_sample_num = int((total_sample_num/(self.ratio_train_test+1))*self.ratio_train_test)
            test_sample_num = total_sample_num - train_sample_num
            train_data = High_Polymer_Material(datas[: train_sample_num])
            test_data = High_Polymer_Material(datas[train_sample_num:])
            train_dataloader = DataLoader(train_data, batch_size=train_sample_num, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=test_sample_num, shuffle=True)

            train_model = YanNet()
            train_model.load_state_dict(torch.load('Data/Init_Weight/initialized_weights.pth'))
            train_model = train_model.to(self.device)
            train_model = train_model.double()


            """
            一个任务通过训练集更新learning_rate次权重
            """
            for i in np.arange(0, self.learning_rate, 1):
                for batch, (X, y) in enumerate(train_dataloader):
                    # 数据读到GPU
                    X, y = X.to(self.device), y.to(self.device)

                    # Compute prediction and loss
                    prediction = train_model(X)
                    prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
                    loss = self.compute_loss(prediction, y)

                    # 添加到向量中
                    # p_train[num] = loss

                    # Backpropagation
                    self.task_optimizer.zero_grad()
                    loss.backward()
                    # update weights and bias
                    self.task_optimizer.step()

                    loss_out = loss.item()
                    print(f"{file_name}-train-MSEloss: {loss_out:>6f}")  #file_name需要改一下

            """
            此时权重已更新
            在测试集上计算一次loss，并记录到test_losses中
            """
            for batch, (X, y) in enumerate(test_dataloader):
                # 数据读到GPU
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction and loss
                prediction = train_model(X)
                prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
                loss = self.compute_loss(prediction, y)

                loss_out = loss.item()
                print(f"{file_name}-test-MSEloss: {loss_out:>6f}")

            test_losses = np.append(test_losses, loss_out)
            del train_model












