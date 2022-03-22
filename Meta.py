import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
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
        self.update_step = arg['update_step']
        self.ratio_train_test = arg['ratio_train_test']
        self.finetune_step = arg['finetune_step']
        self.finetune_rate = arg['finetune_rate']

        # self.net = YanNet()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.task_optimizer = torch.optim.SGD(self.net.parameters(), lr = self.learning_rate)
        # self.meta_optimizer = torch.optim.SGD(self.net.parameters(), lr = self.update_rate)
        # self.finetune_optim = torch.optim.SGD(self.net.parameters(), lr=self.finetune_rate)
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

        """
        我认为不需要，因为外层循环只能更新一次，即，通过所有的测试集更新了初始权重之后，这些若干权重就没有使用意义了
        空的DataFrame,将各任务的test加入其中用于外层循环
        ！需要配套的索引
        """
        # test_datas = pd.DataFrame()

        """
        test_losses:tensor,初始(count==0)为第一个任务的loss，在之后每个任务(count>=0)的测试集计算出loss后相加test_losses+=loss
        count用于计数
        """
        # test_losses = np.empty([])
        # test_losses = torch.tensor([1]).to(self.device)
        count = 0


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
            # 定义训练集和测试集大小并进行划分
            total_sample_num = datas.shape[0]
            train_sample_num = int((total_sample_num/(self.ratio_train_test+1))*self.ratio_train_test)
            test_sample_num = total_sample_num - train_sample_num
            train_data = High_Polymer_Material(datas[: train_sample_num])
            test_data = High_Polymer_Material(datas[train_sample_num:])
            # 定义对应的DataLoader
            train_dataloader = DataLoader(train_data, batch_size=train_sample_num, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=test_sample_num, shuffle=True)

            # 实例化网络后，加载初始权重，注意，对于不同的任务，其训练前的初始权重都是一致的
            train_model = YanNet()
            train_model.load_state_dict(torch.load('Data/Init_Weight/initialized_weights.pth'))

            # 定义优化器
            task_optimizer = torch.optim.SGD(train_model.parameters(), lr=self.learning_rate)


            # 模型移动到GPU
            train_model = train_model.to(self.device)
            # 数据类型统一为double
            train_model = train_model.double()

            # 获取任务名，即文件名(不含后缀名）
            """
            os.path.split(file_name)返回一个二元组
            os.path.split(file_name)[0]
            os.path.split(file_name)[1]就是目标文件名
            """
            """
            os.path.splitext(file_name)返回一个二元组
            os.path.splitext(file_name)[0]是路径加文件名
            os.path.splitext(file_name)[1]就是目标文件的后缀名，例如.txt
            """
            file_name = os.path.split(file_name)[1]
            file_name = os.path.splitext(file_name)[0]


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
                    # self.task_optimizer.zero_grad()
                    task_optimizer.zero_grad()
                    loss.backward()
                    # update weights and bias
                    # self.task_optimizer.step()
                    task_optimizer.step()

                    loss_out = loss.item()
                    print(f"{file_name}-train: {loss_out:>6f}")

            """
            此时权重已更新
            在测试集上计算一次loss，并记录到test_losses中
            """
            with torch.no_grad():
                for batch, (X, y) in enumerate(test_dataloader):
                    # 数据读到GPU
                    X, y = X.to(self.device), y.to(self.device)

                    # Compute prediction and loss
                    prediction = train_model(X)
                    prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
                    loss = self.compute_loss(prediction, y)

                    loss_out = loss.item()
                    print(f"{file_name}-test: {loss_out:>6f}")


            if (count == 0):
                test_losses = loss
            else:
                test_losses += loss


            # 销毁该实例
            del train_model
            count += 1

        """
        所有任务已跑完，各个测试集的loss也记录在test_losses中
        更新初始权重
        注意：
        1.loss需要取平均，即test_losses取和除以数量
        2.权重只更新一次
        3.保存权重
        """
        update_model = YanNet()
        update_model.load_state_dict(torch.load('Data/Init_Weight/initialized_weights.pth'))
        update_model = update_model.to(self.device)
        update_model = update_model.double()

        # 定义优化器
        meta_optimizer = torch.optim.SGD(update_model.parameters(), lr=self.update_rate)

        # 计算各测试集loss的平均值
        """
        np.sum(axis = 1) 每一行的元素相加   //jupyter中一维向量也可以，但pycharm中一维向量np.sum()        
        """
        # print(f"测试损失是{test_losses}")
        loss = test_losses/self.task_num
        loss = Variable(loss, requires_grad = True)


        # 更新初始权重
        # self.meta_optimizer.zero_grad()
        meta_optimizer.zero_grad()
        loss.backward()
        # self.meta_optimizer.step()
        meta_optimizer.step()

        # 保存初始权重
        torch.save(update_model.state_dict(), 'Data/Init_Weight/initialized_weights.pth')

        # 清除对象
        del update_model



    def FineTune(self, file_name):
        """
        :param file_name:需要微调的文件
        :return:
        """
        datas = pd.read_csv(file_name)
        # 包含初始样本一起打乱
        datas = shuffle(datas)
        # 定义训练集和测试集大小并进行划分
        total_sample_num = datas.shape[0]
        train_sample_num = int((total_sample_num / (self.ratio_train_test + 1)) * self.ratio_train_test)
        test_sample_num = total_sample_num - train_sample_num
        train_data = High_Polymer_Material(datas[: train_sample_num])
        test_data = High_Polymer_Material(datas[train_sample_num:])

        # 定义对应的DataLoader
        train_dataloader = DataLoader(train_data, batch_size=train_sample_num, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=test_sample_num, shuffle=True)

        # 获取任务名
        file_name = os.path.split(file_name)[1]
        file_name = os.path.splitext(file_name)[0]

        # 实例化网络对象
        finetune_model = YanNet()
        finetune_model.load_state_dict(torch.load('Data/Init_Weight/initialized_weights.pth'))
        finetune_model = finetune_model.to(self.device)
        finetune_model = finetune_model.double()

        # 定义优化器
        finetune_optim = torch.optim.SGD(finetune_model.parameters(), lr=self.finetune_rate)


        # 在任务上微调finetune_step次
        for t in np.arange(0, self.finetune_step, 1):
            """
            训练
            """
            for batch, (X, y) in enumerate(train_dataloader):
                # 数据读到GPU
                X, y = X.to(self.device), y.to(self.device)
                # Compute prediction and loss
                prediction = finetune_model(X)
                prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
                loss = self.compute_loss(prediction, y)

                finetune_optim.zero_grad()
                loss.backward()
                finetune_optim.step()

                loss_out = loss.item()
                print(f"{file_name}-train: {loss_out:>6f}")

            """
            测试
            """
            with torch.no_grad():
                for batch, (X, y) in enumerate(test_dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    prediction = finetune_model(X)
                    prediction = prediction.squeeze(-1)  # 保证预测和标签维度一致
                    test_loss = self.compute_loss(prediction, y).item()

            # p_test[num] = test_loss
            # test_loss /= num_batches
            print(f"{file_name}-test: {test_loss:>6f}")

            """
            存储权重
            户外数据(共六个地点)的微调后权重存储在Data/Nature_Weight中
            """
            weight_file_name = 'Data/Nature_Weight/NN_{}.pth'.format(file_name)
            torch.save(finetune_model.state_dict(), weight_file_name)






















