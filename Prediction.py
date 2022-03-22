import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Leaner import YanNet
from Dataset import High_Polymer_Material
from Draw import drawer
from Draw import split_nature



"""
nature_files列表存储户外数据文件
"""
data_path = (r'D:\Code\Polymeric_Material_Life_Time\Data')
input_file_path = os.path.join(data_path, 'Nomal_Data')
nature_files = ["GZ.csv", "HLR.csv", "LS.csv", "QD.csv", "QH.csv", "RQ.csv"]
for i in range(0, len(nature_files)):
    nature_files[i] = os.path.join(input_file_path, nature_files[i])

"""
划分原始数据，为画图做准备
第一次运行时使用，往后就不用了
"""
# split_nature()

"""
定义预测的循环
"""
def forecast_loop(dataloader, model, print_list):
    # size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    """
    第一个样本的前向损失是固定的，从第二个开始修改
    :param dataloader:
    :param model:
    :param print_list: 有两个作用，1.存储每一次的预测值用于打印图像；2.记录前向损失，用于拼接输入特征
    :return:
    """
    count = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            p = X.cpu().numpy()
            if (count > 0):
                p[0, 4] = print_list[count-1]
            p = torch.from_numpy(p)
            p = p.to(device)
            prediction = model(p).item()
            print_list[count] = prediction
            count += 1


"""
六个地区依次预测并绘图
"""
for file_name in nature_files:
    """
    1.读取数据
    2.定义dataloader
    3.获取任务名,得到对应权重文件
    4.实例化网络并载入对应权重,网络移动到GPU,并统一数据类型为double
    5.空的numpy数组print_p存储预测值
    6.预测
    7.调用drawer画图  //第一次画图前记得调用分割！！！！！
    
    
    销毁网络实例
    """
    datas = pd.read_csv(file_name)
    predict_data = High_Polymer_Material(datas)
    predict_dataloader = DataLoader(predict_data, batch_size=1, shuffle=False)

    file_name = os.path.split(file_name)[1]
    file_name = os.path.splitext(file_name)[0]
    # print(f"{file_name}\n-------------------------------")
    weight_file_name = 'Data/Nature_Weight/NN_{}.pth'.format(file_name)

    predictor = YanNet()
    predictor.load_state_dict(torch.load(weight_file_name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = predictor.to(device)
    predictor = predictor.double()

    print_p = np.zeros(len(predict_dataloader))

    forecast_loop(predict_dataloader, predictor, print_p)
    # print(print_p)
    """
    注意数据需要还原y' = y*total_nomal_y+total_min_y，
    这里直接给出
    total_nomal_y:0.066217957
    total_min_y:0.000302043
    """
    print_p = print_p*0.066217957+0.000302043

    drawer(file_name, print_p)




    del predictor




