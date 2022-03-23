import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings

#令Numpy不以科学计数法计数
np.set_printoptions(suppress=True)
# 忽略wanring
warnings.filterwarnings("ignore")
path = 'Data/Draw'


def split_nature():
    """
    file_path: 'Data/Draw/Data'，所有数据型文件都存储在这个文件下，包括未划分的'new_data.xlsx'，和已划分的
    file_name: 'Data/Draw/Data/new_data.xlsx'
    :return:
    """
    file_path = os.path.join(path, 'Data')
    file_name = os.path.join(file_path, 'new_data.xlsx')
    xlsx = pd.ExcelFile(file_name)
    frame_natural = pd.read_excel(xlsx, '户外损失')
    # 将dataframe转为numpy，便于处理数据
    dataset_natural = frame_natural.values
    # 拼接
    draw_natural = np.stack((dataset_natural[:, 2], dataset_natural[:, 3]), axis=1)
    # 切割
    QH = draw_natural[0:7, :]
    GZ = draw_natural[7:14, :]
    RQ = draw_natural[14:21, :]
    LS = draw_natural[21:28, :]
    QD = draw_natural[28:35, :]
    HLR = draw_natural[35:42, :]
    # 保存
    np.savetxt(os.path.join(file_path, 'QH.csv'), QH, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'GZ.csv'), GZ, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'RQ.csv'), RQ, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'LS.csv'), LS, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'QD.csv'), QD, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'HLR.csv'), HLR, fmt="%f", delimiter=",")


def split_home():
    file_path = os.path.join(path, 'Data')
    file_name = os.path.join(file_path, 'new_data.xlsx')
    xlsx = pd.ExcelFile(file_name)
    frame_home = pd.read_excel(xlsx, '室内加速')
    # 将dataframe转为numpy，便于处理数据
    dataset_home = frame_home.values
    # 拼接
    draw_home = np.stack((dataset_home[:, 4], dataset_home[:, 6]), axis=1)
    # 切割
    One = draw_home[0:10, :]
    Two = draw_home[10:22, :]
    Three = draw_home[22:32, :]
    Four = draw_home[32:42, :]
    Five = draw_home[42:52, :]
    Six = draw_home[52:62, :]
    Seven = draw_home[62:71, :]
    Eight = draw_home[71:81, :]
    Nine = draw_home[81:90, :]

    np.savetxt(os.path.join(file_path, 'One.csv'), One, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Two.csv'), Two, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Three.csv'), Three, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Four.csv'), Four, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Five.csv'), Five, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Six.csv'), Six, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Seven.csv'), Seven, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Eight.csv'), Eight, fmt="%f", delimiter=",")
    np.savetxt(os.path.join(file_path, 'Nine.csv'), Nine, fmt="%f", delimiter=",")



def drawer(file_name, print_p):
    """
    raw_file_path: 原始数据文件地址
    raw_file_name: 原始数据文件
    file_name: 任务名
    raw_x: 原始数据的x轴，是时间
    raw_y: 原始数据的y轴，是损失值
    """
    raw_file_path = os.path.join(path, 'Data')
    raw_file_name = os.path.join(raw_file_path, '{}.csv'.format(file_name))
    raw_data = pd.read_csv(raw_file_name, header = None)
    # print(f"{file_name}\n-------------------------------")

    raw_data = raw_data.values
    raw_x = raw_data[:, 0]
    raw_y = raw_data[:, 1]
    time = raw_x

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig = plt.figure(figsize=(8, 8.5))
    picture = fig.add_subplot(1, 1, 1)
    picture.plot(time, print_p, color='r', label='Forecast_{}'.format(file_name))
    ticks = picture.set_xticks([0, 1, 3, 6, 12, 18, 24])
    picture.set_title(file_name)
    prediction = fig.add_subplot(1, 1, 1)
    prediction.plot(raw_x, raw_y, color='c', label='Reality_{}'.format(file_name))
    plt.legend(loc='upper left')
    # 保存图片
    picture_path = os.path.join(path, 'Picture')
    plt.savefig(os.path.join(picture_path, 'Forcast_{}.png'.format(file_name)), dpi=400)



# split_nature()
