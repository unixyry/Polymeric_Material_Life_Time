# Polymeric_Material_Life_Time
# 使用MAML策略
先运行main.py进行训练和预测，在运行Prediction.py画图

# 2022.3.22
1.文件地址拼接形式不统一，Linux和(有效的)Windos形式混合了，但是能用  
2.效果和屎一样

# 2022.3.23
1.调了调参数，麻了，结果比较屎  
2.依然比较屎，也许要把所有任务打乱，每次重新采样任务  
3.现在尝试在预测时直接跳过初始样本，保证初始的损失值一定正确  
4.数据需要重新处理，已重新修改间隔和前向  
5.重新处理数据集后，预测结果依然不理想，可能由室内数据过多导致，考虑步骤2  
6.或许不同的任务在微调时使用不同的finetune_step和finetune_rate(还未实现)  
7.绘制meta的学习曲线，绘制finetune的学习曲线

# 也许自动减小学习率有用哦
# 设置一个loss的阈值提前停止可能有效

# meta的学习曲线需要设置x轴的刻度