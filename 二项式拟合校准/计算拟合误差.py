import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import csv

matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['axes.unicode_minus'] = False

# 5个点拟合曲线
# x：起始gain值
# polygt:真实二项式函数
def polyfit_5points(x, polygt):
    x_list = [0.4 * x, 0.7 * x, x, 1.3 * x, 1.6 * x]

    y_list = polygt(x_list)
    y_list_noise = []   # 上下随机加噪的y值(模糊像素)

    for y in y_list:
        # 生成随机噪声（-4 到 4 的整数）
        noise = random.randint(-4, 4)
        noise_y = y + noise
        y_list_noise.append(noise_y)

    coefficients_5points = np.polyfit(x_list, y_list_noise, 2)   # 利用加噪值进行二项式拟合
    poly_5points = np.poly1d(coefficients_5points)

    return poly_5points


# 7个点拟合曲线
# x：起始gain值
# polygt:真实二项式函数
def polyfit_7points(x, polygt):
    x_list = [0.4 * x, 0.6 * x, 0.8 * x, x, 1.2 * x, 1.4 * x, 1.6 * x]

    y_list = polygt(x_list)
    y_list_noise = []   # 上下随机加噪的y值(模糊像素)

    for y in y_list:
        # 生成随机噪声（-4 到 4 的整数）
        # noise = random.randint(-4, 4)
        # 生成随机噪声（-3 到 3 的整数）
        noise = random.randint(-3, 3)
        noise_y = y + noise
        y_list_noise.append(noise_y)

    coefficients_7points = np.polyfit(x_list, y_list_noise, 2)   # 利用加噪值进行二项式拟合
    poly_5points = np.poly1d(coefficients_7points)

    return poly_7points


# 给定范围和取样点数，自动取点，取样点数必须是奇数
# 比如，sample_min = 0.4, sample_max=1.6, num_points = 5，即为5个点采样
# 由于给y加了随机噪声，因此每次输入一样时，返回值每次都不一样
def polyfit_points(x, sample_min, sample_max, num_points, polygt):
    x_list = np.linspace(sample_min, sample_max, num_points)
    x_list = x * x_list

    y_list = polygt(x_list)
    y_list_noise = []   # 上下随机加噪的y值(模糊像素)

    for y in y_list:
        # 生成随机噪声（-4 到 4 的整数）
        # noise = random.randint(-4, 4)
        # 生成随机噪声（-3 到 3 的整数）
        noise = random.randint(-3, 3)
        noise_y = y + noise
        y_list_noise.append(noise_y)

    coefficients_points = np.polyfit(x_list, y_list_noise, 2)   # 利用加噪值进行二项式拟合
    poly_points = np.poly1d(coefficients_points)

    return poly_points


# 给定拟合曲线及x轴范围计算拟合误差MSE
def MSE_MAE_polyfit(poly1, poly2, x_min, x_max):
    step = 0.1  # 步长，用于生成 x 值
    # 初始化均方误差的累积值
    mse_sum = 0
    mae_sum = 0

    # 生成 x 值的范围
    x_range = np.arange(x_min, x_max + step, step)

    # 遍历 x 范围内的每个 x 值
    for x in x_range:
        # 计算两个曲线在 x 处的 y 值
        y1 = poly1(x)
        y2 = poly2(x)

        # 计算平方差并累积
        squared_error = (y1 - y2) ** 2
        mae_error = abs(y1 - y2)
        mse_sum += squared_error
        mae_sum += mae_error

    # 计算均方误差
    mse = mse_sum / len(x_range)
    
    # 计算平均绝对误差
    mae = mae_sum / len(x_range)

    return mse, mae


x = [11748, 12248, 12748, 13248, 13748, 14248, 14748, 15248, 15748, 16248, 16748, 17248, 17748, 18248, 18748, 19248,
     19748, 20248, 20748, 21248, 21748]     # x轴数据
y = [76, 67, 58, 51, 45, 40, 36, 32, 30, 29, 28, 29, 30, 32, 36, 40, 45, 51, 58, 67, 76]    # 真实y轴数据

# 执行二项式拟合，得到真实二项式系数
coefficients_gt = np.polyfit(x, y, 2)
poly_gt = np.poly1d(coefficients_gt)

num_random = 10     # 每个初始值取样点进行上下随机加噪的次数
MSE_3points = []
MSE_5points = []
MSE_7points = []
MSE_9points = []
MAE_3points = []
MAE_5points = []
MAE_7points = []
MAE_9points = []

# 初始化
for i in range(num_random):
    MSE_3points.append(0)
    MSE_5points.append(0)
    MSE_7points.append(0)
    MSE_9points.append(0)
    MAE_3points.append(0)
    MAE_5points.append(0)
    MAE_7points.append(0)
    MAE_9points.append(0)

header = ['初始x', 'MSE_3', '5', '7', '9', 'MAE_3', '5', '7', '9']
with open('E:\\HUAWEI\\二项式拟合校准\\MSE_MAE_0.4_1.6_加噪上下三像素.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    for num_x in range(10):
        print(num_x, '\tbegin')
        x = random.randint(16748 - 2000, 16748 + 2000)  # 随机产生初始k值

        # 计算num_random次模糊像素随机加噪的拟合误差
        for i in range(num_random):
            poly_3points = polyfit_points(x, 0.4, 1.6, 3, poly_gt)
            MSE_3points[i], MAE_3points[i] = MSE_MAE_polyfit(poly_gt, poly_3points, 11748, 21748)
            poly_5points = polyfit_points(x, 0.4, 1.6, 5, poly_gt)
            MSE_5points[i], MAE_5points[i] = MSE_MAE_polyfit(poly_gt, poly_5points, 11748, 21748)
            poly_7points = polyfit_points(x, 0.4, 1.6, 7, poly_gt)
            MSE_7points[i], MAE_7points[i] = MSE_MAE_polyfit(poly_gt, poly_7points, 11748, 21748)
            poly_9points = polyfit_points(x, 0.4, 1.6, 9, poly_gt)
            MSE_9points[i], MAE_9points[i] = MSE_MAE_polyfit(poly_gt, poly_9points, 11748, 21748)

        # 用num_random次MSE误差均值作为该初始x值处的拟合误差
        data = [x, np.mean(MSE_3points), np.mean(MSE_5points), np.mean(MSE_7points), np.mean(MSE_9points),
                np.mean(MAE_3points), np.mean(MAE_5points), np.mean(MAE_7points), np.mean(MAE_9points)]
        writer.writerow(data)
        print(data)
        print(num_x, '\tend')




