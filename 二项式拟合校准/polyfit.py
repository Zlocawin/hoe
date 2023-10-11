import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['axes.unicode_minus'] = False

# 示例数据
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0.5, 2.5, 2.0, 4.0, 3.5, 3])

# 执行二项式拟合
coefficients = np.polyfit(x, y, 2)
poly = np.poly1d(coefficients)

# 生成拟合曲线的x值
x_fit = np.linspace(min(x), max(x), 100)
y_fit = poly(x_fit)

# 绘制原始数据点和拟合曲线
plt.scatter(x, y, label='原始数据')
plt.plot(x_fit, y_fit, label='二项式拟合曲线', color='red')

# 添加图例和标签
plt.legend()
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示图形
plt.show()

# 打印二项式拟合的系数
print("二项式拟合系数：", coefficients)
