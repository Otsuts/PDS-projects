import matplotlib.pyplot as plt

# 四组数据
# data1 = [0.792, 0.858, 0.883, 0.900]
# data2 = [0.816, 0.873, 0.887, 0.899]
# data3 = [0.888, 0.903, 0.906, 0.913]
# data4 = [0.600, 0.710, 0.795, 0.833]
# data5 = [0.927, 0.927, 0.927, 0.927]

data1 = [5.432,4.890,5.393,4.932]
data2 = [4.346,3.569,4.768,5.726]
data3 = [10.773,7.613,7.352,5.563]
data4 = [25.794,29.866,38.913,54.099]
data5 = [51.485,51.485,51.485,51.485]
# 横坐标
x = [32, 64, 128, 256]

# 绘图
plt.plot(x, data1, 'o-', label='SFG')
plt.plot(x, data2, 'o-', label='AutoEncoder')
plt.plot(x, data3, 'o-', label='PCA')
plt.plot(x, data4, 'o-', label='LLE')
plt.plot(x, data5, label='Plain SVM')

# 设置标题和坐标轴标签
plt.title('Precision of dimension reduction methods')
plt.xlabel('Reduced dim')
plt.ylabel('Precision')

# 添加图例
plt.legend()

# 显示图形
plt.show()
