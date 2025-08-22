import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl

# 读取Excel文件
data = pd.read_excel('correlation_data.xlsx')

# 将DataFrame转换为字典格式
data_dict = data.to_dict(orient='list')

# 输出转换后的字典
# print(data_dict)

# 将数据转化为DataFrame
df = pd.DataFrame(data)
print(df)

# 计算相关性矩阵
corr_matrix = df.corr(method='pearson')

# 显示相关性矩阵
print(corr_matrix)

# 显示相关性矩阵
# plt.figure(figsize=(15, 15))
# sns.heatmap(corr_matrix, annot=True, cmap='Blues')
# plt.show()

# 将相关性表写入Excel
corr_matrix.to_excel('correlation_matrix.xlsx')