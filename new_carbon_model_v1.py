import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

## 用了三个哑变量
# STEP1载入数据并添加哑变量
# df 包含了18年的数据，前26个自变量 (X1-X26)，以及后续扩展的自变量(X27-X35)
# 阶段哑变量也在其中：D_stage2, D_stage3
# 设置字体为黑体（SimHei），确保系统中安装了该字体
# 设置中文字体为宋体，英文字体为新罗马
rcParams['font.sans-serif'] = ['SimSun']  # 用于正常显示中文标签
rcParams['axes.unicode_minus'] = False    # 用于正常显示负号


df = pd.read_excel('stage3.xlsx')
# df = df_row.apply(lambda x: np.log1p(x))
# 年份
year = df['year'].values
# 因变量
y = df['y'].values
# 自变量（所有阶段的自变量都在同一个DataFrame中，新增自变量在前期赋值为0）
X = df.iloc[:, 2:]# 全部数据

# 哑变量交互项，手动生成交互项
# 阶段一
df['X1_stage1'] = df['X1'] * df['D_stage1']
df['X2_stage1'] = df['X2'] * df['D_stage1']
df['X3_stage1'] = df['X3'] * df['D_stage1']
df['X4_stage1'] = df['X4'] * df['D_stage1']
#df['X5_stage1'] = df['X5'] * df['D_stage1']
#df['X6_stage1'] = df['X6'] * df['D_stage1']
df['X7_stage1'] = df['X7'] * df['D_stage1']
df['X8_stage1'] = df['X8'] * df['D_stage1']
df['X9_stage1'] = df['X9'] * df['D_stage1']
df['X10_stage1'] = df['X10'] * df['D_stage1']
df['X11_stage1'] = df['X11'] * df['D_stage1']
df['X12_stage1'] = df['X12'] * df['D_stage1']
df['X13_stage1'] = df['X13'] * df['D_stage1']
df['X14_stage1'] = df['X14'] * df['D_stage1']
df['X15_stage1'] = df['X15'] * df['D_stage1']
df['X16_stage1'] = df['X16'] * df['D_stage1']
df['X17_stage1'] = df['X17'] * df['D_stage1']
df['X18_stage1'] = df['X18'] * df['D_stage1']
df['X19_stage1'] = df['X19'] * df['D_stage1']
df['X20_stage1'] = df['X20'] * df['D_stage1']
df['X21_stage1'] = df['X21'] * df['D_stage1']
df['X22_stage1'] = df['X22'] * df['D_stage1']
df['X23_stage1'] = df['X23'] * df['D_stage1']
df['X24_stage1'] = df['X24'] * df['D_stage1']
df['X25_stage1'] = df['X25'] * df['D_stage1']
df['X26_stage1'] = df['X26'] * df['D_stage1']
df['X27_stage1'] = df['X27'] * df['D_stage1']
df['X28_stage1'] = df['X28'] * df['D_stage1']
df['X29_stage1'] = df['X29'] * df['D_stage1']
# 阶段二
df['X1_stage2'] = df['X1'] * df['D_stage2']
df['X2_stage2'] = df['X2'] * df['D_stage2']
df['X3_stage2'] = df['X3'] * df['D_stage2']
df['X4_stage2'] = df['X4'] * df['D_stage2']
#df['X5_stage2'] = df['X5'] * df['D_stage2']
#df['X6_stage2'] = df['X6'] * df['D_stage2']
df['X7_stage2'] = df['X7'] * df['D_stage2']
df['X8_stage2'] = df['X8'] * df['D_stage2']
df['X9_stage2'] = df['X9'] * df['D_stage2']
df['X10_stage2'] = df['X10'] * df['D_stage2']
df['X11_stage2'] = df['X11'] * df['D_stage2']
df['X12_stage2'] = df['X12'] * df['D_stage2']
df['X13_stage2'] = df['X13'] * df['D_stage2']
df['X14_stage2'] = df['X14'] * df['D_stage2']
df['X15_stage2'] = df['X15'] * df['D_stage2']
df['X16_stage2'] = df['X16'] * df['D_stage2']
df['X17_stage2'] = df['X17'] * df['D_stage2']
df['X18_stage2'] = df['X18'] * df['D_stage2']
df['X19_stage2'] = df['X19'] * df['D_stage2']
df['X20_stage2'] = df['X20'] * df['D_stage2']
df['X21_stage2'] = df['X21'] * df['D_stage2']
df['X22_stage2'] = df['X22'] * df['D_stage2']
df['X23_stage2'] = df['X23'] * df['D_stage2']
df['X24_stage2'] = df['X24'] * df['D_stage2']
df['X25_stage2'] = df['X25'] * df['D_stage2']
df['X26_stage2'] = df['X26'] * df['D_stage2']
df['X27_stage2'] = df['X27'] * df['D_stage2']
df['X28_stage2'] = df['X28'] * df['D_stage2']
df['X29_stage2'] = df['X29'] * df['D_stage2']
df['X30_stage2'] = df['X30'] * df['D_stage2']
df['X31_stage2'] = df['X31'] * df['D_stage2']

# 阶段三
df['X1_stage3'] = df['X1'] * df['D_stage3']
df['X2_stage3'] = df['X2'] * df['D_stage3']
df['X3_stage3'] = df['X3'] * df['D_stage3']
df['X4_stage3'] = df['X4'] * df['D_stage3']
#df['X5_stage3'] = df['X5'] * df['D_stage3']
#df['X6_stage3'] = df['X6'] * df['D_stage3']
df['X7_stage3'] = df['X7'] * df['D_stage3']
df['X8_stage3'] = df['X8'] * df['D_stage3']
df['X9_stage3'] = df['X9'] * df['D_stage3']
df['X10_stage3'] = df['X10'] * df['D_stage3']
df['X11_stage3'] = df['X11'] * df['D_stage3']
df['X12_stage3'] = df['X12'] * df['D_stage3']
df['X13_stage3'] = df['X13'] * df['D_stage3']
df['X14_stage3'] = df['X14'] * df['D_stage3']
df['X15_stage3'] = df['X15'] * df['D_stage3']
df['X16_stage3'] = df['X16'] * df['D_stage3']
df['X17_stage3'] = df['X17'] * df['D_stage3']
df['X18_stage3'] = df['X18'] * df['D_stage3']
df['X19_stage3'] = df['X19'] * df['D_stage3']
df['X20_stage3'] = df['X20'] * df['D_stage3']
df['X21_stage3'] = df['X21'] * df['D_stage3']
df['X22_stage3'] = df['X22'] * df['D_stage3']
df['X23_stage3'] = df['X23'] * df['D_stage3']
df['X24_stage3'] = df['X24'] * df['D_stage3']
df['X25_stage3'] = df['X25'] * df['D_stage3']
df['X26_stage3'] = df['X26'] * df['D_stage3']
df['X27_stage3'] = df['X27'] * df['D_stage3']
df['X28_stage3'] = df['X28'] * df['D_stage3']
df['X29_stage3'] = df['X29'] * df['D_stage3']
df['X30_stage3'] = df['X30'] * df['D_stage3']
df['X31_stage3'] = df['X31'] * df['D_stage3']
df['X32_stage3'] = df['X32'] * df['D_stage3']
df['X33_stage3'] = df['X33'] * df['D_stage3']
df['X34_stage3'] = df['X34'] * df['D_stage3']
df['X35_stage3'] = df['X35'] * df['D_stage3']
# 新的自变量矩阵
X = df[['D_stage1','D_stage2','D_stage3','X1_stage1', 'X2_stage1', 'X3_stage1','X4_stage1','X7_stage1','X8_stage1','X9_stage1','X10_stage1','X11_stage1','X12_stage1', 'X13_stage1','X14_stage1', 'X15_stage1','X16_stage1','X17_stage1','X18_stage1','X19_stage1','X20_stage1','X21_stage1','X22_stage1', 'X23_stage1','X24_stage1', 'X25_stage1','X26_stage1', 'X27_stage1', 'X28_stage1','X29_stage1','X1_stage2', 'X2_stage2', 'X3_stage2','X4_stage2','X7_stage2','X8_stage2','X9_stage2','X10_stage2','X11_stage2','X12_stage2', 'X13_stage2','X14_stage2', 'X15_stage2','X16_stage2','X17_stage2','X18_stage2','X19_stage2','X20_stage2','X21_stage2','X22_stage2', 'X23_stage2','X24_stage2', 'X25_stage2','X26_stage2', 'X27_stage2', 'X28_stage2', 'X29_stage2','X30_stage2','X31_stage2','X1_stage3', 'X2_stage3', 'X3_stage3','X4_stage3','X7_stage3','X8_stage3','X9_stage3','X10_stage3','X11_stage3','X12_stage3', 'X13_stage3','X14_stage3', 'X15_stage3','X16_stage3','X17_stage3','X18_stage3','X19_stage3','X20_stage3','X21_stage3','X22_stage3', 'X23_stage3','X24_stage3', 'X25_stage3','X26_stage3', 'X27_stage3', 'X28_stage3', 'X29_stage3','X30_stage3','X31_stage3','X32_stage3','X33_stage3','X34_stage3', 'X35_stage3']].values

# STEP2 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_train = X_scaled[:21,:]
X_scaled_test = X_scaled[21:,:]
y_train = y[:21]
y_test = y[21:]

# STEP3 使用岭回归和Lasso回归
# 使用交叉验证找到最有正则化参数，并分别训练岭回归和Lasso回归模型
# 岭回归
# 设置岭回归的参数范围
ridge_params = {'alpha': np.logspace(-4, 4, 10), 'max_iter': [1000000000]}  # 正则化强度范围

# 使用时间序列分割器，避免未来数据泄漏
tscv = TimeSeriesSplit(n_splits=5)

# 岭回归 + 网格搜索交叉验证
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, ridge_params, cv=tscv, scoring='neg_mean_squared_error')
ridge_cv.fit(X_scaled_train, y_train)

# 打印最优参数和对应的均方误差
print('-------------------------------')
print(f"Best Ridge Alpha: {ridge_cv.best_params_['alpha']}")
print(f"Best Ridge CV MSE: {-ridge_cv.best_score_}")

# Lasso回归
# 设置Lasso的参数范围
lasso_params = {'alpha': np.logspace(-4, 4, 10)}

# Lasso + 网格搜索交叉验证
lasso = Lasso(max_iter=10000000)
lasso_cv = GridSearchCV(lasso, lasso_params, cv=tscv, scoring='neg_mean_squared_error')
lasso_cv.fit(X_scaled_train, y_train)

# 打印最优参数和对应的均方误差
print('-------------------------------')
print(f"Best Lasso Alpha: {lasso_cv.best_params_['alpha']}")
print(f"Best Lasso CV MSE: {-lasso_cv.best_score_}")

# STEP4 模型评估
# 拟合效果预测结果
y_pred_ridge = ridge_cv.predict(X_scaled)
y_pred_lasso = lasso_cv.predict(X_scaled)
y_pred_ridge_train = ridge_cv.predict(X_scaled_train)
y_pred_lasso_train = lasso_cv.predict(X_scaled_train)
# 计算模型的MSE和R²
mse_ridge = mean_squared_error(y_train, y_pred_ridge_train)
r2_ridge = r2_score(y_train, y_pred_ridge_train)
mse_lasso = mean_squared_error(y_train, y_pred_lasso_train)
r2_lasso = r2_score(y_train, y_pred_lasso_train)
print('-------------------------------')
print('拟合效果为：')
print(f"Ridge Regression MSE: {mse_ridge}, R²: {r2_ridge}")
print(f"Lasso Regression MSE: {mse_lasso}, R²: {r2_lasso}")
print('-------------------------------')
# 拟合精度：
y_pred_ridge_test = ridge_cv.predict(X_scaled_test)
y_pred_lasso_test = lasso_cv.predict(X_scaled_test)
# 计算预测误差
errors_ridge = [(y_pred_ridge_test[i] - y_test[i])/y_test[i] for i in range(len(y_test))]
errors_lasso = [(y_pred_lasso_test[i] - y_test[i])/y_test[i] for i in range(len(y_test))]
print('拟合精度为：')
print('岭回归：',errors_ridge)
print('Lasso回归：',errors_lasso)
print('-------------------------------')

# STEP5 鲁棒性检验
# 1.添加噪声
# 在自变量中添加噪声
noise_factor = 0.1  # 噪声强度
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_scaled.shape)

# 使用Ridge模型进行训练
model1 = Ridge(alpha=1.0)
model1.fit(X_noisy, y)

# 预测结果
y1_pred_noisy = model1.predict(X_noisy)

# 计算模型的评估指标
mse_noisy = mean_squared_error(y, y1_pred_noisy)
r2_noisy = r2_score(y, y1_pred_noisy)
print('-------------------------------')
print('噪声测试结果')
print(f'Model performance with noise in Radge: MSE = {mse_noisy}, R² = {r2_noisy}')

# 使用Lasso模型进行训练
model2 = Lasso(alpha=1.0)
model2.fit(X_noisy, y)

# 预测结果
y2_pred_noisy = model2.predict(X_noisy)

# 计算模型的评估指标
mse_noisy = mean_squared_error(y, y2_pred_noisy)
r2_noisy = r2_score(y, y2_pred_noisy)
print('-------------------------------')
print(f'Model performance with noise in Lasso: MSE = {mse_noisy}, R² = {r2_noisy}')

# 2.添加异常值
print('-------------------------------')
print('异常值添加测试：')
# 在原数据中随机添加异常值
num_outliers = 10  # 添加10个异常点
outlier_indices = np.random.choice(np.arange(X_scaled.shape[0]), num_outliers, replace=False)

X_outliers = X_scaled.copy()
X_outliers[outlier_indices] += 5 * np.random.normal(size=X_outliers[outlier_indices].shape)  # 添加异常值

# 训练模型
ridge_cv.fit(X_outliers, y)

# 预测并计算评估指标
y_pred_outliers = ridge_cv.predict(X_outliers)

mse_outliers = mean_squared_error(y, y_pred_outliers)
r2_outliers = r2_score(y, y_pred_outliers)

print(f'Model performance with outliers in Ridge: MSE = {mse_outliers}, R² = {r2_outliers}')
# 训练模型
lasso_cv.fit(X_outliers, y)

# 预测并计算评估指标
y_pred_outliers = lasso_cv.predict(X_outliers)

mse_outliers = mean_squared_error(y, y_pred_outliers)
r2_outliers = r2_score(y, y_pred_outliers)
print('-------------------------------')
print(f'Model performance with outliers in lasso: MSE = {mse_outliers}, R² = {r2_outliers}')

# STEP6 可视化模型预测结果
plt.figure(figsize=(15, 10))
plt.scatter(year, y, color = 'red', label='真实碳排放量', marker='o',s=50)
plt.plot(year, y_pred_ridge,color = 'blue',linewidth = 4, label='岭回归',  linestyle='--')
plt.plot(year, y_pred_lasso,color = 'black',linewidth = 4, label='Lasso回归',  linestyle='-')
plt.xticks(year, rotation=45, fontsize=30)  # 将 x 轴标签设置为所有年份
plt.yticks(fontsize=30)  # 增大 y 轴标签的字体大小
plt.xlabel('年份',fontsize=30,labelpad=20)
plt.ylabel('碳排放量（百万吨）',fontsize=30,labelpad=20)
# plt.title('emission analysis per year',fontsize=20)
plt.legend(fontsize=30)
# plt.grid(True)
plt.show()

# 不同阶段系数解读
# 获取回归系数
ridge_coefficients = ridge_cv.best_estimator_.coef_
lasso_coefficients = lasso_cv.best_estimator_.coef_
# 显示基础系数（第一阶段的系数）
print('-------------------------------')
print("Base coefficients (Stage 1)in ridge:", ridge_coefficients[3:30])
# 哑变量系数
print("Constant coefficient for Stage 1 (β0)in ridge:", ridge_coefficients[0])
print("Dummy variable coefficient for Stage 2 (γ1)in ridge:", ridge_coefficients[1])
print("Dummy variable coefficient for Stage 3 (γ2)in ridge:", ridge_coefficients[2])
# 显示交互项系数（第二、三阶段的系数变化）
print("Interaction coefficients for Stage 2 (α_i)in ridge:", ridge_coefficients[30:30+29])
print("Interaction coefficients for Stage 3 (δ_i)in ridge:", ridge_coefficients[30+29:])
print('-------------------------------')
# 显示基础系数（第一阶段的系数）
print("Base coefficients (Stage 1)in lasso:", lasso_coefficients[3:30])
# 哑变量系数
print("Constant coefficient for Stage 1 (β0)in ridge:", ridge_coefficients[0])
print("Dummy variable coefficient for Stage 2 (γ1)in lasso:", lasso_coefficients[1])
print("Dummy variable coefficient for Stage 3 (γ2)in lasso:", lasso_coefficients[2])
# 显示交互项系数（第二、三阶段的系数变化）
print("Interaction coefficients for Stage 2 (α_i)in lasso:", lasso_coefficients[30:30+29])
print("Interaction coefficients for Stage 3 (δ_i)in lasso:", lasso_coefficients[30+29:])

## STEP2 定义滚动窗口验证参数
window_size = 10  # 窗口大小（年）
step = 1          # 步长（年）
n_years = len(year)
n_validations = n_years - window_size  # 验证次数（2023-2010=13次）

# 初始化存储容器
ridge_rolling_predictions = []
lasso_rolling_predictions = []
rmse_ridge = []
rmse_lasso = []
mae_ridge = []
mae_lasso = []
validation_years = year[window_size:]  # 验证年份：2011-2023年

## STEP3 滚动窗口验证循环
for i in range(window_size, n_years):
    # 定义当前窗口的索引
    train_idx = range(i - window_size, i)
    val_idx = [i]
    
    # 提取数据并标准化（每个窗口独立标准化，避免数据泄漏）
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 使用之前网格搜索得到的最佳参数（注意max_iter需为整数）
    best_ridge_alpha = ridge_cv.best_params_['alpha']
    best_lasso_alpha = lasso_cv.best_params_['alpha']
    
    # 训练Ridge模型并预测（将max_iter改为整数，原1e9改为1000000000）
    ridge_model = Ridge(alpha=best_ridge_alpha, max_iter=1000000000)  # 修正此处，使用整数
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_val_scaled)
    
    # 训练Lasso模型并预测（Lasso的max_iter也需检查，确保为整数，原1e7改为10000000）
    lasso_model = Lasso(alpha=best_lasso_alpha, max_iter=10000000)  # 修正此处，使用整数
    lasso_model.fit(X_train_scaled, y_train)
    lasso_pred = lasso_model.predict(X_val_scaled)
    
    # 记录预测结果和误差
    ridge_rolling_predictions.append(ridge_pred[0])
    lasso_rolling_predictions.append(lasso_pred[0])
    
    rmse_ridge.append(np.sqrt(mean_squared_error(y_val, ridge_pred)))
    rmse_lasso.append(np.sqrt(mean_squared_error(y_val, lasso_pred)))
    # 手动计算MAE
    mae_ridge.append(np.mean(np.abs(y_val - ridge_pred)))
    mae_lasso.append(np.mean(np.abs(y_val - lasso_pred)))
    
## STEP4 可视化滚动窗口验证结果
### 子图1：预测值与实际值对比（滚动窗口部分）
plt.figure(figsize=(18, 12))

# 绘制全时段真实值
plt.plot(year, y, 'o-', color='red', label='真实碳排放量', markersize=15, linewidth=4)

# 绘制滚动窗口预测值（Ridge和Lasso）
plt.plot(validation_years, ridge_rolling_predictions, 'd-', color='blue', label='岭回归滚动预测', markersize=15, linewidth=4)
plt.plot(validation_years, lasso_rolling_predictions, '^-', color='green', label='Lasso回归滚动预测', markersize=15, linewidth=4)

# 标注窗口滑动逻辑（示例：第一个窗口2001-2010年训练，2011年验证）
plt.annotate("训练窗口：2001-2010", xy=(2010.5, y[10]+0.5), xytext=(2012, y[10]+2),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=30)
plt.annotate("验证点：2011", xy=(2011, ridge_rolling_predictions[0]), xytext=(2013, ridge_rolling_predictions[0]-1.2),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=30)

plt.xticks(year, rotation=45, fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('年份', fontsize=30)
plt.ylabel('碳排放量（百万吨）', fontsize=30)
#plt.title('滚动窗口验证预测结果对比（窗口=10年，步长=1年）', fontsize=40)
plt.legend(fontsize=25)
#plt.grid(alpha=0.3)

### 子图2：RMSE误差趋势
plt.figure(figsize=(16, 6))
plt.plot(validation_years, rmse_ridge, 'o--', color='blue', label='Ridge RMSE')
plt.plot(validation_years, rmse_lasso, 'd-', color='green', label='Lasso RMSE')
plt.xticks(validation_years, rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('验证年份', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('滚动窗口验证RMSE趋势', fontsize=20)
plt.legend(fontsize=20)
#plt.grid(alpha=0.3)

### 子图3：MAE误差趋势（可选）
plt.figure(figsize=(16, 6))
plt.plot(validation_years, mae_ridge, 'o--', color='blue', label='Ridge MAE')
plt.plot(validation_years, mae_lasso, 'd-', color='green', label='Lasso MAE')
plt.xticks(validation_years, rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('验证年份', fontsize=20)
plt.ylabel('MAE', fontsize=20)
plt.title('滚动窗口验证MAE趋势', fontsize=20)
plt.legend(fontsize=20)
#plt.grid(alpha=0.3)

plt.show()

## STEP5 输出滚动窗口验证结果（可选）
print("-------------------------------")
print("滚动窗口验证结果（2011-2023年）：")
print("Ridge模型各年RMSE：", np.round(rmse_ridge, 3))
print("Lasso模型各年RMSE：", np.round(rmse_lasso, 3))
print("-------------------------------")
