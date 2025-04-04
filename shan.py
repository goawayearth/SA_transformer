import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ================== 数据输入 ==================
# 示例数据集：假设我们有五个样本，三个自变量和五个指标
np.random.seed(42)
X = pd.DataFrame({
    '互联网普及率': np.random.uniform(0.3, 0.8, 5),
    '数字农业技术投入': np.random.uniform(10, 50, 5),
    '电商交易额': np.random.uniform(100, 500, 5)
})

Y_indicators = pd.DataFrame({
    '农业总产值': np.random.uniform(1000, 5000, 5),
    '单位面积粮食产量': np.random.uniform(300, 600, 5),
    '农业化肥使用强度': np.random.uniform(100, 200, 5),  # 反向指标
    '农业灌溉水利用效率': np.random.uniform(0.4, 0.9, 5),
    '农民人均纯收入': np.random.uniform(5000, 20000, 5)
})

# ================== 数据标准化 ==================
scaler = MinMaxScaler()

# 正向指标
positive_indicators = ['农业总产值', '单位面积粮食产量', '农业灌溉水利用效率', '农民人均纯收入']
Y_indicators[positive_indicators] = scaler.fit_transform(Y_indicators[positive_indicators])

# 反向指标
Y_indicators['农业化肥使用强度'] = 1 - scaler.fit_transform(Y_indicators[['农业化肥使用强度']])

# ================== 熵值法计算 ==================
P = Y_indicators.div(Y_indicators.sum(axis=0), axis=1)
E = -np.nansum(P * np.log(P + 1e-9), axis=0) / np.log(len(Y_indicators))  # 熵值
W = (1 - E) / np.sum(1 - E)  # 权重

# ================== 综合得分计算 ==================
Y_score = np.dot(P, W)

# ================== 回归分析 ==================
X_with_const = sm.add_constant(X)  # 添加常数项
model = sm.OLS(Y_score, X_with_const).fit()

# ================== 输出结果 ==================
print('==== 权重 (W) ====', W)
print('\n==== 综合得分 (Y) ====', Y_score)
print('\n==== 回归结果 ====')
print(model.summary())
