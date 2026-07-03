import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 构造数据
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([8000, 10000, 12000, 14000, 16000])

# 创建模型
model = LinearRegression()

# 训练
model.fit(X, y)

print("权重:", model.coef_)
print("偏置:", model.intercept_)

# 预测
print(model.predict([[6]]))

# 可视化
plt.scatter(X, y)
plt.plot(X, model.predict(X), color="red")
plt.show()