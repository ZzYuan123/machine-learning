import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)
# 生成0-19的整数 然后转化为(20,1)
# 因为Scikit-learn要求特征数据X必须是二维数组即(样本数, 特征数) 即使只有一个特征也要这么转化
X = np.arange(20).reshape(-1, 1)
y = X.ravel() * 2 + 5

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # 0.2表示只有20%的数据作为测试 剩下80%的数据用来做训练
    test_size=0.2,
    # 固定划分方式 为了让多次运行的结果一致 和seed作用一样
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("测试集：", X_test.ravel())
print("真实值：", y_test)
print("预测值：", model.predict(X_test))
