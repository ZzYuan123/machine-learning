from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error


# 数据
X, y = load_diabetes(return_X_y=True)
# 切分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 模型（核心参数都在这里）
model = XGBRegressor(
    n_estimators=100,     # 100棵树（一步一步修正）
    learning_rate=0.1,    # 每次只修一点点
    max_depth=3,          # 每棵树最多3层
    subsample=0.8,        # 每棵树看80%数据（防过拟合）
    colsample_bytree=0.8, # 每棵树看80%特征
    random_state=42
)
# 训练
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)

# 误差
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# 打印每一棵树在干嘛
# 打印结果是一个棵二叉树
print("\n" + "="*60)
print("第1棵树的结构（文本表示）")
print("="*60)
booster = model.get_booster()
print(booster.get_dump()[0])

