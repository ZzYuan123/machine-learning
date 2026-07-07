'''
XGBoot 一棵一棵树训练 后边的树修正前面的树 最后把每棵树的结果加起来 串行思想 每一次都是针对错误学习
       每一棵树都在学习前一轮的错误，但只走一小步（learning_rate），一步一步逼近真实值。

每一步优化方向来自于损失函数的梯度(导数)

最终预测公式:
    最终预测 = 初始预测 + Σ(learning_rate * 每棵树的预测)
    一开始随便猜一个值 然后不断让新树去"修正错误" 每次只修一点点（learning_rate） 最后所有修正加起来，就是最终答案
'''

from sklearn.datasets import load_diabetes  # 加载糖尿病数据集（回归任务）
from sklearn.metrics import mean_squared_error  # 加载糖尿病数据集（回归任务）
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from xgboost import XGBRegressor  # XGBoost 回归器

# 加载数据
# load_diabetes(return_X_y=True) 返回 (特征矩阵, 目标向量)
# X: 442个样本 × 10个特征（年龄、性别、BMI、血压等）
# y: 442个样本对应的糖尿病病情量化值（连续数值，回归任务）
X, y = load_diabetes(return_X_y=True)
# 划分测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建模型
model = XGBRegressor(
    # 设置树的数量为100
    # 构建100棵决策树 后边每一棵修正前面一棵的错误 串行训练
    n_estimators=100,
    # 设置学习率
    # 每棵树的预测结果乘以0.1再加总，防止单棵树过拟合
    # 相当于：最终预测 = 0.1×(树1 + 树2 + ... + 树100)
    learning_rate=0.1,
    # 每棵树的最大深度
    max_depth=3,
    random_state=42,
)
# 训练
# 这一步内部发生了：
#   for i in range(100):
#       第i棵树拟合前面所有树的残差（负梯度方向）
#       这就是 "后边的树修正前面的树"
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 计算误差
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
