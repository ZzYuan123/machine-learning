'''
五种模型一句话总结
① LinearRegression
利用线性关系进行预测，模型简单，适用于回归问题。

② Ridge
在线性回归基础上加入L2正则化，限制模型参数大小，降低过拟合。

③ DecisionTree
根据特征不断分裂数据形成树结构，可用于分类和回归，但容易过拟合。

④ RandomForest
训练多棵随机决策树，通过投票（分类）或平均（回归）得到最终结果，提高泛化能力。

⑤ XGBoost
采用 Boosting 思想，每棵树学习上一棵树的误差，并结合 learning_rate、gamma、lambda 等机制，实现高精度且不易过拟合的预测。
'''