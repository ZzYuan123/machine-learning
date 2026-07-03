'''
Root 根节点 决定了第一刀怎么切数据
Node 节点 表示树上的每一个问题 机器在这里决定下一步往哪走
Branch 分支 回答问题后的路线 是/否
Leaf 叶子节点 表示最后的结果 到这里不再继续问问题了

gini 基尼指数 表示当前节点有多混乱
     gini = 0 表示一点都不混乱
     gini = 0.5 表示十分混乱   最小是0 最大是0.5
     越小 说明数据越纯 越大就越不纯
samples 表示当前节点有多少个训练样本 当前节点的数据数量
value 表示当前节点每个类别中的数据分别有多少个
class 表示如果现在停止 机器的最终预测结果
      class = 正常    就表示当前机器预测的结果是正常
Feature 特征值 每一次机器分类 都是按照最容易区分的类别的特征去分裂的
Threshold 阈值 不是我们定义的 是机器按照规律自动分的 是机器自动找出来的
         X<=2.5 这里2.5就是阈值 就是机器自己找的分类标准
'''
# 决策树
# 决策树每一步都会选择一个问题，让数据变得更纯。
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

X = [
    [0],
    [1],
    [2],
    [3],
    [5],
    [8]
]

y = [
    "正常",
    "正常",
    "正常",
    "垃圾",
    "垃圾",
    "垃圾"
]

model = DecisionTreeClassifier()

model.fit(X, y)

print(model.predict([[4]]))

print(model.classes_)

plt.figure(figsize=(10, 6))

tree.plot_tree(
    model,
    filled=True,
    feature_names=["X"],
    class_names=model.classes_
)

plt.show()
