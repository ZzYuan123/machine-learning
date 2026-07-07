from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

plt.rcParams["font.family"]="Microsoft YaHei"
plt.rcParams["axes.unicode_minus"]=False

X=[[i] for i in range(20)]
y=[0]*10+[1]*10

model=DecisionTreeClassifier(
    # 预剪枝 在训练过程中限制树的生长
    # 控制树的层数 最多3层           限制树的深度
    max_depth=3,
    # 一个节点至少要3个数据才能继续分
    min_samples_split=3,
    # 每个叶子至少要2个数据
    min_samples_leaf=2,

    # 后剪枝 先长满再删掉无用的分支
    # ccp_alpha 剪枝强度控制器
    # 0      不剪枝 疯狂长
    # 小     轻微剪枝
    # 大     强剪枝 树很简单
    ccp_alpha=0
)

model.fit(X,y)

plt.figure(figsize=(12,6))

tree.plot_tree(
    model,
    filled=True
)

plt.show()