'''
KMeans = 不断把点分组 + 用均值更新中心，直到稳定

KMeans是一种无监督学习算法 通过不断 分组->计算均值->更新中心->重新分组 最终把相似的数据聚成K个类别

步骤:
    指定K-随机选取K个中心-每个样本找最近中心-每组重新计算均值-重复直到中心值不变

优点:
    简单 速度快 很容易理解
缺点:
    必须自己指定K中心 对初始随即中心敏感 不是个奇怪形状的数据
'''

from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],

    [9, 9],
    [9, 10],
    [10, 9]
])

# 创建KMean模型
model = KMeans(
    # 指定分成几类
    n_clusters=2,
    random_state=42
)
model.fit(X)
# 每个样本属于哪一类
print("标签：")
print(model.labels_)
# 聚类中心
print("聚类中心：")
print(model.cluster_centers_)