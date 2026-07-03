import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#  在 [0, 10] 的区间内生等距的50个数
X=np.linspace(0,10,50)
# np.sin(X)计算每一个X的正弦值 目的是为了防止线性
# np.random.normal 生成50个正态分布的随机数
# scale 是噪声数
# scale 越小 噪声水平越低 越接近曲线
# scale 越大 噪声水平越高 几乎看不到曲线 随机点
y=np.sin(X)+np.random.normal(0,0.3,50)

plt.scatter(X,y)
plt.show()