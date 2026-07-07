# 欠拟合
# PolynomialFeatures(1-2) 模型太简单 无法捕捉数据太多 属于欠拟合
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X=np.linspace(0,10,50).reshape(-1,1)
y=np.sin(X).ravel()+np.random.normal(0,0.2,50)

model=make_pipeline(
    # 升维 将单个X特征转化为[1, x, x², x³, x⁴, x⁵] 6个特征
    PolynomialFeatures(5),
    LinearRegression()
)

model.fit(X,y)

y_pred=model.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred,color="red")
plt.show()

# 正常
# PolynomialFeatures(3-5) 模型复杂程度适中 有拟合趋势但是不会捕捉噪声 正常情况
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X=np.linspace(0,10,50).reshape(-1,1)
y=np.sin(X).ravel()+np.random.normal(0,0.2,50)

model=make_pipeline(
    # 升维 将单个X特征转化为[1, x, x², x³, x⁴, x⁵] 6个特征
    PolynomialFeatures(2),
    LinearRegression()
)

model.fit(X,y)

y_pred=model.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred,color="red")
plt.show()

# 过拟合
# PolynomialFeatures(10-15) 模型太灵活 什么都会学 包括噪声 过拟合
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X=np.linspace(0,10,50).reshape(-1,1)
y=np.sin(X).ravel()+np.random.normal(0,0.2,50)

model=make_pipeline(
    # 升维 将单个X特征转化为[1, x, x², x³, x⁴, x⁵] 6个特征
    PolynomialFeatures(20),
    LinearRegression()
)

model.fit(X,y)

y_pred=model.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred,color="red")
plt.show()
