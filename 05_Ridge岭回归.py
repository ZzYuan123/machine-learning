import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 生成数据
X = np.linspace(0, 10, 8).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.5, 8)

# 不同正则化强度
# alpha = 0 表示无正则化（普通线性回归）
# alpha 越大，正则化惩罚越强，模型越平滑
alphas = [0, 0.01, 0.1, 1.0, 10.0]
colors = ['#FF0000', '#FF8C00', '#32CD32', '#1E90FF', '#8B008B']
linestyles = ['-', '-', '-', '-', '-']

# ax1: 左图 - 显示拟合曲线对比
# ax2: 右图 - 显示误差分析
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ========== 左图：拟合曲线对比 ==========
X_plot = np.linspace(-0.5, 10.5, 500).reshape(-1, 1)

ax1.scatter(X, y, s=100, color='black', label='数据点 (8个)', zorder=5)

for alpha, color in zip(alphas, colors):
    # ----- 构建机器学习管道 -----
    # Pipeline按顺序执行：
    # 步骤1: StandardScaler() - 标准化特征（均值0，标准差1）
    #        为什么需要？防止X^5等数值爆炸（10^5=100000）
    # 步骤2: PolynomialFeatures(5) - 生成5次多项式特征
    #        将 [x] 变为 [1, x, x², x³, x⁴, x⁵]
    # 步骤3: Ridge(alpha=alpha) - 带L2正则化的线性回归
    #        L2正则化 = 惩罚系数的平方和，防止过拟合
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(5),
        Ridge(alpha=alpha)
    )
    # 训练模型
    model.fit(X, y)
    y_pred = model.predict(X_plot)

    label = f'无正则化' if alpha == 0 else f'α = {alpha}'
    ax1.plot(X_plot, y_pred, color=color, linewidth=2, label=label)

ax1.plot(X_plot, np.sin(X_plot), color='green', linestyle='--',
         linewidth=2.5, label='真实正弦曲线', alpha=0.8)

ax1.set_xlabel('X', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
ax1.set_title('不同正则化强度的拟合曲线对比', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-2.5, 2.5)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)

# ========== 右图：误差分析 ==========
train_errors = []
test_errors = []
coef_norms = []

# 循环每个alpha值，训练模型并计算指标
for alpha in alphas:
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(5),
        Ridge(alpha=alpha)
    )
    model.fit(X, y)

    # 训练误差
    y_pred = model.predict(X)
    train_errors.append(mean_squared_error(y, y_pred))

    # 系数L2范数
    coef = model.named_steps['ridge'].coef_
    coef_norms.append(np.linalg.norm(coef))

# 绘制误差对比
ax2.bar(range(len(alphas)), train_errors, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(alphas)))
ax2.set_xticklabels(['无正则化' if a == 0 else f'α={a}' for a in alphas], rotation=0)
ax2.set_xlabel('正则化强度', fontsize=13, fontweight='bold')
ax2.set_ylabel('均方误差 (MSE)', fontsize=13, fontweight='bold')
ax2.set_title('训练集误差分析', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值
for i, err in enumerate(train_errors):
    ax2.text(i, err + 0.01, f'{err:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ========== 打印详细分析 ==========
print("=" * 60)
print("正则化强度对模型的影响分析")
print("=" * 60)
print(f"{'α值':<10} {'训练MSE':<12} {'系数L2范数':<12} {'过拟合程度'}")
print("-" * 60)

for alpha, err, norm in zip(alphas, train_errors, coef_norms):
    alpha_str = '无正则化' if alpha == 0 else f'{alpha}'
    overfit = '严重' if alpha == 0 else ('轻微' if alpha < 0.1 else '良好')
    print(f"{alpha_str:<10} {err:<12.4f} {norm:<12.2f} {overfit}")

print("=" * 60)