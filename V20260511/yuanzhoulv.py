import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 计算π近似值
n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
pi_approx = n * np.sin(np.pi / n)
error = np.abs(np.pi - pi_approx)
h = 1 / n

# 绘图
plt.figure(figsize=(8, 6))
plt.loglog(h, error, 'o-', linewidth=2, markersize=8, label='数值解误差')

# 画两条参考线
h_ref = np.array([1e-3, 1])
plt.loglog(h_ref, 10*h_ref**2, 'k--', label='斜率=2')
plt.loglog(h_ref, 100*h_ref**4, 'k--', label='斜率=4')

# 标注数值
labels = [
    (h[1], error[1], '1.46'), (h[2], error[2], '1.87'),
    (h[3], error[3], '1.97'), (h[4], error[4], '1.99'),
    (h[5], error[5], '2.00'), (h[6], error[6], '2.00'),
    (h[7], error[7], '2.00'), (h[8], error[8], '2.00'),
    (h[3], error[3]/100, '5.31'), (h[4], error[4]/100, '7.50'),
    (h[5], error[5]/100, '9.76')
]
for x, y, text in labels:
    plt.annotate(text, (x, y), xytext=(5,5), textcoords='offset points')

# 图表设置
plt.xlabel('单元尺寸 h=1/n')
plt.ylabel('误差 eₙ=|π-πₙ|')
plt.title('有限元收敛性分析')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(1e-3, 1)
plt.ylim(1e-10, 1e3)

# 输出数值表格
print("n\tπ近似值\t\t\t误差")
print("-"*50)
for i in range(len(n)):
    print(f"{n[i]}\t{pi_approx[i]:.15f}\t{error[i]:.15f}")

plt.tight_layout()
plt.show()