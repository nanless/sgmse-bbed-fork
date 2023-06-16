import torch
import numpy as np
import matplotlib.pyplot as plt

sigma_min = 0.0005
sigma_max = 0.5
logsig = np.log(sigma_max / sigma_min)

t = torch.linspace(0, 1, 100) # 产生t=0到1之间的100个值

# 在1.5和100之间均匀分布50个点的theta值
thetas = torch.linspace(1, 10, 10)

plt.figure(figsize=(8,6)) # 设置图表大小

# 制作50个对应theta值下的图表
for idx, theta in enumerate(thetas):
    result = torch.sqrt(
        (
        sigma_min**2
        * torch.exp(-2 * theta * t)
        * (torch.exp(2 * (theta + logsig) * t) - 1)
        * logsig
        )
        /
        (theta + logsig)
    )

    plt.plot(t.numpy(), result.numpy(), linestyle='-', linewidth=2,
             label=r'$\theta$ = {:.2f}'.format(theta)) # 绘制曲线，并设置标签

plt.xlabel('t') # 添加x轴标签
plt.ylabel('std') # 添加y轴标签
plt.title('Comparison of Curves') # 添加图表标题
plt.legend(loc='upper left') # 添加图例，并设置位置为右上角
plt.show()
