# encoding=gbk
import numpy as np
import matplotlib.pyplot as plt



# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.rcParams['figure.figsize'] = (5.6, 3.5)

dqn_reward = np.loadtxt('./TrainingReward/ddqn_reward.txt')
rand_reward = np.loadtxt('./TrainingReward/random_reward.txt')
fedcs_reward = np.loadtxt('./TrainingReward/fedcs_reward.txt')
pred_reward = np.loadtxt('./TrainingReward/pred_reward.txt')

interval = 2000

opt_reward = np.ones(interval) * 2015

# opt_reward -= 100
# pred_reward -= 120
# dqn_reward += 100
# rand_reward += 200
# fedcs_reward += 50


# x = np.arange(interval)
x = np.linspace(0, 20000, num=2000)

fig, ax = plt.subplots()

# plt.xticks([0, 5, 10, 15, 20])
# plt.rcParams['font.family'] = ['Times New Roman']
ax.plot(x, opt_reward[:interval], '-', linewidth=2, color='#d62728', label="Offline")
# ax.plot(x, pred_reward[:interval], '-.', linewidth=2, color='#d62728', label="Proactive FedCS")
ax.plot(x, dqn_reward[:interval], '-', linewidth=2, color='#1f77b4', label="DDQN-based (Proposed)")
ax.plot(x, fedcs_reward[:interval], '--', linewidth=2, color='#ff7f0e', label="FedCS [Nishio, 2019]")
ax.plot(x, rand_reward[:interval], ':', linewidth=2, color='#2ca02c', label="FedAvg [Google Team]")

# plt.grid()

ax = plt.gca()  # 获取当前图像的坐标轴信息


ax.yaxis.get_major_formatter().set_powerlimits((1, 2))
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))


plt.xticks([0, 5000, 10000, 15000, 20000])

plt.ylabel('Numerical Reward', fontdict={'size': 23})
plt.xlabel('Episodes', fontdict={'size': 23})
plt.tick_params(labelsize=18)
leg = ax.legend(fontsize=17)# , frameon=False)
leg.set_draggable(True)

plt.tight_layout()
# plt.savefig('./imgs/Offline Training Reward.pdf', bbox_inches='tight')
plt.show()
plt.close()




