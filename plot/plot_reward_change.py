# encoding=gbk


import numpy as np
import matplotlib.pyplot as plt



dqn_reward = np.loadtxt('./change_reward/dqn_reward_chan1.txt')

rand_reward = np.loadtxt('./change_reward/random_reward4.txt')

fedcs_reward = np.loadtxt('./change_reward/fedcs_reward4.txt')

pred_reward = np.loadtxt('./change_reward/pred_reward4.txt')

opt_reward = np.ones(2000)
opt_reward[:1000] = opt_reward[:1000] * 2015
opt_reward[1000:] = opt_reward[1000:] * 1937

# pred_reward -= 100
# dqn_reward += 100
# fedcs_reward += 50

interval = 2000

x = np.linspace(0, 20000, num=interval)

fig, ax = plt.subplots()

plt.xticks([0, 5000, 10000, 15000, 20000])
plt.yticks([0, 500, 1000, 1500, 2000])

opt_line = ax.plot(x, opt_reward[:interval], '-', linewidth=2, color='#d62728', label="Offline")
# pred_line = ax.plot(x, pred_reward[:interval], '-.', linewidth=2, color='#d62728', label="Proactive FedCS")
ddqn_line = ax.plot(x, dqn_reward[:interval], '-', linewidth=2, color='#1f77b4', label="DDQN-based (Proposed)")
fedcs_line = ax.plot(x, fedcs_reward[:interval], '--', linewidth=2, color='#ff7f0e', label="FedCS [Nishio, 2019]")
rand_line = ax.plot(x, rand_reward[:interval], ':', linewidth=2, color='#2ca02c', label="FedAvg [Google Team]")

plt.ylabel('Numerical Reward', fontdict={'size': 23})
plt.xlabel('Episodes', fontdict={'size': 23})
plt.tick_params(labelsize=18)
leg = ax.legend(fontsize=17)# , frameon=False)

# first_legend = plt.legend(handles=[opt_line, pred_line, ddqn_line], fontsize=14)
# # ax = plt.gca().add_artist(first_legend)
# first_legend.set_draggable(True)
#
# second_legend = plt.legend(handles=[fedcs_line, rand_line], fontsize=14)
# second_legend.set_draggable(True)


leg.set_draggable(True)

ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.yaxis.get_major_formatter().set_powerlimits((1, 2))
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

# avgDataVal_JKQ = rand_reward[0]
# avgDataVal_JSQ = opt_reward[0]
# ax.annotate(r'', (3, avgDataVal_JKQ),
#             (3, avgDataVal_JSQ),
#             ha="right", va="center",
#             size=12,
#             arrowprops=dict(arrowstyle='-',
#                             fc="b", ec="dodgerblue",
#                             connectionstyle="arc3,rad=-0.1",
#                             ),
#             )
#
# distance = format((avgDataVal_JSQ-avgDataVal_JKQ), '.1e')
# yaxis_strTextPre = avgDataVal_JKQ + 0.6 * (avgDataVal_JSQ-avgDataVal_JKQ)
# yaxis_strDist = avgDataVal_JKQ + 0.45 * (avgDataVal_JSQ-avgDataVal_JKQ)
# strText = r"Distance is:"
# strDistance = str(distance)
# plt.text(3.1, yaxis_strTextPre, strText, size=15, color="dodgerblue")
# plt.text(3.1, yaxis_strDist, strDistance, size=15, color="dodgerblue")

plt.annotate(color='r', s="Uncertainties\noccurred here",
             xy=(10000, 1150), xycoords='data',
             xytext=(12000, 1075), textcoords='data',
             size=18,
             arrowprops=dict(arrowstyle="fancy", color='r', connectionstyle="arc3"))
plt.axvline(x=10000, ls="-", c="r", lw=1)  # 添加垂直直线


plt.tight_layout()
plt.show()
plt.close()



