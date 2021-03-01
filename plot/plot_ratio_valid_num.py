import matplotlib.pyplot as plt
import numpy as np

label_list = ['50', '100', '150', '200']  # 横坐标刻度显示值

# # niid
# fedavg_list = np.array([117, 267, 387, 521])  # 纵坐标值1
# fedcs_list = np.array([219, 432, 646, 847])  # 纵坐标值2
# mab_list = np.array([203, 443, 688, 1020])  # 纵坐标值3
# linucb_list = np.array([388, 823, 1273, 1732])  # 纵坐标值4
# optimal_list = np.array([500, 1000, 1500, 2000])  # 纵坐标值4

# iid
fedavg_list = np.array([117, 267, 400, 520])  # 纵坐标值1
fedcs_list = np.array([219, 431, 656, 845])  # 纵坐标值2
mab_list = np.array([216, 505, 907, 1298])  # 纵坐标值3
linucb_list = np.array([425, 890, 1360, 1826])  # 纵坐标值4
optimal_list = np.array([500, 1000, 1500, 2000])  # 纵坐标值4

fedavg_choose_list = np.array([500, 1000, 1500, 2000])
fedcs_choose_list = np.array([500, 1000, 1500, 2000])
mab_choose_list = np.array([500, 1000, 1500, 2000])
linucb_choose_list = np.array([500, 1000, 1500, 2000])
optimal_choose_list = np.array([500, 1000, 1500, 2000])


x = range(len(label_list))

fig, ax = plt.subplots()

rects5 = plt.bar([i + 0.6 for i in x], height=optimal_list / optimal_choose_list, color='#9467bd', width=0.15, label="Optimal", hatch='\\')
rects4 = plt.bar([i + 0.45 for i in x], height=linucb_list / linucb_choose_list, color='#d62728', width=0.15, label="cMAB-based (Proposed)", hatch='x')
rects3 = plt.bar([i + 0.3 for i in x], height=mab_list / mab_choose_list, color='#2ca02c', width=0.15, label="MAB-based [Yoshida, 2020]", hatch='')
rects2 = plt.bar([i + 0.15 for i in x], height=fedcs_list/fedcs_choose_list, color='#ff7f0e', width=0.15, label="FedCS [Nishio, 2019]", hatch='/')
rects1 = plt.bar(x, height=fedavg_list/fedavg_choose_list, width=0.15, color='#1f77b4', label="FedAvg [McMahan, 2016]", hatch='.')

# rects1 = plt.bar(x, height=fedavg_list/fedavg_choose_list, width=0.15, color='#1f77b4', label="FedAvg [Google Team]")
# rects2 = plt.bar([i + 0.15 for i in x], height=fedcs_list/fedcs_choose_list, color='#ff7f0e', width=0.15, label="FedCS [Nishio, 2019]")
# rects3 = plt.bar([i + 0.3 for i in x], height=mab_list / mab_choose_list, color='#2ca02c', width=0.15, label="MAB-based [Yoshida, 2020]")
# rects4 = plt.bar([i + 0.45 for i in x], height=linucb_list / linucb_choose_list, color='#d62728', width=0.15, label="cMAB-based (Proposed)")
# rects5 = plt.bar([i + 0.6 for i in x], height=optimal_list / optimal_choose_list, color='#9467bd', width=0.15, label="Optimal")


plt.ylabel("Ratio of Valid Participants", fontsize=20)
plt.ylim(0, 1.8)
plt.yticks([0, 0.5, 1])

plt.xticks([index + 0.22 for index in x], label_list)

ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.xlabel("FL Rounds", fontsize=21)
plt.tick_params(labelsize=18)
leg = ax.legend(fontsize=15)  # , frameon=False)
leg.set_draggable(True)
plt.tight_layout()

plt.show()

