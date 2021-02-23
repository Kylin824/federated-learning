import matplotlib.pyplot as plt
import numpy as np

label_list = ['50', '100', '150', '200']  # 横坐标刻度显示值

# fedavg_list = np.array([184, 384, 600, 810])  # 纵坐标值1
# fedcs_list = np.array([116, 236, 372, 500])  # 纵坐标值2
# mab_list = np.array([183, 383, 599, 808])  # 纵坐标值3
# linucb_list = np.array([184, 384, 600, 810])  # 纵坐标值4
# optimal_list = np.array([184, 384, 600, 810])  # 纵坐标值4

fedavg_list = np.array([117, 268, 388, 516])  # 纵坐标值1
fedcs_list = np.array([219, 432, 646, 848])  # 纵坐标值2
mab_list = np.array([239, 520, 891, 1271])  # 纵坐标值3
linucb_list = np.array([429, 895, 1361, 1822])  # 纵坐标值4
optimal_list = np.array([500, 1000, 1500, 2000])  # 纵坐标值4

fedavg_choose_list = np.array([500, 1000, 1500, 2000])
fedcs_choose_list = np.array([500, 1000, 1500, 2000])
ddqn_choose_list = np.array([500, 1000, 1500, 2000])
pred_choose_list = np.array([500, 1000, 1500, 2000])
offline_choose_list = np.array([500, 1000, 1500, 2000])

x = range(len(label_list))

error_params = dict(elinewidth=2, ecolor='black', capsize=3)  # 设置误差标记参数

fig, ax = plt.subplots()

rects1 = plt.bar(x, height=fedavg_list, width=0.15, color='#2ca02c', label="FedAvg [Google Team]",
                 hatch='.')  # , yerr=err_fa_list, error_kw=error_params)
rects11 = plt.bar(x, height=fedavg_choose_list - fedavg_list, width=0.15, color='#2ca02c',
                  alpha=0.5, bottom=fedavg_list)

rects2 = plt.bar([i + 0.15 for i in x], height=fedcs_list, color='#1f77b4', width=0.15, label="FedCS [Nishio, 2019]",
                 hatch='xx')  # , yerr=err_fc_list, error_kw=error_params)
rects21 = plt.bar([i + 0.15 for i in x], height=fedcs_choose_list - fedcs_list, width=0.15, color='#1f77b4',
                  alpha=0.5, bottom=fedcs_list)

rects3 = plt.bar([i + 0.3 for i in x], height=mab_list, color='#ff7f0e', width=0.15, label="DDQN-based (Proposed)",
                 hatch=' ')  # , yerr=err_dq_list, error_kw=error_params)
rects31 = plt.bar([i + 0.3 for i in x], height=pred_choose_list - mab_list, width=0.15, color='#ff7f0e',
                  alpha=0.5, bottom=mab_list)

rects4 = plt.bar([i + 0.45 for i in x], height=optimal_list, color='#d62728', width=0.15,
                 label="Offline", hatch='\\')  # , yerr=err_pr_list, error_kw=error_params)
rects41 = plt.bar([i + 0.45 for i in x], height=offline_choose_list - optimal_list, width=0.15, color='#d62728',
                  alpha=0.5, bottom=optimal_list)

plt.xticks([index + 0.22 for index in x], label_list)
plt.ylim(0, 2999)
plt.ylabel("# of Valid / Invalid Participants", fontsize=18)
ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.xlabel("FL Rounds", fontsize=20)
plt.tick_params(labelsize=16)
leg = ax.legend(fontsize=15)  # , frameon=False)
leg.set_draggable(True)
plt.tight_layout()

left, bottom, width, height = 0.13, 0.42, 0.3, 0.3
ax2 = fig.add_axes([left, bottom, width, height])

men_means = (20, 10, 10, 10, 10)
women_means = (25, 10, 10, 10, 10)
ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.35  # the width of the bars

rects1 = ax2.bar(ind - width / 2, men_means, width, color='white')
rects2 = ax2.bar(0.15, 15, width, color='#49b9c2', hatch='')
rects3 = ax2.bar(0.15, 25, width, color='#49b9c2', alpha=0.5, hatch='')

ax2.annotate(color='#000000', s="# of invalid\n Participants",
             xy=(0.45, 20), xycoords='data',
             xytext=(0.8, 15), textcoords='data',
             size=15,
             arrowprops=dict(arrowstyle="-[", color='#000000', connectionstyle='angle3'))

ax2.annotate(color='#000000', s="# of valid\n Participants",
             xy=(0.45, 7), xycoords='data',
             xytext=(0.8, 3), textcoords='data',
             size=15,
             arrowprops=dict(arrowstyle="-[", color='#000000', connectionstyle='angle3'))

plt.ylim(0, 40)
ax2.axis('off')

plt.show()

