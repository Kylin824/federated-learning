import numpy as np
import matplotlib.pyplot as plt

# random_reward_list = np.loadtxt('./reward/iid_random_reward_list.txt')
# fedcs_reward_list = np.loadtxt('./reward/iid_fedcs_reward_list.txt')
# ucb_reward_list = np.loadtxt('./reward/iid_ucb_reward_list.txt')
# linucb_reward_list = np.loadtxt('./reward/iid_linucb_reward_list.txt')

random_reward_list = np.loadtxt('./reward/noniid_random_reward_list.txt')
fedcs_reward_list = np.loadtxt('./reward/noniid_fedcs_reward_list.txt')
ucb_reward_list = np.loadtxt('./reward/noniid_ucb_reward_list.txt')
linucb_reward_list = np.loadtxt('./reward/noniid_linucb_reward_list.txt')

optimal_reward_list = np.linspace(0, 4000, 401)

len = 200

interval = 20

random_reward_list = [random_reward_list[i] for i in range(0, len + interval, interval)]
fedcs_reward_list = [fedcs_reward_list[i] for i in range(0, len + interval, interval)]
ucb_reward_list = [ucb_reward_list[i] for i in range(0, len + interval, interval)]
linucb_reward_list = [linucb_reward_list[i] for i in range(0, len + interval, interval)]
# optimal_reward_list = [optimal_reward_list[i] for i in range(0, len + interval, interval)]

x = np.linspace(0, len, int(len / interval + 1))

# x = x[:int(len / interval)]

fig, ax = plt.subplots()

# ax.plot(x, optimal_reward_list[:len], '-', linewidth=2, color='#9467bd', label="Optimal")
# ax.plot(x, linucb_reward_list[:len], '-', linewidth=2, color='#d62728', label="Proposed")
# ax.plot(x, ucb_reward_list[:len], '-', linewidth=2, color='#2ca02c', label="MAB-based")
# ax.plot(x, fedcs_reward_list[:len], '-', linewidth=2, color='#ff7f0e', label="FedCS [Nishio, 2019]")
# ax.plot(x, random_reward_list[:len], '-', linewidth=2, color='#1f77b4', label="FedAvg [Google Team]")

# ax.plot(x, optimal_reward_list, '^-', linewidth=2, color='#9467bd', label="Optimal")
# ax.plot(x, linucb_reward_list, '^-', markersize=7, linewidth=2, color='#d62728', label="cMAB-based (Proposed)")
# ax.plot(x, ucb_reward_list, 's-', markersize=7, linewidth=2, color='#2ca02c', label="MAB-based [Yoshida, 2020]")
# ax.plot(x, fedcs_reward_list, 'v-', markersize=7, linewidth=2, color='#ff7f0e', label="FedCS [Nishio, 2019]")
# ax.plot(x, random_reward_list, 'o-', markersize=7, linewidth=2, color='#1f77b4', label="FedAvg [McMahan, 2016]")

ax.plot(x, linucb_reward_list, '^-', linestyle='-', markersize=7, linewidth=2, color='#d62728', label="cMAB-based (Proposed)")
ax.plot(x, ucb_reward_list, 's--', linestyle='--', markersize=7, linewidth=2, color='#2ca02c', label="MAB-based [Yoshida, 2020]")
ax.plot(x, fedcs_reward_list, 'v-.', linestyle='-.', markersize=7, linewidth=2, color='#ff7f0e', label="FedCS [Nishio, 2019]")
ax.plot(x, random_reward_list, 'd:', linestyle=':', markersize=7, linewidth=2, color='#1f77b4', label="FedAvg [McMahan, 2016]")

# plt.ylim(0, 1300000)

ax.yaxis.get_major_formatter().set_powerlimits((1, 2))
# ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

plt.ylabel('Cumulative Reward', fontdict={'size': 20})
plt.xlabel('Round', fontdict={'size': 20})
plt.tick_params(labelsize=18)
leg = ax.legend(fontsize=16)  # , frameon=False)
leg.set_draggable(True)

plt.tight_layout()
# plt.savefig('./imgs/Offline Training Reward.pdf', bbox_inches='tight')
plt.show()
plt.close()
