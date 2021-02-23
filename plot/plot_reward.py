import numpy as np
import matplotlib.pyplot as plt

random_reward_list = np.loadtxt('./reward/random_reward_list.txt')
fedcs_reward_list = np.loadtxt('./reward/fedcs_reward_list.txt')
ucb_reward_list = np.loadtxt('./reward/ucb_reward_list.txt')
linucb_reward_list = np.loadtxt('./reward/linucb_reward_list.txt')

len = 200
x = np.arange(len)
plt.xlabel("round")
plt.ylabel("cumulative reward")
plt.plot(x, random_reward_list[:len], label='random')
plt.plot(x, fedcs_reward_list[:len], label='fedcs')
plt.plot(x, ucb_reward_list[:len], label='ucb')
plt.plot(x, linucb_reward_list[:len], label='linucb')
plt.legend()
plt.show()