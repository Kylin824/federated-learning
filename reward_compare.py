import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

poi_data = np.loadtxt('poi_pred.txt', dtype=int)
real_poi = poi_data[0]  # 类别数量： [433.  66.  30. 199.   0.   1.  20.   0.]
pred_poi = poi_data[1]  # 类别数量： [446.  67.  21. 200.   1.   0.  14.   0.]
poi_success_prob = [0.7, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1]

np.random.seed(0)

round_num = 10

round_client_num = 10
round_client_idx = []

total_random_reward = 0
total_fedcs_reward = 0

for round in range(round_num):

    # random choose
    round_client_idx = np.random.randint(low=0, high=len(real_poi) - 1, size=round_client_num)
    # print("current poi: ", real_poi[round_client_idx])
    # print("next poi: ", real_poi[round_client_idx + 1])

    # get random reward
    for idx in round_client_idx:
        next_poi = real_poi[idx + 1]
        reward = np.random.binomial(n=1, p=poi_success_prob[next_poi])
        total_random_reward += reward

    # fedcs choose
    round_client_idx = np.random.randint(low=0, high=len(real_poi) - 1, size=round_client_num)
    for idx in round_client_idx:
        # 只选状态最好的那些
        if real_poi[idx] == 0:
            next_poi = real_poi[idx + 1]
            reward = np.random.binomial(n=1, p=poi_success_prob[next_poi])
            total_fedcs_reward += reward






print("total_random_reward: ", total_random_reward)
print("total_fedcs_reward: ", total_fedcs_reward)

