import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# 计算delta
def ucb_calculate_delta(t, chosen_count, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(t) / chosen_count[item])


def ucb_choose_arm(upper_bound_probs):
    max = np.max(upper_bound_probs)
    idx = np.where(upper_bound_probs == max)  # 返回tuple，包含符合条件值的下标
    idx = np.array(idx[0])  # 转为array
    if np.size(idx) == 1:
        return idx[0]
    else:
        return np.random.choice(idx, 1)[0]

# 初始化参数矩阵
def linucb_init(num_features, num_arms):
    # 初始化A，b，θ，p
    A = np.array([np.eye(num_features).tolist()] * num_arms)
    b = np.zeros((num_arms, num_features, 1))

    theta = np.zeros((num_arms, num_features, 1))
    p = np.zeros(num_arms)
    return A, b, theta, p

def create_simulate_client_state():

    client_state = np.array(100)



    np.savetxt('./client_state', client_state)


if __name__ == "__main__":

    poi_data = np.loadtxt('real_poi_pred.txt', dtype=int)

    sim_client_state = np.load('sim_client_feature.npy')

    real_poi = poi_data[0]  # 类别数量： [433.  66.  30. 199.   0.   1.  20.   0.]
    pred_poi = poi_data[1]  # 类别数量： [446.  67.  21. 200.   1.   0.  14.   0.]
    poi_success_prob = [0.7, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1]

    # computation , conmunication, datasize, next_poi
    poi_feature = [[7, 7, 7],
                   [2, 8, 5],
                   [8, 2, 5],
                   [8, 8, 8],
                   [2, 2, 2],
                   [2, 2, 2],
                   [8, 2, 5],
                   [1, 1, 1]]

    poi_feature_num = 3

    np.random.seed(1)

    poi_arm_num = 8

    round_num = 100
    round_client_num = 10
    round_client_idx = []
    total_random_reward = 0
    total_fedcs_reward = 0

    # ucb attributes
    ucb_estimated_rewards = np.zeros(poi_arm_num)
    ucb_chosen_count = np.zeros(poi_arm_num)
    total_ucb_reward = 0

    # linucb attributes
    A, b, theta, p = linucb_init(poi_feature_num, poi_arm_num)
    total_linucb_reward = 0

    total_oracle_reward = 0

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

        # fedcs choose 1
        round_client_idx = np.random.randint(low=0, high=len(real_poi) - 1, size=math.ceil(1.5*round_client_num))
        total_fedcs_selected_num = 0
        for idx in round_client_idx:
            # 只选状态最好的那些
            if real_poi[idx] == 0 and total_fedcs_selected_num < 10:
                total_fedcs_selected_num += 1
                next_poi = real_poi[idx + 1]
                reward = np.random.binomial(n=1, p=poi_success_prob[next_poi])
                total_fedcs_reward += reward


        # ucb choose
        for i in range(round_client_num):
            upper_bound_probs = [ucb_estimated_rewards[item] + ucb_calculate_delta(round * 10 + i, ucb_chosen_count, item) for item in
                                 range(poi_arm_num)]
            choosen_poi_arm = ucb_choose_arm(upper_bound_probs)

            # ucb reward
            reward = np.random.binomial(n=1, p=poi_success_prob[choosen_poi_arm])

            total_ucb_reward += reward

            ucb_estimated_rewards[choosen_poi_arm] = (ucb_chosen_count[choosen_poi_arm] * ucb_estimated_rewards[choosen_poi_arm] + reward) / (
                    ucb_chosen_count[choosen_poi_arm] + 1)

            ucb_chosen_count[choosen_poi_arm] += 1

        # linucb choose
        alpha = 0.5

        for i in range(round_client_num):

            # 求每个臂的p
            for a in range(0, poi_arm_num):
                x_t = np.expand_dims(poi_feature[a], axis=1)
                # 求逆
                A_inv = np.linalg.inv(A[a])
                # 相乘
                theta[a] = np.matmul(A_inv, b[a])
                # 求臂的p
                p[a] = np.matmul(theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

            best_predicted_poi_arm = int(np.argmax(p))
            reward = np.random.binomial(n=1, p=poi_success_prob[best_predicted_poi_arm])

            total_linucb_reward += reward

        # oracle choose
        for i in range(round_client_num):
            total_oracle_reward += np.random.binomial(n=1, p=poi_success_prob[0])


    print("total_random_reward: ", total_random_reward)
    print("total_fedcs_reward: ", total_fedcs_reward)
    print("total_ucb_reward: ", total_ucb_reward)
    print("estimated_ucb_reward: \n", ucb_estimated_rewards)

    print("total_linucb_reward: \n", total_linucb_reward)


    # 归一化p
    p_max = np.max(p)
    p_min = np.min(p)
    p = (p - p_min) / (p_max - p_min)
    print("linucb p: \n", p)

    print("total_oracle_reward: \n", total_oracle_reward)

