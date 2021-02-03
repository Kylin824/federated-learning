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


def calculate_reward(sim_client_state, client_idx):
    client_state = sim_client_state[client_idx]
    cur_cq = client_state[1]
    cur_nq = client_state[2]
    client_datasize = client_state[6]
    next_cq = client_state[8]
    next_nq = client_state[9]

    reward = 0
    if cur_cq + cur_nq + next_cq + next_nq >= 2:
        reward = client_datasize / 1000
        # reward = cur_cq + cur_nq + next_cq + next_nq - 2 + client_datasize / 1000
        # reward = cur_cq + cur_nq + next_cq + next_nq - 2 + client_datasize / 2000
        # reward = cur_cq + cur_nq + next_cq + next_nq - 2

    return reward


if __name__ == "__main__":

    sim_client_state = np.load('sim_client_feature.npy')

    client_feature_num = 7

    np.random.seed(1)

    client_arm_num = 100

    round_num = 200
    round_client_num = 10
    round_client_idx = []


    total_random_reward = 0
    random_reward_list = []
    random_chosen_count = np.zeros(client_arm_num)

    total_fedcs_reward = 0
    fedcs_reward_list = []
    fedcs_chosen_count = np.zeros(client_arm_num)

    # ucb attributes
    ucb_estimated_rewards = np.zeros(client_arm_num)
    ucb_chosen_count = np.zeros(client_arm_num)
    total_ucb_reward = 0
    ucb_reward_list = []

    # linucb attributes
    A, b, theta, p = linucb_init(client_feature_num, client_arm_num)
    linucb_chosen_count = np.zeros(client_arm_num)
    total_linucb_reward = 0
    linucb_reward_list = []

    client_idxs = np.arange(100)

    for round in range(round_num):

        # random choose
        round_client_idx = np.random.choice(client_idxs, size=round_client_num, replace=False)

        # get random reward
        for idx in round_client_idx:

            reward = calculate_reward(sim_client_state, idx)

            total_random_reward += reward

            random_chosen_count[idx] += 1

        random_reward_list.append(total_random_reward)

        # fedcs choose 1
        round_client_idx = np.random.choice(client_idxs, size=int(2*round_client_num), replace=False)

        total_fedcs_selected_num = 0

        for idx in round_client_idx:

            client_state = sim_client_state[idx]
            cur_cq = client_state[1]
            cur_nq = client_state[2]
            client_datasize = client_state[6]
            next_cq = client_state[8]
            next_nq = client_state[9]

            # 只选状态最好的那些
            if cur_cq + cur_nq >= 1.0 and total_fedcs_selected_num < 10:

                reward = calculate_reward(sim_client_state, idx)
                total_fedcs_selected_num += 1
                fedcs_chosen_count[idx] += 1
                total_fedcs_reward += reward

        fedcs_reward_list.append(total_fedcs_reward)

        # ucb choose

        # # ucb init  每个都选一次
        # if round == 0:
        #     for idx in range(100):
        #         ucb_estimated_rewards[idx] = calculate_reward(sim_client_state, idx)
        #
        # else:

        for i in range(round_client_num):

            round_client_idx = np.random.choice(client_idxs, size=round_client_num, replace=False)

            upper_bound_probs = [ucb_estimated_rewards[item] + ucb_calculate_delta(round * round_client_num + i, ucb_chosen_count, item) for item in
                                 round_client_idx]

            chosen_client_arm = ucb_choose_arm(upper_bound_probs)

            chosen_client_arm = round_client_idx[chosen_client_arm]

            reward = calculate_reward(sim_client_state, chosen_client_arm)

            total_ucb_reward += reward

            ucb_estimated_rewards[chosen_client_arm] = (ucb_chosen_count[chosen_client_arm] * ucb_estimated_rewards[chosen_client_arm] + reward) / (
                    ucb_chosen_count[chosen_client_arm] + 1)

            ucb_chosen_count[chosen_client_arm] += 1


        ucb_reward_list.append(total_ucb_reward)

        # linucb choose
        alpha = 0.25

        for i in range(round_client_num):

            round_client_idx = np.random.choice(client_idxs, size=round_client_num, replace=False)

            client_feature = []

            for idx in round_client_idx:
                client_feature.append(sim_client_state[idx][:7])

            # 求每个臂的p
            for a in range(10):
                x_t = np.expand_dims(client_feature[a], axis=1)
                # 求逆
                A_inv = np.linalg.inv(A[a])
                # 相乘
                theta[a] = np.matmul(A_inv, b[a])
                # 求臂的p
                p[a] = np.matmul(theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

            best_pred_client_arm = int(np.argmax(p))

            chosen_client_arm = round_client_idx[best_pred_client_arm]

            reward = calculate_reward(sim_client_state, chosen_client_arm)

            total_linucb_reward += reward

            linucb_chosen_count[chosen_client_arm] += 1

        linucb_reward_list.append(total_linucb_reward)


    print("total_random_reward: ", total_random_reward)
    print("total_fedcs_reward: ", total_fedcs_reward)
    print("total_ucb_reward: ", total_ucb_reward)
    print("total_linucb_reward: ", total_linucb_reward)

    print("estimated_ucb_reward: \n", ucb_estimated_rewards)

    print("\ndata size: ")
    for i in range(100):
        print(int(sim_client_state[i][6]), end='  ')

    print("")

    print("\nclient index: ")
    for i in range(100):
        print("%3d" %i, end='  ')

    print("")

    print("\nrandom chosen count: ")
    for i in range(100):
        print("%3d" % int(random_chosen_count[i]), end='  ')

    print("\nfedcs chosen count: ")
    for i in range(100):
        print("%3d" % int(fedcs_chosen_count[i]), end='  ')

    print("\nucb chosen count: ")
    for i in range(100):
        print("%3d" % int(ucb_chosen_count[i]), end='  ')

    print("\nlinucb chosen count: ")
    for i in range(100):
        print("%3d" % int(linucb_chosen_count[i]), end='  ')

    x = np.arange(round_num)
    plt.xlabel("round")
    plt.ylabel("cumulative reward")
    plt.plot(x, random_reward_list, label='random')
    plt.plot(x, fedcs_reward_list, label='fedcs')
    plt.plot(x, ucb_reward_list, label='ucb')
    plt.plot(x, linucb_reward_list, label='linucb')
    plt.legend()
    plt.show()
