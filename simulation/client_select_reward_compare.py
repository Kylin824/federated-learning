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


    # beta = 0.4  # best non-iid 200-1000 setting

    beta = 0  # iid 600 setting

    client_state = sim_client_state[client_idx]
    cur_cq = client_state[5]
    cur_nq = client_state[6]
    client_datasize = client_state[9]
    next_cq = client_state[3]
    next_nq = client_state[4]

    reward = 0

    if cur_cq + cur_nq + next_cq + next_nq >= 2:
        reward = cur_cq + cur_nq + next_cq + next_nq - 1 + client_datasize * beta
        # reward = 1
    # else:
    #     reward = -0.1

    return reward


if __name__ == "__main__":

    sim_client_state = np.load('simulative_client_state.npy')

    client_feature_num = 5

    np.random.seed(0)

    client_arm_num = 100

    round_num = 200
    round_client_num = 10
    round_client_idx = []

    total_random_reward = 0
    random_reward_list = []
    random_total_valid = 0
    random_chosen_count = np.zeros(client_arm_num)
    random_chosen_valid_list = []

    total_fedcs_reward = 0
    fedcs_reward_list = []
    fedcs_total_valid = 0
    fedcs_chosen_count = np.zeros(client_arm_num)
    fedcs_chosen_valid_list = []

    # ucb attributes
    ucb_estimated_rewards = np.zeros(client_arm_num)
    ucb_chosen_count = np.zeros(client_arm_num)
    total_ucb_reward = 0
    ucb_total_valid = 0
    ucb_reward_list = []
    ucb_chosen_valid_list = []

    # linucb attributes
    A, b, theta, p = linucb_init(client_feature_num, client_arm_num)
    linucb_chosen_count = np.zeros(client_arm_num)
    total_linucb_reward = 0
    linucb_reward_list = []
    linucb_total_valid = 0
    linucb_client_feature = []
    linucb_chosen_valid_list = []

    client_idxs = np.arange(100)

    for round in range(round_num):

        # random choose
        round_client_idx = np.random.choice(client_idxs, size=round_client_num, replace=False)

        round_valid_client_idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        valid_count = 0

        # get random reward
        for idx in round_client_idx:

            reward = calculate_reward(sim_client_state, idx)

            total_random_reward += reward

            if reward > 0:
                random_total_valid += 1
                round_valid_client_idx[valid_count] = idx
                valid_count += 1


            random_chosen_count[idx] += 1

        # if len(round_valid_client_idx) < 10:
        #     for k in range(10 - len(round_valid_client_idx)):
        #         round_valid_client_idx.append(-1)

        random_chosen_valid_list.append(round_valid_client_idx)

        random_reward_list.append(total_random_reward)

        # fedcs choose 1
        round_client_idx = np.random.choice(client_idxs, size=int(2.5*round_client_num), replace=False)

        total_fedcs_selected_num = 0

        round_valid_client_idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        valid_count = 0

        for idx in round_client_idx:

            client_state = sim_client_state[idx]
            cur_cq = client_state[5]
            cur_nq = client_state[6]

            # 只选状态最好的那些
            if cur_cq + cur_nq >= 1.0 and total_fedcs_selected_num < 10:

                reward = calculate_reward(sim_client_state, idx)

                if reward > 0:
                    fedcs_total_valid += 1
                    round_valid_client_idx[valid_count] = idx
                    valid_count += 1

                total_fedcs_selected_num += 1
                fedcs_chosen_count[idx] += 1
                total_fedcs_reward += reward

        fedcs_chosen_valid_list.append(round_valid_client_idx)

        fedcs_reward_list.append(total_fedcs_reward)


        # ucb choose

        # # ucb init  前十轮 每个都选一次
        # if round < 10:
        #     round_valid_client_idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        #     valid_count = 0
        #
        #     for idx in range(round * 10, round * 10 + 10):
        #         reward = calculate_reward(sim_client_state, idx)
        #
        #         if reward > 0:
        #             ucb_total_valid += 1
        #             round_valid_client_idx[valid_count] = idx
        #             valid_count += 1
        #
        #         total_ucb_reward += reward
        #         ucb_estimated_rewards[idx] = reward
        #
        #         ucb_chosen_count[idx] += 1
        #
        #     ucb_chosen_valid_list.append(round_valid_client_idx)
        #     ucb_reward_list.append(total_ucb_reward)
        #
        # else:

        round_valid_client_idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        valid_count = 0

        ucb_client_idxs = client_idxs

        round_client_idx = []

        for i in range(round_client_num):

            round_client_idx = np.random.choice(ucb_client_idxs, size=round_client_num, replace=False)

            upper_bound_probs = [ucb_estimated_rewards[item] + ucb_calculate_delta(round * round_client_num + i, ucb_chosen_count, item) for item in
                                 round_client_idx]

            chosen_client_arm = ucb_choose_arm(upper_bound_probs) # 0-10

            chosen_client_arm = round_client_idx[chosen_client_arm] # real idx

            reward = calculate_reward(sim_client_state, chosen_client_arm)

            if reward > 0:
                ucb_total_valid += 1
                round_valid_client_idx[valid_count] = chosen_client_arm
                valid_count += 1

            total_ucb_reward += reward

            ucb_estimated_rewards[chosen_client_arm] = (ucb_chosen_count[chosen_client_arm] * ucb_estimated_rewards[chosen_client_arm] + reward) / (
                    ucb_chosen_count[chosen_client_arm] + 1)

            ucb_chosen_count[chosen_client_arm] += 1

            # 本轮选过的不再参与选择
            ucb_client_idxs = np.delete(ucb_client_idxs, np.where(ucb_client_idxs == chosen_client_arm))


        ucb_chosen_valid_list.append(round_valid_client_idx)

        ucb_reward_list.append(total_ucb_reward)

        # linucb choose
        alpha = 1

        if round == 0:
            for a in range(100):
                linucb_client_feature.append(sim_client_state[a][5:])  # cur_cq, cur_nq, pred_cq, pred_nq, datasize
                # x_t = np.expand_dims(linucb_client_feature[a], axis=1)
                # # 求逆
                # A_inv = np.linalg.inv(A[a])
                # # 相乘
                # theta[a] = np.matmul(A_inv, b[a])
                # # 求臂的p
                # p[a] = np.matmul(theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

        linucb_client_idxs = client_idxs

        round_valid_client_idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        valid_count = 0

        for i in range(round_client_num):

            round_client_idx = np.random.choice(linucb_client_idxs, size=round_client_num, replace=False)

            # 求每个臂的p
            for idx in round_client_idx:
                x_t = np.expand_dims(linucb_client_feature[idx], axis=1)
                # 求逆
                A_inv = np.linalg.inv(A[idx])
                # 相乘
                theta[idx] = np.matmul(A_inv, b[idx])
                # 求臂的p
                p[idx] = np.matmul(theta[idx].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

            best_pred_client_arm = int(np.argmax(p[round_client_idx]))

            chosen_client_arm = round_client_idx[best_pred_client_arm]

            reward = calculate_reward(sim_client_state, chosen_client_arm)

            if reward > 0:
                linucb_total_valid += 1
                round_valid_client_idx[valid_count] = chosen_client_arm
                valid_count += 1

            total_linucb_reward += reward

            linucb_chosen_count[chosen_client_arm] += 1

            x_t = np.expand_dims(linucb_client_feature[chosen_client_arm], axis=1)

            # 更新Aat，bat
            A[chosen_client_arm] = A[chosen_client_arm] + np.matmul(x_t, x_t.T)
            b[chosen_client_arm] = b[chosen_client_arm] + reward * x_t

            # 本轮选过的client不再参与选择
            linucb_client_idxs = np.delete(linucb_client_idxs, np.where(linucb_client_idxs == chosen_client_arm))

        linucb_chosen_valid_list.append(round_valid_client_idx)

        linucb_reward_list.append(total_linucb_reward)


    print("total_random_reward: ", total_random_reward)
    print("total_fedcs_reward: ", total_fedcs_reward)
    print("total_ucb_reward: ", total_ucb_reward)
    print("total_linucb_reward: ", total_linucb_reward)

    # print("estimated_ucb_reward: \n", ucb_estimated_rewards)

    print("\ndata size: ")
    for i in range(100):
        print(int(sim_client_state[i][9] * 1000), end='  ')

    print("\nclient index: ")
    for i in range(100):
        print("%3d" %i, end='  ')

    print("\nrandom chosen count: ", np.sum(random_chosen_count))
    for i in range(100):
        print("%3d" % int(random_chosen_count[i]), end='  ')

    print("\nfedcs chosen count: ", np.sum(fedcs_chosen_count))
    for i in range(100):
        print("%3d" % int(fedcs_chosen_count[i]), end='  ')

    print("\nucb chosen count: ", np.sum(ucb_chosen_count))
    for i in range(100):
        print("%3d" % int(ucb_chosen_count[i]), end='  ')

    print("\nlinucb chosen count: ", np.sum(linucb_chosen_count))
    for i in range(100):
        print("%3d" % int(linucb_chosen_count[i]), end='  ')

    print("\nvalid client idx: ")
    valid_client_idx = []
    for i in range(100):
        client_state = sim_client_state[i]
        cur_cq = client_state[5]
        cur_nq = client_state[6]
        next_cq = client_state[3]
        next_nq = client_state[4]
        # if cur_cq + cur_nq + next_cq + next_nq >= 2:
        if cur_cq + cur_nq >= 1 and next_cq + next_nq >= 1:
            valid_client_idx.append(1)
        else:
            valid_client_idx.append(0)

    for i in range(100):
        print("%3d" % valid_client_idx[i], end='  ')


    x = np.arange(round_num)
    plt.xlabel("round")
    plt.ylabel("cumulative reward")
    plt.plot(x, random_reward_list, label='random')
    plt.plot(x, fedcs_reward_list, label='fedcs')
    plt.plot(x, ucb_reward_list, label='ucb')
    plt.plot(x, linucb_reward_list, label='linucb')
    plt.legend()
    plt.show()

    print("\ntotal random valid", random_total_valid)
    print("total fedcs valid", fedcs_total_valid)
    print("total ucb valid", ucb_total_valid)
    print("total linucb valid", linucb_total_valid)


    # 保存历史选择
    # np.savetxt('valid_list_random.txt', random_chosen_valid_list)
    # np.savetxt('valid_list_fedcs.txt', fedcs_chosen_valid_list)
    # np.savetxt('valid_list_ucb.txt', ucb_chosen_valid_list)
    # np.savetxt('valid_list_linucb.txt', linucb_chosen_valid_list)
