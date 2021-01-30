import numpy as np
import matplotlib.pyplot as plt


# 计算delta
def calculate_delta(t, chosen_count, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(t) / chosen_count[item])


def choose_arm(upper_bound_probs):
    max = np.max(upper_bound_probs)
    idx = np.where(upper_bound_probs == max)  # 返回tuple，包含符合条件值的下标
    idx = np.array(idx[0])  # 转为array
    if np.size(idx) == 1:
        return idx[0]
    else:
        return np.random.choice(idx, 1)[0]


def train():
    # 时间
    T = []

    # 可选的臂（根据数据）
    num_arms = 10

    # 总回报
    total_reward = 0

    total_best_reward = 0

    total_reward_with_T = []

    total_regret_with_T = []

    np.random.seed(23)

    true_rewards_prop = np.random.uniform(low=0, high=1, size=num_arms)  # 每个老虎机真实的吐钱概率
    true_max_prop_arm = np.argmax(true_rewards_prop)

    print("true reward prop: \n", true_rewards_prop)

    print("\ntrue_max_prop_arm: ", true_max_prop_arm)

    estimated_rewards = np.zeros(num_arms)  # 每个老虎机吐钱的观测概率，初始都为0

    chosen_count = np.zeros(num_arms)  # 每个老虎机当前已经探索的次数，初始都为0

    for i in range(10):
        choosen_arm = i % 10
        reward = np.random.binomial(n=1, p=true_rewards_prop[choosen_arm])
        best_reward = np.random.binomial(n=1, p=true_rewards_prop[true_max_prop_arm])

        total_reward += reward
        total_best_reward += best_reward
        T.append(i)
        total_reward_with_T.append(total_reward)
        total_regret_with_T.append(total_best_reward - total_reward)

        if i < 10:
            estimated_rewards[choosen_arm] = reward
        else:
            # estimated_rewards[choosen_arm] = ((i - 1) * estimated_rewards[choosen_arm] + reward) / i
            estimated_rewards[choosen_arm] = (chosen_count[choosen_arm] * estimated_rewards[choosen_arm] + reward) / (
                    chosen_count[choosen_arm] + 1)
        chosen_count[choosen_arm] += 1

    print("\ninit estimated reward: ")
    print(estimated_rewards)

    # 初始化
    for t in range(10, 10000):
        upper_bound_probs = [estimated_rewards[item] + calculate_delta(t, chosen_count, item) for item in
                             range(num_arms)]

        # 选择最大置信区间上界的arm
        # choosen_arm = np.argmax(upper_bound_probs)

        choosen_arm = choose_arm(upper_bound_probs)

        reward = np.random.binomial(n=1, p=true_rewards_prop[choosen_arm])
        best_reward = np.random.binomial(n=1, p=true_rewards_prop[true_max_prop_arm])

        total_reward += reward
        total_best_reward += best_reward

        T.append(t)

        total_reward_with_T.append(total_reward)
        total_regret_with_T.append(total_best_reward - total_reward)

        # 更新每个老虎机的吐钱概率
        # estimated_rewards[choosen_arm] = ((t - 1) * estimated_rewards[choosen_arm] + reward) / t
        estimated_rewards[choosen_arm] = (chosen_count[choosen_arm] * estimated_rewards[choosen_arm] + reward) / (
                chosen_count[choosen_arm] + 1)

        chosen_count[choosen_arm] += 1

        # if t % 200 == 0:
        #     print("estimated reward: ")
        #     print(estimated_rewards)

    print("\ntotal reward: ", total_reward)
    print("\nbest reward: ", total_best_reward)
    print("choosen arm: ", chosen_count)

    # CTR趋势画图
    plt.xlabel("T")
    plt.ylabel("Total regret")
    plt.plot(T, total_regret_with_T)
    # 存入路径
    plt.savefig('./regret.png')


if __name__ == "__main__":
    # 训练
    train()
