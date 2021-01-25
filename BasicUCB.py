import numpy as np

T = 1000  # T������
N = 10  # N���ϻ���

true_rewards = np.random.uniform(low=0, high=1, size=N)  # ÿ���ϻ�����ʵ����Ǯ����
estimated_rewards = np.zeros(N)  # ÿ���ϻ�����Ǯ�Ĺ۲���ʣ���ʼ��Ϊ0
chosen_count = np.zeros(N)  # ÿ���ϻ�����ǰ�Ѿ�̽���Ĵ�������ʼ��Ϊ0
total_reward = 0


# ����delta
def calculate_delta(T, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])

# ����ÿ���ϻ�����p+delta��ͬʱ����ѡ��
def UCB(t, N):
    upper_bound_probs = [estimated_rewards[item] + calculate_delta(t, item) for item in range(N)]
    item = np.argmax(upper_bound_probs)
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward


for t in range(1, T):  # ���ν���T������
    # ѡ��һ���ϻ��������õ��Ƿ���Ǯ�Ľ��
    item, reward = UCB(t, N)
    total_reward += reward  # һ���ж��ٿ��˽������Ƽ�

    # ����ÿ���ϻ�������Ǯ����
    estimated_rewards[item] = ((t - 1) * estimated_rewards[item] + reward) / t
    chosen_count[item] += 1