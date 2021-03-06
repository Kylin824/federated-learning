import numpy as np

def create_client_state():

    np.random.seed(2)

    client_state_list = np.zeros((100, 10))

    client_dataset = np.load("dataset_noniid_200_1000.npy", allow_pickle=True)
    client_dataset = client_dataset.item()

    poi_cu_prob = [0.45, 0.5, 0.4, 0.45, 0.4, 0.4, 0.4, 0.4]
    poi_cq_prob = [0.45, 0.4, 0.6, 0.45, 0.5, 0.5, 0.6, 0.6]

    # 8个poi, 0和3分别20个, 其他各10个
    #  0-20 : 0
    for i in range(100):
        if i < 20:
            poi = 0
        elif i >= 20 and i < 30:
            poi = 1
        elif i >= 30 and i < 40:
            poi = 2
        elif i >= 40 and i < 60:
            poi = 3
        elif i >= 60 and i < 70:
            poi = 4
        elif i >= 70 and i < 80:
            poi = 5
        elif i >= 80 and i < 90:
            poi = 6
        elif i >= 90 and i < 100:
            poi = 7

        c = client_state_list[i]

        c[1] = poi  # current_poi
        c[5] = np.random.normal(loc=poi_cu_prob[int(c[1])], scale=0.1, size=1)  # current_computation
        c[6] = np.random.normal(loc=poi_cq_prob[int(c[1])], scale=0.1, size=1)  # current_communication

        # real_poi
        if np.random.rand() <= 0.8:  # 移动概率 (50%概率不动，50%概率移动）
            c[0] = c[1]
        else:
            c[0] = np.random.randint(low=0, high=8)
        c[3] = np.random.normal(loc=poi_cu_prob[int(c[0])], scale=0.1, size=1)
        c[4] = np.random.normal(loc=poi_cq_prob[int(c[0])], scale=0.1, size=1)

        # pred_poi
        if np.random.rand() <= 0.9:  # 预测准确度 (90%概率准确预测，10%概率错误预测)
            c[2] = c[0]
        else:
            c[2] = np.random.randint(low=0, high=8)
        c[7] = np.random.normal(loc=poi_cu_prob[int(c[2])], scale=0.1, size=1)
        c[8] = np.random.normal(loc=poi_cq_prob[int(c[2])], scale=0.1, size=1)

        c[9] = len(client_dataset[i]) / 1000

    # print(client_state_list)

    total_valid_num = 0

    for idx in range(100):

        beta = 0.5

        client_state = client_state_list[idx]
        cur_cq = client_state[5]
        cur_nq = client_state[6]
        client_datasize = client_state[9]
        next_cq = client_state[3]
        next_nq = client_state[4]

        reward = 0
        if cur_cq + cur_nq + next_cq + next_nq >= 2:
            reward = cur_cq + cur_nq + next_cq + next_nq - 2 + client_datasize * beta
            print('valid: ', idx)
            total_valid_num += 1
        else:
            reward = -0.05
            print('fail: ', idx)

    print('total valid num: ', total_valid_num)

    np.save('simulative_client_state.npy', client_state_list)

def create_change_client_state():

    np.random.seed(2)

    client_state_list = np.zeros((100, 10))

    client_dataset = np.load("dataset_noniid_200_1000.npy", allow_pickle=True)
    client_dataset = client_dataset.item()

    poi_cu_prob = [0.6, 0.4, 0.6, 0.6, 0.5, 0.5, 0.6, 0.6]
    poi_cq_prob = [0.6, 0.5, 0.4, 0.6, 0.4, 0.4, 0.4, 0.4]

    # 8个poi, 0和3分别20个, 其他各10个
    #  0-20 : 0
    for i in range(100):
        if i < 20:
            poi = 0
        elif i >= 20 and i < 30:
            poi = 1
        elif i >= 30 and i < 40:
            poi = 2
        elif i >= 40 and i < 60:
            poi = 3
        elif i >= 60 and i < 70:
            poi = 4
        elif i >= 70 and i < 80:
            poi = 5
        elif i >= 80 and i < 90:
            poi = 6
        elif i >= 90 and i < 100:
            poi = 7

        c = client_state_list[i]

        c[1] = poi  # current_poi
        c[5] = np.random.normal(loc=poi_cu_prob[int(c[1])], scale=0.1, size=1)  # current_computation
        c[6] = np.random.normal(loc=poi_cq_prob[int(c[1])], scale=0.1, size=1)  # current_communication

        # real_poi
        if np.random.rand() <= 0.8:  # 移动概率 (50%概率不动，50%概率移动）
            c[0] = c[1]
        else:
            c[0] = np.random.randint(low=0, high=8)
        c[3] = np.random.normal(loc=poi_cu_prob[int(c[0])], scale=0.1, size=1)
        c[4] = np.random.normal(loc=poi_cq_prob[int(c[0])], scale=0.1, size=1)

        # pred_poi
        if np.random.rand() <= 0.9:  # 预测准确度 (90%概率准确预测，10%概率错误预测)
            c[2] = c[0]
        else:
            c[2] = np.random.randint(low=0, high=8)
        c[7] = np.random.normal(loc=poi_cu_prob[int(c[2])], scale=0.1, size=1) # pred_computation
        c[8] = np.random.normal(loc=poi_cq_prob[int(c[2])], scale=0.1, size=1) # pred_communication

        c[9] = len(client_dataset[i]) / 1000

    # print(client_state_list)

    total_valid_num = 0

    for idx in range(100):

        # beta = 0.5

        client_state = client_state_list[idx]
        cur_cq = client_state[5]
        cur_nq = client_state[6]
        client_datasize = client_state[9]
        next_cq = client_state[3]
        next_nq = client_state[4]

        reward = 0
        if cur_cq + cur_nq + next_cq + next_nq >= 2:
            # reward = cur_cq + cur_nq + next_cq + next_nq - 2 + client_datasize * beta
            print('valid: ', idx)
            total_valid_num += 1
        else:
            # reward = -0.05
            print('fail: ', idx)

    print('total valid num: ', total_valid_num)

    np.save('simulative__change_client_state.npy', client_state_list)

def create_change_client_state_plus():

    np.random.seed(2)

    client_state_list = np.load('simulative_client_state.npy')

    print(client_state_list[0])

    for idx in range(100):

        # beta = 0.5

        client_state = client_state_list[idx]
        cur_cq = client_state[5]
        cur_nq = client_state[6]
        client_datasize = client_state[9]
        next_cq = client_state[3]
        next_nq = client_state[4]

        reward = 0
        if cur_cq + cur_nq + next_cq + next_nq >= 2:
            # reward = cur_cq + cur_nq + next_cq + next_nq - 2 + client_datasize * beta
            # client_state[6] = 2 - (cur_cq + next_cq + next_nq) - np.random.normal(loc=0.1, scale=0.1, size=1)
            # client_state[4] = 2 - (cur_cq + cur_nq + next_cq) - np.random.normal(loc=0.1, scale=0.1, size=1)
            # client_state[6] = 2 - (cur_cq + next_cq + next_nq) - np.random.normal(loc=0.1, scale=0.1, size=1)
            client_state[6] = 1 - client_state[6]
            client_state[4] = 1 - client_state[4]
            # client_state[8] = 1 - client_state[8] - 0.075
            print('valid: ', idx)
        else:
            # reward = -0.05
            # client_state[6] = 2 - (cur_cq + next_cq + next_nq) + np.random.normal(loc=0.1, scale=0.1, size=1)
            # client_state[4] = 2 - (cur_cq + cur_nq + next_cq) + np.random.normal(loc=0.1, scale=0.1, size=1)
            client_state[6] = 1 - client_state[6]
            client_state[4] = 1 - client_state[4]
            # client_state[8] = 1 - client_state[8] + 0.075
            print('fail: ', idx)

    np.save('simulative__change_client_state.npy', client_state_list)

def create_real_client_state():
    real_poi = np.loadtxt('../realenv/real_poi_pred.txt')
    # print(real_poi[0])
    last = 3
    total_change = 0
    for i in real_poi[0]:
        if i != last:
            total_change += 1
        last = i
    print(len(real_poi[0]))
    print(total_change)

def read_client_state():
    client_state = np.load('simulative_client_state.npy')
    i = 0
    valid = 0
    for s in client_state:
        print("idx: %2d,  poi:[ next %d, cur %d, pred %d ], next: %.3f, %.3f, cur: %.3f, %.3f, pred: %.3f, %.3f, size: %d" % (
        i, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9] * 1000))
        i += 1
        cur_cq = s[5]
        cur_nq = s[6]
        next_cq = s[3]
        next_nq = s[4]
        if cur_cq + cur_nq + next_cq + next_nq >= 2:
            valid += 1

    print('total valid num: ', valid)


def print_valid_client_each_round():
    # valid_list = np.loadtxt('valid_list_fedcs.txt')
    # valid_list = np.loadtxt('valid_list_random.txt')
    valid_list = np.loadtxt('valid_list_ucb.txt')
    # valid_list = np.loadtxt('valid_list_linucb.txt')
    for round in range(len(valid_list)):
        total_this_round = valid_list[round]
        valid_this_round = total_this_round[np.where(total_this_round != -1)]
        print(valid_this_round)


if __name__ == "__main__":
    # print_valid_client_each_round()
    # read_client_state()
    # create_client_state()
    # create_real_client_state()
    # create_change_client_state()
    create_change_client_state_plus()