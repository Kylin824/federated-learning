import numpy as np
import math

# cur_poi, cur_cu, cur_cq, real_poi, real_cu, real_cq, pred_poi, pred_cu, pred_cq, datasize
client_state_list = np.zeros((100, 10))

client_dataset = np.load("../utils/noniid.npy", allow_pickle=True)
client_dataset = client_dataset.item()

poi_cu_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
poi_cq_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

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

    c[0] = poi  # current_poi
    c[1] = np.random.normal(loc=poi_cu_prob[int(c[0])], scale=0.1, size=1)  # current_computation
    c[2] = np.random.normal(loc=poi_cq_prob[int(c[0])], scale=0.1, size=1)  # current_communication

    # real_poi
    if np.random.rand() <= 0.8:  # 移动概率 (80%概率不动，20%概率移动）
        c[7] = c[0]
    else:
        c[7] = np.random.randint(low=0, high=8)
    c[8] = np.random.normal(loc=poi_cu_prob[int(c[3])], scale=0.1, size=1)
    c[9] = np.random.normal(loc=poi_cq_prob[int(c[3])], scale=0.1, size=1)

    # pred_poi
    if np.random.rand() <= 0.9:  # 预测准确度 (90%概率准确预测，10%概率错误预测)
        c[3] = c[7]
    else:
        c[3] = np.random.randint(low=0, high=8)
    c[4] = np.random.normal(loc=poi_cu_prob[int(c[6])], scale=0.1, size=1)
    c[5] = np.random.normal(loc=poi_cq_prob[int(c[6])], scale=0.1, size=1)

    c[6] = len(client_dataset[i])

print(client_state_list)
np.save('../sim_client_feature.npy', client_state_list)