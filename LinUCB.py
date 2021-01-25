import numpy as np

# ref: https://zhuanlan.zhihu.com/p/38875273

class LinUCB:

    def __init__(self):
        self.alpha = 0.25  # 探索程序
        self.r1 = 0.6  # 正反馈奖励值 if worse -> 0.7, 0.8
        self.r0 = -16  # 负反馈奖励值 if worse, -19, -21

        self.d = 6  # 特征维度
        self.Aa = {}  # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}  # AaI : store the inverse of all Aa matrix

        self.ba = {}  # ba : collection of vectors to compute disjoin part, d*1
        self.a_max = 0
        self.theta = {}
        self.x = None
        self.xT = None

    """
    初始化矩阵对应上面的4-7步，A设置为单位矩阵，b设置为0矩阵，参数也设置为0矩阵
    值得注意的是，每个arm都有这么一套矩阵：
    """
    def set_matrix(self, arm):
        for i in arm:
            self.Aa[i] = np.identity(self.d)  # 创建单位矩阵
            self.ba[i] = np.zeros((self.d, 1))
            self.AaI[i] = np.identity(self.d)
            self.theta[i] = np.zeros((self.d, 1))

    """
    计算推荐结果对应于上面的8-11步，我们直接根据公式计算当前的最优参数和置信上界，
    并选择最大的arm作为推荐结果。
    """
    def recommend(self, timestamp, user_features, articles):
        xaT = np.array([user_features])  # d * 1
        xa = np.transpose(xaT)

        # 获取在update阶段已经更新过的AaI(求逆结果)
        AaI_tmp = np.array([self.AaI[article] for article in articles])

        theta_tmp = np.array([self.theta[article] for article in articles])

        art_max = articles[np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]
        # 缓存选择结果，用于update
        self.x = xa
        self.xT = xaT
        # article index with largest UCB
        self.a_max = art_max
        return self.a_max

    """
    对应于上面的12-13步，根据选择的最优arm，以及得到的用户反馈，我们更新A和b矩阵：
    """

    def update(self, reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0

            self.Aa[self.a_max] += np.dot(self.x, self.xT)
            self.ba[self.a_max] += r * self.x
            self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])  # 求逆
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])

        else:
            # error
            pass



