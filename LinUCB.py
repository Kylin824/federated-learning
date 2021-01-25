import numpy as np

# ref: https://zhuanlan.zhihu.com/p/38875273

class LinUCB:

    def __init__(self):
        self.alpha = 0.25  # ̽������
        self.r1 = 0.6  # ����������ֵ if worse -> 0.7, 0.8
        self.r0 = -16  # ����������ֵ if worse, -19, -21

        self.d = 6  # ����ά��
        self.Aa = {}  # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}  # AaI : store the inverse of all Aa matrix

        self.ba = {}  # ba : collection of vectors to compute disjoin part, d*1
        self.a_max = 0
        self.theta = {}
        self.x = None
        self.xT = None

    """
    ��ʼ�������Ӧ�����4-7����A����Ϊ��λ����b����Ϊ0���󣬲���Ҳ����Ϊ0����
    ֵ��ע����ǣ�ÿ��arm������ôһ�׾���
    """
    def set_matrix(self, arm):
        for i in arm:
            self.Aa[i] = np.identity(self.d)  # ������λ����
            self.ba[i] = np.zeros((self.d, 1))
            self.AaI[i] = np.identity(self.d)
            self.theta[i] = np.zeros((self.d, 1))

    """
    �����Ƽ������Ӧ�������8-11��������ֱ�Ӹ��ݹ�ʽ���㵱ǰ�����Ų����������Ͻ磬
    ��ѡ������arm��Ϊ�Ƽ������
    """
    def recommend(self, timestamp, user_features, articles):
        xaT = np.array([user_features])  # d * 1
        xa = np.transpose(xaT)

        # ��ȡ��update�׶��Ѿ����¹���AaI(������)
        AaI_tmp = np.array([self.AaI[article] for article in articles])

        theta_tmp = np.array([self.theta[article] for article in articles])

        art_max = articles[np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]
        # ����ѡ����������update
        self.x = xa
        self.xT = xaT
        # article index with largest UCB
        self.a_max = art_max
        return self.a_max

    """
    ��Ӧ�������12-13��������ѡ�������arm���Լ��õ����û����������Ǹ���A��b����
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
            self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])  # ����
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])

        else:
            # error
            pass



