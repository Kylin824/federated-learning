import numpy as np
import matplotlib.pyplot as plt

acc_random = np.loadtxt('res/noniid_prob_0.8/acc_random_mnist_cnn_E200_C0.1_iidFalse.txt')
acc_fedcs = np.loadtxt('res/noniid_prob_0.8/acc_fedcs_mnist_cnn_E200_C0.1_iidFalse.txt')
acc_ucb = np.loadtxt('res/noniid_prob_0.8/acc_ucb_mnist_cnn_E200_C0.1_iidFalse.txt')
acc_linucb = np.loadtxt('res/noniid_prob_0.8/acc_linucb_mnist_cnn_E200_C0.1_iidFalse.txt')

# acc_random = np.loadtxt('res/noniid_prob_0.9/acc_random_mnist_cnn_E50_C0.1_iidFalse.txt')
# acc_fedcs = np.loadtxt('res/noniid_prob_0.9/acc_fedcs_mnist_cnn_E50_C0.1_iidFalse.txt')
# acc_ucb = np.loadtxt('res/noniid_prob_0.9/acc_ucb_mnist_cnn_E50_C0.1_iidFalse.txt')
# acc_linucb = np.loadtxt('res/noniid_prob_0.9/acc_linucb_mnist_cnn_E50_C0.1_iidFalse.txt')

# plt.plot(range(len(acc_random)), acc_random)
# plt.plot(range(len(acc_fedcs)), acc_fedcs)
# plt.plot(range(len(acc_ucb)), acc_ucb)
# plt.xlabel('round')
# plt.ylabel('Acc')
# plt.show()

round = range(50)

plt.title('noniid prob: 0.8')
plt.plot(round, acc_random[:50], label='random')
plt.plot(round, acc_fedcs[:50], label='fedcs')
plt.plot(round, acc_ucb[:50], label='ucb')
plt.plot(round, acc_linucb[:50], label='linucb')
plt.xlabel('round')
plt.ylabel('Acc')
plt.legend()
plt.show()