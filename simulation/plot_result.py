import numpy as np
import matplotlib.pyplot as plt

# acc_random = np.loadtxt('res/noniid_prob_0.8/acc_random_mnist_cnn_E200_C0.1_iidFalse.txt')
# acc_fedcs = np.loadtxt('res/noniid_prob_0.8/acc_fedcs_mnist_cnn_E200_C0.1_iidFalse.txt')
# acc_ucb = np.loadtxt('res/noniid_prob_0.8/acc_ucb_mnist_cnn_E200_C0.1_iidFalse.txt')
# acc_linucb = np.loadtxt('res/noniid_prob_0.8/acc_linucb_mnist_cnn_E200_C0.1_iidFalse.txt')

acc_random = np.loadtxt('res/noniid_prob_0.9/acc_random_mnist_cnn_E50_C0.1_iidFalse.txt')
acc_fedcs = np.loadtxt('res/noniid_prob_0.9/acc_fedcs_mnist_cnn_E50_C0.1_iidFalse.txt')
acc_ucb = np.loadtxt('res/noniid_prob_0.9/acc_ucb_mnist_cnn_E50_C0.1_iidFalse.txt')
acc_linucb = np.loadtxt('res/noniid_prob_0.9/acc_linucb_mnist_cnn_E50_C0.1_iidFalse.txt')


# acc_random = np.loadtxt('res/cifar_iid_cnn/acc_random_cifar_cnn_E200_C0.1_iidTrue.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid_cnn/acc_fedcs_cifar_cnn_E200_C0.1_iidTrue.txt')
# acc_ucb = np.loadtxt('res/cifar_iid_cnn/acc_ucb_cifar_cnn_E200_C0.1_iidTrue.txt')
# acc_linucb = np.loadtxt('res/cifar_iid_cnn/acc_linucb_cifar_cnn_E200_C0.1_iidTrue.txt')

# plt.plot(range(len(acc_random)), acc_random)
# plt.plot(range(len(acc_fedcs)), acc_fedcs)
# plt.plot(range(len(acc_ucb)), acc_ucb)
# plt.xlabel('round')
# plt.ylabel('Acc')
# plt.show()

len = 50

round = range(len)

plt.title('mnist noniid prob: 0.8')
plt.plot(round, acc_random[:len], label='random')
plt.plot(round, acc_fedcs[:len], label='fedcs')
plt.plot(round, acc_ucb[:len], label='ucb')
plt.plot(round, acc_linucb[:len], label='linucb')
plt.xlabel('round')
plt.ylabel('Acc')
plt.ylim(50, 100)
plt.legend()
plt.show()