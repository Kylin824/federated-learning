import numpy as np
import matplotlib.pyplot as plt

# acc_random = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_random_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_fedcs_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_ucb_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_linucb_mnist_cnn_E200_C0.1_iid_False.txt')

# acc_random = np.loadtxt('res/cifar_iid/acc_random_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid/acc_fedcs_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/cifar_iid/acc_ucb_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/cifar_iid/acc_linucb_cifar_cnn_E200_C0.1_iid_True.txt')


acc_random = np.loadtxt('res/cifar_noniid/acc_random_cifar_cnn_E200_C0.1_iid_False.txt')
acc_fedcs = np.loadtxt('res/cifar_noniid/acc_fedcs_cifar_cnn_E200_C0.1_iid_False.txt')
acc_ucb = np.loadtxt('res/cifar_noniid/acc_ucb_cifar_cnn_E200_C0.1_iid_False.txt')
acc_linucb = np.loadtxt('res/cifar_noniid/acc_linucb_cifar_cnn_E200_C0.1_iid_False.txt')


# acc_random = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_random_mnist_cnn_E100_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_fedcs_mnist_cnn_E100_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_ucb_mnist_cnn_E100_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/mnist_noniid_main0.8_other9/acc_linucb_mnist_cnn_E100_C0.1_iid_False.txt')

# acc_random = np.loadtxt('res/mnist_iid/acc_random_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/mnist_iid/acc_fedcs_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/mnist_iid/acc_ucb_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/mnist_iid/acc_linucb_mnist_cnn_E200_C0.1_iid_True.txt')

# plt.plot(range(len(acc_random)), acc_random)
# plt.plot(range(len(acc_fedcs)), acc_fedcs)
# plt.plot(range(len(acc_ucb)), acc_ucb)
# plt.xlabel('round')
# plt.ylabel('Acc')
# plt.show()

len = 200

round = range(len)

plt.title('mnist noniid prob: 0.8')
plt.plot(round, acc_random[:len], label='random')
plt.plot(round, acc_fedcs[:len], label='fedcs')
plt.plot(round, acc_ucb[:len], label='ucb')
plt.plot(round, acc_linucb[:len], label='linucb')
plt.xlabel('round')
plt.ylabel('Acc')
# plt.ylim(0, 60)
plt.legend()
plt.show()