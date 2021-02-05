import numpy as np
import matplotlib.pyplot as plt

acc_random = np.loadtxt('acc_random_mnist_cnn_E200_C0.1_iidFalse.txt')
acc_fedcs = np.loadtxt('acc_fedcs_mnist_cnn_E200_C0.1_iidFalse.txt')
acc_ucb = np.loadtxt('acc_ucb_mnist_cnn_E200_C0.1_iidFalse.txt')

# plt.plot(range(len(acc_random)), acc_random)
# plt.plot(range(len(acc_fedcs)), acc_fedcs)
# plt.plot(range(len(acc_ucb)), acc_ucb)
# plt.xlabel('round')
# plt.ylabel('Acc')
# plt.show()

plt.plot(range(100), acc_random[:100], label='random')
plt.plot(range(100), acc_fedcs[:100], label='fedcs')
plt.plot(range(100), acc_ucb[:100], label='ucb')
plt.xlabel('round')
plt.ylabel('Acc')
plt.legend()
plt.show()