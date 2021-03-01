import json
import matplotlib.pyplot as plt
import numpy as np


def subplot_fedcs(length, interval):
    f1_path = '../simulation/res/mnist_iid/acc_fedcs_mnist_cnn_E200_C0.1_iid_True.txt'
    f2_path = '../simulation/res/mnist_iid/acc_fedcs_mnist_cnn_E200_C0.1_iid_True.txt'
    f3_path = '../simulation/res/mnist_iid/acc_fedcs_mnist_cnn_E200_C0.1_iid_True.txt'

    # f1
    data = np.loadtxt(f1_path)
    f1_acc = data

    data = np.loadtxt(f2_path)
    f2_acc = data

    data = np.loadtxt(f3_path)
    f3_acc = data

    acc_max = []
    acc_min = []
    acc_mean = []

    for i in range(length):
        if i % interval == 0:
            tmp = np.array([f1_acc[i], f2_acc[i], f3_acc[i]])  # , f4_acc[i], f5_acc[i]])
            acc_max.append(np.max(tmp))
            acc_min.append(np.min(tmp))
            acc_mean.append(np.mean(tmp))

    return acc_min, acc_max, acc_mean, f1_acc

def subplot_fedavg(length, interval):
    f1_path = '../simulation/res/mnist_iid/acc_random_mnist_cnn_E200_C0.1_iid_True.txt'
    f2_path = '../simulation/res/mnist_iid/acc_random_mnist_cnn_E200_C0.1_iid_True.txt'
    f3_path = '../simulation/res/mnist_iid/acc_random_mnist_cnn_E200_C0.1_iid_True.txt'

    # f1
    data = np.loadtxt(f1_path)
    f1_acc = data

    data = np.loadtxt(f2_path)
    f2_acc = data

    data = np.loadtxt(f3_path)
    f3_acc = data

    acc_max = []
    acc_min = []
    acc_mean = []

    for i in range(length):
        if i % interval == 0:
            tmp = np.array([f1_acc[i], f2_acc[i], f3_acc[i]])  # , f4_acc[i], f5_acc[i]])
            acc_max.append(np.max(tmp))
            acc_min.append(np.min(tmp))
            acc_mean.append(np.mean(tmp))

    return acc_min, acc_max, acc_mean, f1_acc


def subplot_mab(length, interval):
    f1_path = '../simulation/res/mnist_iid/acc_ucb_mnist_cnn_E200_C0.1_iid_True.txt'
    f2_path = '../simulation/res/mnist_iid/acc_ucb_mnist_cnn_E200_C0.1_iid_True.txt'
    f3_path = '../simulation/res/mnist_iid/acc_ucb_mnist_cnn_E200_C0.1_iid_True.txt'

    data = np.loadtxt(f1_path)
    f1_acc = data

    data = np.loadtxt(f2_path)
    f2_acc = data

    data = np.loadtxt(f3_path)
    f3_acc = data

    acc_max = []
    acc_min = []
    acc_mean = []

    for i in range(length):
        if i % interval == 0:
            tmp = np.array([f1_acc[i], f2_acc[i], f3_acc[i]])  # , f4_acc[i], f5_acc[i]])
            acc_max.append(np.max(tmp))
            acc_min.append(np.min(tmp))
            acc_mean.append(np.mean(tmp))

    return acc_min, acc_max, acc_mean, f1_acc


def subplot_linucb(length, interval):
    f1_path = '../simulation/res/mnist_iid/acc_linucb_mnist_cnn_E200_C0.1_iid_True.txt'
    f2_path = '../simulation/res/mnist_iid/acc_linucb_mnist_cnn_E200_C0.1_iid_True.txt'
    f3_path = '../simulation/res/mnist_iid/acc_linucb_mnist_cnn_E200_C0.1_iid_True.txt'

    data = np.loadtxt(f1_path)
    f1_acc = data

    data = np.loadtxt(f2_path)
    f2_acc = data

    data = np.loadtxt(f3_path)
    f3_acc = data

    acc_max = []
    acc_min = []
    acc_mean = []

    for i in range(length):
        if i % interval == 0:
            tmp = np.array([f1_acc[i], f2_acc[i], f3_acc[i]])  # , f4_acc[i], f5_acc[i]])
            acc_max.append(np.max(tmp))
            acc_min.append(np.min(tmp))
            acc_mean.append(np.mean(tmp))

    return acc_min, acc_max, acc_mean, f1_acc


def plot_cifar_result(interval=3):
    length = 200
    x = np.arange(0, length, interval)
    fedcs_min, fedcs_max, fedcs_mean, fedcs_acc = subplot_fedcs(length, interval)
    fedavg_min, fedavg_max, fedavg_mean, fedavg_acc = subplot_fedavg(length, interval)
    mab_min, mab_max, mab_mean, mab_acc = subplot_mab(length, interval)
    linucb_min, linucb_max, linucb_mean, linucb_acc = subplot_linucb(length, interval)

    fig, ax = plt.subplots()

    # left, bottom, width, height = 0.13, 0.2, 0.8, 0.7
    # ax = fig.add_axes([left, bottom, width, height])

    fedavg_mean = [i / 100 for i in fedavg_mean]
    linucb_mean = [i / 100 for i in linucb_mean]
    mab_mean = [i / 100 for i in mab_mean]
    fedcs_mean = [i / 100 for i in fedcs_mean]

    # plt.plot(x, fedavg_mean, '.-', linewidth=1.5, color='#1f77b4', label='FedAvg [McMahan, 2016]')
    # # plt.fill_between(x, fedavg_min, fedavg_max, color='#59a869', alpha=0.25)
    # plt.plot(x, fedcs_mean, 'd-', linewidth=1.5, color='#ff7f0e', label='FedCS [Nishio, 2019]')
    # # plt.fill_between(x, fedcs_min, fedcs_max, color='#1f77b4', alpha=0.25)
    # plt.plot(x, mab_mean, '*-', linewidth=1.5, color='#2ca02c', label='MAB-based [Yoshida, 2020]')
    # # plt.fill_between(x, mab_min, mab_max, color='#ff7f0e', alpha=0.25)
    # plt.plot(x, linucb_mean, 'v-', linewidth=1.5, color='#d62728', label='cMAB-based (Proposed)')
    # # plt.fill_between(x, linucb_min, linucb_max, color='#d62728', alpha=0.25)
    plt.plot(x, fedavg_mean, 'd:', markersize=6, linewidth=1.5, color='#1f77b4', label='FedAvg [McMahan, 2016]')
    plt.plot(x, fedcs_mean, 'v-.', markersize=6, linewidth=1.5, color='#ff7f0e', label='FedCS [Nishio, 2019]')
    plt.plot(x, mab_mean, 's--', markersize=6, linewidth=1.5, color='#2ca02c', label='MAB-based [Yoshida, 2020]')
    plt.plot(x, linucb_mean, '^-', markersize=6, linewidth=1.5, color='#d62728', label='cMAB-based (Proposed)')


    # plt.plot(x, fedavg_mean, markersize=6, linewidth=1.5, color='y', label='FedAvg [McMahan, 2016]')
    # plt.plot(x, fedcs_mean, markersize=6, linewidth=1.5, color='b', label='FedCS [Nishio, 2019]')
    # plt.plot(x, mab_mean, markersize=6, linewidth=1.5, color='g', label='MAB-based [Yoshida, 2020]')
    # plt.plot(x, linucb_mean,markersize=6, linewidth=1.5, color='r', label='cMAB-based (Proposed)')

    # niid
    plt.ylim(0.92, 0.993)
    # plt.yticks(np.arange(0.85, 1, 5))
    # plt.rcParams['figure.figsize'] = (6, 4)
    plt.xlabel('FL Rounds', fontdict={'size': 20})
    plt.ylabel('Test Accuracy', fontdict={'size': 20})
    plt.tick_params(labelsize=18)
    leg = ax.legend(fontsize=17)  # , frameon=False)
    leg.set_draggable(True)
    # plt.title('MNIST (i.i.d.)', fontdict={'size': 20})

    # ax1 = fig.add_axes([0.66, 0.58, 0.24, 0.24])  # 不用figure的形式则无须用set
    # x = [150, 160, 170, 180, 190, 200]
    #
    # y1 = fedavg_mean[-6:]
    # plt.tick_params(labelsize=16)
    # ax1.plot(x, y1, color='y', linewidth=1.5, marker='d', markersize=6, linestyle='-')
    # ax1.plot(x, fedcs_mean[-6:], color='b', linewidth=1.5, marker='v', markersize=6, linestyle='-')
    # ax1.plot(x, mab_mean[-6:], color='g', linewidth=1.5, marker='s', markersize=6, linestyle='-')
    # ax1.plot(x, linucb_mean[-6:], color='r', linewidth=1.5, marker='^', markersize=6, linestyle='-')


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_cifar_result(interval=7)
    # data = np.loadtxt('../simulation/res/mnist_iid/acc_fedcs_cifar_cnn_E200_C0.1_iid_True.txt')
    # print(data[0])


