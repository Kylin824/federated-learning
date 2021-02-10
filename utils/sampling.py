import numpy as np
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import torch

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300  # 200份 每份300张图片
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # idx:   [--------------------]    idx代表图片在原始数据集中的索引
    # label: [0, 0, 0, ... 9, 9, 9]    label代表图片对应的数字标签

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def mnist_noniid_modified(dataset, num_users, min_train = 200, max_train = 1000, main_label_prop = 0.8, other = 5):
    """
    non-i.i.d数据生成

    100个client

    数量分布随机从min_train到max_train

    每个client 80%数据为一类图片， 20%为其他类图片

    """

    np.random.seed(0)

    num_shards, num_imgs = 10, 6000  # 10类图片，每类6000张
    # min_train = 200  # 最少200张
    # max_train = 1000  # 最多1000张
    # main_label_prop = 0.8  # 80%来自同一张图片，20%均匀来自其他类图片

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # idxs:           [--------------------]    idx代表图片在原始数据集中的索引
    # idxs_labels[1]: [0, 0, 0, ... 9, 9, 9]    label代表图片对应的数字标签

    for i in range(num_users):
        datasize = np.random.randint(min_train, max_train + 1)  # 随机数量
        main_label = np.random.randint(0, 10)  # 0-9随机选一个为主类
        print("user: %d, data_size: %d, main_label: %d" %(i, datasize, main_label))

        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size
        # print("user: %d, main_label_size: %d, other_label_size: %d" % (i, main_label_size, other_label_size))

        # main label idx array
        idx_begin = np.random.randint(0, num_imgs - main_label_size) + main_label * num_imgs
        # print("idx_begin: %d, begin class: %d, end class: %d" %(idx_begin, idxs_labels[1][idx_begin], idxs_labels[1][idx_begin + main_label_size]))
        dict_users[i] = np.concatenate((dict_users[i], idxs[idx_begin : idx_begin+main_label_size]), axis=0)

        # other label idx array
        other_label_dict = np.zeros(other_label_size, dtype='int64')

        other_nine_label = np.delete(np.arange(10), main_label)


        other_label_class = np.random.choice(other_nine_label, size=other, replace=False)

        count = 0

        for j in range(other_label_size):
            label = other_label_class[count % other]
            other_label_dict[j] = idxs[int(np.random.randint(0, num_imgs) + label * num_imgs)]
            count += 1

        dict_users[i] = np.concatenate((dict_users[i], other_label_dict), axis=0)

        # for k in range(datasize):
        #     idx = dict_users[i][k]
        #     print("idx: %d, label: %d" %(dict_users[i][k], labels[idx]))
        #
        # print("++++++++++++++++++++++++++++++++++++++")

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, min_train=200, max_train=1000, main_label_prop=0.8, other=9):
    """
    non-i.i.d数据生成

    100个client

    数量分布随机从min_train到max_train

    每个client 80%数据为一类图片， 20%为其他类图片

    """

    np.random.seed(0)

    num_shards, num_imgs = 10, 5000  # 10类图片，每类6000张

    # min_train = 200  # 最少200张
    # max_train = 1000  # 最多1000张
    # main_label_prop = 0.8  # 80%来自同一张图片，20%均匀来自其他9类图片

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)


    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # idxs:           [--------------------]    idx代表图片在原始数据集中的索引
    # idxs_labels[1]: [0, 0, 0, ... 9, 9, 9]    label代表图片对应的数字标签

    for i in range(num_users):
        datasize = np.random.randint(min_train, max_train + 1)  # 随机数量
        main_label = np.random.randint(0, 10)  # 0-9随机选一个为主类
        print("user: %d, data_size: %d, main_label: %d" %(i, datasize, main_label))

        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size
        # print("user: %d, main_label_size: %d, other_label_size: %d" % (i, main_label_size, other_label_size))

        # main label idx array
        idx_begin = np.random.randint(0, num_imgs - main_label_size) + main_label * num_imgs
        # print("idx_begin: %d, begin class: %d, end class: %d" %(idx_begin, idxs_labels[1][idx_begin], idxs_labels[1][idx_begin + main_label_size]))
        dict_users[i] = np.concatenate((dict_users[i], idxs[idx_begin : idx_begin+main_label_size]), axis=0)

        # other label idx array
        other_label_dict = np.zeros(other_label_size, dtype='int64')

        other_nine_label = np.delete(np.arange(10), main_label)


        other_label_class = np.random.choice(other_nine_label, size=other, replace=False)

        count = 0

        for j in range(other_label_size):
            label = other_label_class[count % other]
            other_label_dict[j] = idxs[int(np.random.randint(0, num_imgs) + label * num_imgs)]
            count += 1

        dict_users[i] = np.concatenate((dict_users[i], other_label_dict), axis=0)

    return dict_users


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  #transpose()作用是调换数组行列值的索引值


if __name__ == '__main__':

    num_user = 100

    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,), (0.3081,))
    #                                ]))
    #
    # dict_users = mnist_noniid_modified(dataset_train, num_user)

    # np.save('../simulation/dataset_noniid_200_1000.npy', dict)
    # cs = np.load('../simulative_client_state.npy')
    # print(cs[0])


    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)

    # 类别标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # dict_users = cifar_iid(dataset_train, num_user)

    dict_users = cifar_noniid(dataset_train, num_user)

    print(len(dict_users))

    # # 获取随机数据
    # dataiter = iter(trainloader)  # 把trainloader变成一个可迭代的对象
    # images, labels = dataiter.next()
    #
    # # 展示图像
    # imshow(torchvision.utils.make_grid(images))
    # # 显示图像标签
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #
    # plt.show()




