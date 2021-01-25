#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


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


def mnist_noniid_modified(dataset, num_users):
    """
    Sample non-i.i.d. client data from mnist dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 10, 6000  # 10类图片，每类6000张
    min_train = 200  # 最少200张
    max_train = 1000  # 最多1000张
    main_label_prop = 0.8  # 80%来自同一张图片，20%均匀来自其他类图片


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


    for i in range(num_users):
        datasize = np.random.randint(min_train, max_train + 1)  # 随机数量
        main_label = np.random.randint(0, 10)  # 0-9随机选一个为主类
        print("user: %d, data_size: %d, main_label: %d" %(i, datasize, main_label))

        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size
        print("user: %d, main_label_size: %d, other_label_size: %d" % (i, main_label_size, other_label_size))

        idx_begin = np.random.randint(0, num_imgs - main_label_size) + main_label * num_imgs
        # print("idx_begin class: %d, end class: %d" %(idxs_labels[1][idx_begin], idxs_labels[1][idx_begin + main_label_size]))
        dict_users[i] = np.concatenate((dict_users[i], idxs[idx_begin:idx_begin+main_label_size]), axis=0)



        print("test")


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


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num_user = 5
    # d = mnist_iid(dataset_train, num)
    # d = mnist_noniid(dataset_train, num)
    d = mnist_noniid_modified(dataset_train, num_user)
    print("test")