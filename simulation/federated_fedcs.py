import matplotlib
matplotlib.use('Agg')  # 绘图不显示
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time

from utils.sampling import mnist_iid, cifar_iid, mnist_noniid_modified
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':

    valid_list = np.loadtxt('valid_list_fedcs.txt')

    # load args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print("cuda is available : ", torch.cuda.is_available())

    # load dataset
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users (100)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid_modified(dataset_train, args.num_users, main_label_prop=0.8)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        global_net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        global_net = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        global_net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')


    print(global_net)

    global_net.train()

    # start time
    # time_start = time.time()

    # copy weights
    w_glob = global_net.state_dict()

    loss_avg_client = []
    acc_global_model = []

    # training
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    last_loss_avg = 0
    last_acc_global = 0

    for round in range(args.epochs):
        loss_locals = []

        if not args.all_clients:
            w_locals = []

        round_idx = valid_list[round]

        user_idx_this_round = round_idx[np.where(round_idx != -1)]

        if len(user_idx_this_round) > 0:

            for idx in user_idx_this_round:

                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

                weight, loss = local.train(net=copy.deepcopy(global_net).to(args.device))

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(weight)
                else:
                    w_locals.append(copy.deepcopy(weight))

                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            global_net.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(round, loss_avg))
            loss_avg_client.append(loss_avg)

            last_loss_avg = loss_avg

            acc_test, loss_test = test_img(global_net, dataset_test, args)

            acc_global_model.append(acc_test)

            last_acc_global = acc_test
        else:

            print('Round {:3d}, Average loss {:.3f}, null client'.format(round, last_loss_avg))
            loss_avg_client.append(last_loss_avg)
            acc_global_model.append(last_acc_global)

    # time_end = time.time()
    # print('totally cost time: {:3f}s'.format(time_end - time_start))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_avg_client)), loss_avg_client)
    plt.ylabel('train_loss')
    plt.savefig('loss_fedcs_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    plt.figure()
    plt.plot(range(len(acc_global_model)), acc_global_model)
    plt.ylabel('acc_global')
    plt.savefig('acc_fedcs_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    np.savetxt('loss_fedcs_{}_{}_E{}_C{}_iid{}.txt'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), loss_avg_client)
    np.savetxt('acc_fedcs_{}_{}_E{}_C{}_iid{}.txt'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), acc_global_model)

    # testing
    global_net.eval()
    acc_train, loss_avg_client = test_img(global_net, dataset_train, args)
    acc_test, loss_test = test_img(global_net, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

