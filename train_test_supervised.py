# -*- coding: utf-8 -*-


import datetime
from dataloader.ucr2018 import load_ucr2018
from optim.train import supervised_train
import argparse
import torch


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs_test', type=int, default=400,
                        help='number of test epochs')
    parser.add_argument('--patience_test', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')

    # model dataset
    parser.add_argument('--dataset_name', type=str, default='CricketX',
                        choices=['CricketX', 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound', 'DodgerLoopDay',
                                 'MFPT', 'XJTU']
                        )
    parser.add_argument('--ucr_path', type=str, default='./datasets/',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--model_name', type=str, default='SupCE',
                        choices=['SupCE'], help='choose method')

    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    import numpy as np

    opt = parse_option()

    Seeds = [0, 1, 2, 3, 4]
    Runs = range(0, 10, 1)

    model_name='SupCE'

    exp = 'exp-linear-evaluation'

    aug1 = 'magnitude_warp'
    aug2 = 'time_warp'

    model_paras='none'

    results=[]
    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    log_dir = './log/{}/{}/{}/{}/{}'.format(
        exp, opt.dataset_name, opt.model_name, '_'.join(opt.aug_type), model_paras)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file2print = open("{}/test.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print)
    print("Dataset\tAcc_mean\tAcc_std\tEpoch_max",
          file=file2print)
    file2print.flush()

    file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max",
          file=file2print_detail)
    file2print_detail.flush()

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        ACCs_run={}
        MAX_EPOCHs_run = {}
        for run in Runs:

            opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
                exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
                model_paras, str(seed))

            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)

            print('[INFO] Running at:', opt.dataset_name)

            x_train, y_train, x_val, y_val, x_test, y_test, nb_class, _ \
                = load_ucr2018(opt.ucr_path, opt.dataset_name)
            ####
            # Test
            ####
            acc_test, epoch_max_point = supervised_train(
                x_train, y_train, x_val, y_val, x_test, y_test,nb_class,
                opt)

            ACCs_run[run] = acc_test
            MAX_EPOCHs_run[run] = epoch_max_point

        ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
        MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], nb_class,
            seed,ACCs_seed[seed],MAX_EPOCHs_seed[seed]
        ),
              file=file2print_detail)
        file2print_detail.flush()

    ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
    ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
    MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

    print("{}\t{}\t{}\t{}".format(
        opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max),
          file=file2print)
    file2print.flush()





