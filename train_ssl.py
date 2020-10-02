# -*- coding: utf-8 -*-

import datetime
from optim.pretrain import *
import argparse
import torch
from utils.utils import get_config_from_json


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--K', type=int, default=16, help='Number of augmentation for each sample') # Bigger is better.

    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=400,
                        help='training patience')
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')
    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')
    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='CricketX',
                        choices=['CricketX', 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound','DodgerLoopDay',
                                 'MFPT','XJTU'],
                        help='dataset')
    parser.add_argument('--ucr_path', type=str, default='./datasets/',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4')
    parser.add_argument('--model_name', type=str, default='SelfTime',
                        choices=['InterSample', 'IntraTemporal', 'SelfTime'], help='choose method')
    parser.add_argument('--config_dir', type=str, default='./config', help='The Configuration Dir')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    import numpy as np

    opt = parse_option()

    exp = 'linear_eval'

    Seeds = [0, 1, 2, 3, 4]

    aug1 = ['magnitude_warp']
    aug2 = ['time_warp']

    config_dict = get_config_from_json('{}/{}_config.json'.format(
        opt.config_dir, opt.dataset_name))

    opt.class_type = config_dict['class_type']
    opt.piece_size = config_dict['piece_size']

    if opt.model_name=='InterSample':
        model_paras = 'none'
    else:
        model_paras = '{}_{}'.format(opt.piece_size, opt.class_type)

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

    file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail_train)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max", file=file2print_detail_train)
    file2print_detail_train.flush()

    ACCs = {}

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
            exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
            model_paras, str(seed))

        if not os.path.exists(opt.ckpt_dir):
            os.makedirs(opt.ckpt_dir)

        print('[INFO] Running at:', opt.dataset_name)

        x_train, y_train, x_val, y_val, x_test, y_test, nb_class, _ \
            = load_ucr2018(opt.ucr_path, opt.dataset_name)

        ################
        ## Train #######
        ################
        if opt.model_name == 'InterSample':
            acc_max, epoch_max = pretrain_InterSampleRel(x_train, y_train, opt)
        elif 'IntraTemporal' in opt.model_name:
            acc_max, epoch_max = pretrain_IntraSampleRel(x_train, y_train, opt)
        elif 'SelfTime' in opt.model_name:
            acc_max, epoch_max = pretrain_SelfTime(x_train, y_train, opt)

        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], nb_class,
            seed, round(acc_max, 2), epoch_max),
            file=file2print_detail_train)
        file2print_detail_train.flush()





