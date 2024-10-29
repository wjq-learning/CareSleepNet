import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
from utils import *
import random
from datasets.dataset import LoadDataset
from trainer import Trainer
from models.model import Model
import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def main():

    parser = argparse.ArgumentParser(description='Greenhouse Image Analysis')
    parser.add_argument('--seed', type=int, default=666, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=7, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 32)')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')

    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--cross_validation', type=int, default=10, help='cross_validation')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
    # parser.add_argument('--loss_function', type=str, default='FocalLoss', help='dropout')
    # parser.add_argument('--seqs_type', type=str, default=seqs_type,
    #                     help='seqs_type')

    # parser.add_argument('--sleep_expand', type=bool, default=True, help='sleep_expand')
    # parser.add_argument('--sleep_expand', type=bool, default=False, help='sleep_expand')

    parser.add_argument('--seqs_dir', type=str, default='.........',
                        help='seqs_dir')
    parser.add_argument('--labels_dir', type=str, default='........',
                        help='labels_dir')


    parser.add_argument('--model_dir',   type=str,   default='/data/wjq/models_weights/0.7', help='model_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)

    torch.cuda.set_device(params.cuda)
    load_dataset = LoadDataset(params)

    evaluation_bests = None
    cm_bests = None
    evaluation_best_list = []
    model_init = Model(params)
    # cm_best_list = []

    for cv_num in range(params.cross_validation):
        print(f"10-fold Cross Validation: {cv_num + 1}")
        model = copy.deepcopy(model_init)
        data_loader = load_dataset.get_data_loader(cv_num)
        trainer = Trainer(params, data_loader, model)
        evaluation_best, cm_best = trainer.train()
        evaluation_best_list.append(evaluation_best)
        # cm_best_list.append(cm_best)

        if evaluation_bests is None:
            evaluation_bests = evaluation_best
        else:
            evaluation_bests += evaluation_best

        if cm_bests is None:
            cm_bests = cm_best
        else:
            cm_bests += cm_best

    for item in evaluation_best_list:
        print(item)

    print("acc, f1, kappa, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1:")
    print(evaluation_bests/10)
    print(cm_bests)



if __name__ == '__main__':
    main()
