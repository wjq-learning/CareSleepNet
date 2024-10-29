import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random



class CustomDataset(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        seq_eeg = np.load(seq_path)[:, 2:8, :]
        seq_eog = np.load(seq_path)[:, :2, :]
        seq = np.concatenate((seq_eeg, seq_eog), axis=1)
        label = np.load(label_path)
        return seq, label

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_seq), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.seqs_dir = params.seqs_dir
        self.labels_dir = params.labels_dir
        self.seqs_labels_path_pair = self.load_path()

    def get_data_loader(self, cv_num):
        train_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair, cv_num)
        train_set = CustomDataset(train_pairs)
        test_set = CustomDataset(test_pairs)
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=1,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader

    def load_path(self):
        seqs_labels_path_pair = []
        subject_nums = os.listdir(self.seqs_dir)
        # print(subject_nums)
        subject_dirs_seq = []
        subject_dirs_labels = []
        for subject_num in subject_nums:
            subject_dirs_seq.append(os.path.join(self.seqs_dir, subject_num, 'data'))
            subject_dirs_labels.append(os.path.join(self.seqs_dir, subject_num, 'label'))
        subject_dirs_seq.sort()
        subject_dirs_labels.sort()

        # print(self.params.sleep_expand)
        # if not self.params.sleep_expand:
        #     subject_dirs_seq = subject_dirs_seq[:20]
        #     subject_dirs_labels = subject_dirs_labels[:20]

        for subject_seq, subject_label in zip(subject_dirs_seq, subject_dirs_labels):
            # print(subject_seq, subject_label)
            subject_pairs = []
            seq_fnames = os.listdir(subject_seq)
            label_fnames = os.listdir(subject_label)
            for seq_fname, label_fname in zip(seq_fnames, label_fnames):
                subject_pairs.append((os.path.join(subject_seq, seq_fname), os.path.join(subject_label, label_fname)))
            seqs_labels_path_pair.append(subject_pairs)
        random.shuffle(seqs_labels_path_pair)

        return seqs_labels_path_pair

    def split_dataset(self, seqs_labels_path_pair, cv_num):
        # length = len(seqs_labels_path_pair)
            # cv_20_fold = [cv_num*4, cv_num*4+1, cv_num*4+2, cv_num*4+3, ]

        cv_k_fold = range(100)[10 * cv_num:100 * (cv_num + 1)]
        train_pairs = []
        test_pairs = []
        for i, subject_pairs in enumerate(seqs_labels_path_pair):
            if i in cv_k_fold:
                # print(i, len(subject_pairs))
                test_pairs.extend(subject_pairs)
            else:
                train_pairs.extend(subject_pairs)
        # print(len(train_pairs), len(test_pairs))
        return train_pairs, test_pairs


    # def split_dataset(self, seqs_labels_path_pair):
    #     split_ratio = 0.95
    #     train_split = int(len(seqs_labels_path_pair) * split_ratio)
    #     train_pairs = seqs_labels_path_pair[:train_split]
    #     test_pairs = seqs_labels_path_pair[train_split:]
    #
    #     return train_pairs, test_pairs

    # def split_dataset(self, seqs_labels_path_pair):
    #     split_ratio = 0.95
    #     train_pairs = []
    #     test_pairs = []
    #     for item in seqs_labels_path_pair:
    #         random_num = random.random()
    #         if random_num <= split_ratio:
    #             train_pairs.append(item)
    #         else:
    #             test_pairs.append(item)
    #
    #     return train_pairs, test_pairs




if __name__ == '__main__':
    import argparse


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Greenhouse Image Analysis')
    parser.add_argument('--seed', type=int, default=2048, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=2, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 32)')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--cross_validation', type=int, default=20, help='cross_validation')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
    # parser.add_argument('--loss_function', type=str, default='FocalLoss', help='dropout')
    # parser.add_argument('--seqs_type', type=str, default=seqs_type,
    #                     help='seqs_type')

    parser.add_argument('--sleep_expand', type=bool, default=True, help='sleep_expand')
    # parser.add_argument('--sleep_expand', type=bool, default=False, help='sleep_expand')

    parser.add_argument('--seqs_dir', type=str, default='/data/datasets/sleep-edf-seq-npy/seq/' ,
                        help='seqs_dir')
    parser.add_argument('--labels_dir', type=str, default='/data/datasets/sleep-edf-seq-npy/labels/', help='labels_dir')

    # parser.add_argument('--seqs_dir', type=str, default='/data/datasets/sleep-edf39-npy/%s/' % seqs_type,
    #                     help='seqs_dir')
    # parser.add_argument('--labels_dir', type=str, default='/data/datasets/sleep-edf39-npy/labels/', help='labels_dir')

    parser.add_argument('--model_dir',   type=str,   default='/data/wjq/models_weights/0.7', help='model_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)

    torch.cuda.set_device(params.cuda)
    load_dataset = LoadDataset(params)

    for cv_num in range(params.cross_validation):
        data_loader = load_dataset.get_data_loader(cv_num)
        print(data_loader)
