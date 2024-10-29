import numpy as np
from tqdm import tqdm

all_seqs = np.load('/data/datasets/sleep-edf39-npy/all_seqs.npy')
all_labels = np.load('/data/datasets/sleep-edf39-npy/all_labels.npy')

print(all_seqs.shape, all_labels.shape)
#
i = 0
for seq in tqdm(all_seqs):
    seq_fname = r'/data/datasets/sleep-edf39-npy/seq/seq' + str(i) + '.npy'
    with open(seq_fname, 'wb') as f:
        np.save(f, seq)
    i += 1

j = 0
for label in tqdm(all_labels):
    label_fname = r'/data/datasets/sleep-edf39-npy/labels/label' + str(j) + '.npy'
    with open(label_fname, 'wb') as f:
        np.save(f, label)
    j += 1