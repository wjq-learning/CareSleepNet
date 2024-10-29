import numpy as np
from tqdm import tqdm

a = np.load('/data/datasets/sleep-edf-npy/all_seqs.npy')

b = np.transpose(a, (2, 0, 1, 3))


#%%
c = b.reshape(7, -1)
c = (c - c.min(axis=-1, keepdims=True)) / (c.max(axis=-1, keepdims=True) - c.min(axis=-1, keepdims=True))
#%%

d = c.reshape(7, 9317, 20, 3000)


e = np.transpose(d, (1, 2, 0, 3))


i = 0
for seq in tqdm(e):
    seq_fname = r'/data/datasets/sleep-edf-npy/min_max_norm/min_max_norm' + str(i) + '.npy'
    with open(seq_fname, 'wb') as f:
        np.save(f, seq)
    i += 1
