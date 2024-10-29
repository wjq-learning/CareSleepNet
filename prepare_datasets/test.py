import numpy as np


a = np.load('/data/datasets/sleep-edf-npy/822seq.npy')
b = np.load('/data/datasets/sleep-edf-npy/822label.npy')

# print(a[4, 1:3, 2855:2859])
# print(b[4, 1:3, 2855:2859])
print(a.shape, b.shape)

