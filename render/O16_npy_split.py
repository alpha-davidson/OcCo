import numpy as np
from sklearn.model_selection import train_test_split

INPUT_PATH = 'OcCo_TF/data/O16/train.npy'
TRAIN_PATH = 'OcCo_TF/data/O16/actual_train.npy'
TEST_PATH = 'OcCo_TF/data/O16/validation.npy'

SPLIT = 0.8 # Fraction of data for training

data = np.load(INPUT_PATH, mmap_mode='r')
print('Loaded ' + str(data.shape[0]) + ' events.')
train, test = train_test_split(data, train_size=SPLIT)
np.save(TRAIN_PATH, train)
np.save(TEST_PATH, test)
print('Saved ' + str(train.shape[0]) + ' events to training and ' + str(test.shape[0]) + ' events to testing.')