import argparse, numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/O16/raw/O16_w_event_keys.npy')
parser.add_argument('--train_path', type=str, default='data/O16/split/train.npy')
parser.add_argument('--test_path', type=str, default='data/O16/split/test.npy')
parser.add_argument('--split', type=float, default=0.8) # Fraction of data for training
args = parser.parse_args()

data = np.load(args.input_path, mmap_mode='r')
print('Loaded ' + str(data.shape[0]) + ' events.')
train, test = train_test_split(data, train_size=args.split)
np.save(args.train_path, train)
np.save(args.test_path, test)
print('Saved ' + str(train.shape[0]) + ' events to training and ' + str(test.shape[0]) + ' events to testing.')