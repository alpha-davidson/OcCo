import numpy as np

INPUT_PATH = 'O16_w_event_keys.npy'
TRAIN_PATH = ''
TEST_PATH = ''

SPLIT = 0.8 # Fraction of data for training
CHUNK_SIZE = 1000

full_data = np.load(file_path, mmap_mode='r')
full_size = data.shape[0]
print('Loaded ' + str(full_size) + ' point clouds.')

indices = np.arange(num_samples)
np.random.shuffle(indices)

train_size = int(full_size * SPLIT)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

def save_in_chunks(filename, data, indices, chunk_size):
    num_chunks = len(indices) // chunk_size + 1
    with open(filename, 'wb') as f:
        for i in range(num_chunks):
            chunk_indices = indices[i*chunk_size:(i+1)*chunk_size]
            np.save(f, data[chunk_indices])

save_in_chunks(TRAIN_PATH, data, train_indices, CHUNK_SIZE)
save_in_chunks(TEST_PATH, data, test_indices, CHUNK_SIZE)
print(str(train_size) + ' point clouds saved to training and ' + str(full_size - train_size) + ' point clouds saved to testing.')