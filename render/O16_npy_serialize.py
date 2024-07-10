# This script is a modified version of OcCo_Torch/utils/LMDB_Writer.py

import os, argparse, numpy as np
from tensorpack import DataFlow, dataflow

class pcd_df(DataFlow):
    def __init__(self, complete_dir, occluded_dir):
        self.complete_clouds = np.load(complete_dir, mmap_mode='r')
        self.num_examples = self.complete_clouds.shape[0]
        self.occluded_clouds = np.load(occluded_dir, mmap_mode='r')
        self.num_snapshots = self.occluded_clouds.shape[1]

    def size(self):
        return self.num_examples * self.num_snapshots

    def get_data(self):
        for j in range(self.num_snapshots):
            for i in range(self.num_examples):
                complete = self.complete_clouds[i]
                occluded = self.occluded_clouds[i][j]
                yield ('example_' + str(i)), occluded, complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete_dir', default=r'../data/O16/complete.npy')
    parser.add_argument('--occluded_dir', default=r'../data/O16/occluded.npy')
    parser.add_argument('--output_file', default=r'../data/O16/train.lmdb')
    args = parser.parse_args()

    df = pcd_df(args.complete_dir, args.occluded_dir)
    if os.path.exists(args.output_file):
        os.system('rm %s' % args.output_file)
    dataflow.LMDBSerializer.save(df, args.output_file)