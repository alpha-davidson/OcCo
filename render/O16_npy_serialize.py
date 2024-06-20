# This script is a modified version of OcCo_Torch/utils/LMDB_Writer.py

import os, argparse, numpy as np
from tensorpack import DataFlow, dataflow

class pcd_df(DataFlow):
    def __init__(self, complete_dir, partial_dir):
        self.complete_clouds = np.load(complete_dir, mmap_mode='r')
        self.num_examples = self.complete_clouds.shape[0]
        self.partial_clouds = np.load(partial_dir, mmap_mode='r')
        self.num_snapshots = self.partial_clouds.shape[1]

    def size(self):
        return self.num_examples * self.num_snapshots

    def get_data(self):
        for i in range(self.num_examples):
            complete = self.complete_clouds[i]
            for j in range(self.num_snapshots):
                partial = self.partial_clouds[i][j]
                yield ('example_' + str(i)), partial, complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete_dir', default=r'../data/ModelNet40')
    parser.add_argument('--partial_dir', default=r'../render/dump_modelnet_normalised_supercoarse/pcd')
    parser.add_argument('--output_file', default=r'../data/ModelNet40_train_1024_supercoarse.lmdb')
    args = parser.parse_args()

    df = pcd_df(args.complete_dir, args.partial_dir)
    if os.path.exists(args.output_file):
        os.system('rm %s' % args.output_file)
    dataflow.LMDBSerializer.save(df, args.output_file)