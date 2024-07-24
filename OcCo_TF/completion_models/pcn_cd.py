#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py

import pdb, tensorflow as tf
from utils.tf_util import mlp, mlp_conv, point_maxpool, point_unpool, chamfer, \
    add_train_summary, add_valid_summary

class Model:
    def __init__(self, inputs, npts, gt, **kwargs):
        self.__dict__.update(kwargs)  # batch_decay and is_training
        self.num_coarse = 512
        self.features = self.create_encoder(inputs, npts)
        self.coarse = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, gt)
        self.outputs = self.coarse
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, gt]
        self.visualize_titles = ['input', 'output', 'ground truth']

    def create_encoder(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])
        return coarse

    def create_loss(self, coarse, gt):
        loss = chamfer(coarse, gt)
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_loss]
