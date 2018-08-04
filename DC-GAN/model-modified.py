import os
import glob
import time

import numpy as np
import tensorflow as tf

from ops import batch_norm, lrelu, conv2d, linear, get_conved_size, deconv2d
from utils import imread

class Config(object):
    def __init__(self,  input_height=108, input_width=108, 
                        output_height=64, output_width=64,
                        crop=True,
                        batch_size=64, sample_num=64,
                        z_dim=100, gf_dim=64, df_dim=64,
                        dataset_name='default',
                        input_fname_pattern='*.jpg',
                        checkpoint_dir='./ckpt',
                        sample_dir='./sample',
                        data_dir='./data'):
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir


class DVGAN(object):
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg

        # batch normalization层
        self.bn_d1 = batch_norm(name='bn_d1')
        self.bn_d2 = batch_norm(name='bn_d2')
        self.bn_d3 = batch_norm(name='bn_d3')

        self.bn_g0 = batch_norm(name='bn_g0')
        self.bn_g1 = batch_norm(name='bn_g1')
        self.bn_g2 = batch_norm(name='bn_g2')
        self.bn_g3 = batch_norm(name='bn_g3')
    
        data_path = os.path.join(self.cfg.data_dir, self.cfg.dataset_name, self.cfg.input_fname_pattern)
        self.data = glob.glob(data_path)
        self.c_dim = imread(self.data[0]).shape[-1]

        self.build_model()

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.cfg.df_dim, scope='d_h0_conv'))
            h1 = lrelu(self.bn_d1(conv2d(h0, self.cfg.df_dim*2, scope='d_h1_conv'), train=True))
            h2 = lrelu(self.bn_d2(conv2d(h1, self.cfg.df_dim*4, scope='d_h2_conv'), train=True))
            h3 = lrelu(self.bn_d3(conv2d(h2, self.cfg.df_dim*8, scope='d_h3_conv'), train=True))
            h4 = linear(tf.reshape(h3, [self.cfg.batch_size, -1]), 1, scope='d_h4_linear')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope('generator') as scope:
            # 从输出大小推各步尺寸
            o_h0, o_w0 = self.cfg.output_height, self.cfg.output_width
            o_h1, o_w1 = get_conved_size(o_h0, 2), get_conved_size(o_w0, 2)
            o_h2, o_w2 = get_conved_size(o_h1, 2), get_conved_size(o_w1, 2)
            o_h3, o_w3 = get_conved_size(o_h2, 2), get_conved_size(o_w2, 2)
            o_h4, o_w4 = get_conved_size(o_h3, 2), get_conved_size(o_w3, 2)

            # 把relu过程分开是因为deconv2d过程需要权重共享
            z_ = linear(z, self.cfg.gf_dim*8*o_h4*o_w4, scope='g_h0_lin')
            h0 = tf.reshape(z_, [-1, o_h4, o_w4, self.cfg.gf_dim*8])
            h0 = tf.nn.relu(self.bn_g0(h0, train=True))
            h1 = deconv2d(h0, [self.cfg.batch_size, o_h3, o_w3, self.cfg.gf_dim*4], scope='g_h1')
            h1 = tf.nn.relu(self.bn_g1(h1))
            h2 = deconv2d(h1, [self.cfg.batch_size, o_h2, o_w2, self.cfg.gf_dim*2], scope='g_h2')
            h2 = tf.nn.relu(self.bn_g2(h2))
            h3 = deconv2d(h2, [self.cfg.batch_size, o_h1, o_w1, self.cfg.gf_dim*1], scope='g_h3')
            h3 = tf.nn.relu(self.bn_g3(h3))
            h4 = deconv2d(h3, [self.cfg.batch_size, o_h0, o_w0, self.c_dim], scope='g_h4')

            return tf.nn.tanh(h4)

    def sampler(self, z):

