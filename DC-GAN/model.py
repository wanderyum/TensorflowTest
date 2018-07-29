import numpy as np
import tensorflow as tf
from ops import *

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
            batch_size=64, sample_num=64, output_height=64, output_width=64,
            z_dim=100, gf_dim=64, df_dim=64):
        self.batch_size = batch_size
        self.df_dim = df_dim
        
        self.bn_d1 = batch_norm(name='bn_d1')
        self.bn_d2 = batch_norm(name='bn_d2')
        self.bn_d3 = batch_norm(name='bn_d3')

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, scope='d_h0_conv'))
            h1 = lrelu(self.bn_d1(conv2d(h0, self.df_dim*2, scope='d_h1_conv'), train=True))
            h2 = lrelu(self.bn_d2(conv2d(h1, self.df_dim*4, scope='d_h2_conv'), train=True))
            h3 = lrelu(self.bn_d3(conv2d(h2, self.df_dim*8, scope='d_h3_conv'), train=True))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, scope='d_h4_linear')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope('generator') as scope:












if __name__ == '__main__':
    sess = None
    img = np.array([[1,2,3],[4,5,6]])
    with tf.Session() as sess:
        dcgan = DCGAN(sess)
        img = tf.placeholder(tf.float32, (64, 96, 96, 1))
        dcgan.discriminator(img)
