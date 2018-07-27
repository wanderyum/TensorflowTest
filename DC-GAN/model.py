import tensorflow as tf
from ops import *

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
            batch_size=64, sample_num=64, output_height=64, output_width=64,
            z_dim=100, gf_dim=64, df_dim=64):
        self.df_dim = df_dim

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            h0 = tf.lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            #h1 = tf.lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
