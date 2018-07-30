import numpy as np
import tensorflow as tf
from ops import *

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
            batch_size=64, sample_num=64, output_height=64, output_width=64,
            z_dim=100, gf_dim=64, df_dim=64, c_dim=3):
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.crop = crop
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        
        self.bn_d1 = batch_norm(name='bn_d1')
        self.bn_d2 = batch_norm(name='bn_d2')
        self.bn_d3 = batch_norm(name='bn_d3')

        self.bn_g0 = batch_norm(name='bn_g0')
        self.bn_g1 = batch_norm(name='bn_g1')
        self.bn_g2 = batch_norm(name='bn_g2')
        self.bn_g3 = batch_norm(name='bn_g3')

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
            o_h, o_w = self.output_height, self.output_width
            o_h1, o_w1 = get_conved_size(o_h, 2), get_conved_size(o_w, 2)
            o_h2, o_w2 = get_conved_size(o_h1, 2), get_conved_size(o_w1, 2)
            o_h3, o_w3 = get_conved_size(o_h2, 2), get_conved_size(o_w2, 2)
            o_h4, o_w4 = get_conved_size(o_h3, 2), get_conved_size(o_w3, 2)
            
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*o_h4*o_w4, scope='g_h0_lin', with_w=True)
            
            self.h0 = tf.reshape(self.z_, [-1, o_h4, o_w4, self.gf_dim*8])
            h0 = tf.nn.relu(self.bn_g0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, o_h3, o_w3, self.gf_dim*4], scope='g_h1', with_w=True)
            h1 = tf.nn.relu(self.bn_g1(self.h1))

            self.h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, o_h2, o_w2, self.gf_dim*2], scope='g_h2', with_w=True)
            h2 = tf.nn.relu(self.bn_g2(self.h2))

            self.h3, self.h3_w, self.h3_b = deconv(h2, [self.batch_size, o_h1, o_w1, self.gf_dim], scope='g_h3', with_w=True)
            h3 = tf.nn.relu(slef.bn_g3(self.h3))

            self.h4, self.h4_w, self.h4_b = deconv(h3, [self.batch_size, o_h, o_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(self.h4)
    
    def sampler(self, z):
        # 与Generator不同在于: 1)设置了变量重用 2)在batch_norm时设置train=False
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()


    
    def build_model(self):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size]+image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram('z', self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.sampler = ???














if __name__ == '__main__':
    pass
