import os
import glob

import numpy as np
import tensorflow as tf

import time
from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
            batch_size=64, sample_num=64, output_height=64, output_width=64,
            z_dim=100, gf_dim=64, df_dim=64, c_dim=3, dataset_name='default', input_fname_pattern='*.jpg', checkpoint_dir=None, data_dir='./data'):
        self.sess = sess
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
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

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
        
        self.data = glob.glob(data_path)    # glob.glob()返回符合要求的文件列表
        np.random.shuffle(self.data)
        imreadImg = imread(self.data[0])
        self.c_dim = imreadImg.shape[-1]

        self.build_model()

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
            
            # 把relu过程分开是因为deconv2d过程需要权重共享
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*o_h4*o_w4, scope='g_h0_lin', with_w=True)
            
            self.h0 = tf.reshape(self.z_, [-1, o_h4, o_w4, self.gf_dim*8])
            h0 = tf.nn.relu(self.bn_g0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, o_h3, o_w3, self.gf_dim*4], scope='g_h1', with_w=True)
            h1 = tf.nn.relu(self.bn_g1(self.h1))

            self.h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, o_h2, o_w2, self.gf_dim*2], scope='g_h2', with_w=True)
            h2 = tf.nn.relu(self.bn_g2(self.h2))

            self.h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, o_h1, o_w1, self.gf_dim], scope='g_h3', with_w=True)
            h3 = tf.nn.relu(self.bn_g3(self.h3))

            self.h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, o_h, o_w, self.c_dim], scope='g_h4', with_w=True)

            return tf.nn.tanh(self.h4)
    
    def sampler(self, z):
        # 与Generator不同在于: 1)设置了变量重用 2)在batch_norm时设置train=False
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            o_h, o_w = self.output_height, self.output_width
            o_h1, o_w1 = get_conved_size(o_h, 2), get_conved_size(o_w, 2)
            o_h2, o_w2 = get_conved_size(o_h1, 2), get_conved_size(o_w1, 2)
            o_h3, o_w3 = get_conved_size(o_h2, 2), get_conved_size(o_w2, 2)
            o_h4, o_w4 = get_conved_size(o_h3, 2), get_conved_size(o_w3, 2)
            
            h0 = tf.reshape(linear(z, self.gf_dim*8*o_h4*o_w4, 'g_h0_lin'), [-1, o_h4, o_w4, self.gf_dim*8])
            h0 = tf.nn.relu(self.bn_g0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, o_h3, o_w3, self.gf_dim*4], scope='g_h1')
            h1 = tf.nn.relu(self.bn_g1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, o_h2, o_w2, self.gf_dim*2], scope='g_h2')
            h2 = tf.nn.relu(self.bn_g2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, o_h1, o_w1, self.gf_dim], scope='g_h3')
            h3 = tf.nn.relu(self.bn_g3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, o_h, o_w, self.c_dim], scope='g_h4')

            return tf.nn.tanh(h4)

    
    def build_model(self):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size]+image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram('z', self.z)
        
        # 因为self.discriminator会被调用两次, 所以要添加reuse参数
        self.G = self.generator(self.z)
        self.D1, self.D1_logits = self.discriminator(self.inputs, reuse=False)
        self.sampler = self.sampler(self.z)
        self.D2, self.D2_logits = self.discriminator(self.G, reuse=True)

        self.d1_sum = tf.summary.histogram('d1', self.D1)
        self.d2_sum = tf.summary.histogram('d2', self.D2)
        self.g_sum = tf.summary.histogram('g', self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2)))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        #
        # 这里是不应该洗牌一下?
        #
        sample_files = self.data[0:self.sample_num]
        sample = [set_image(sample_file, 
                input_height=self.input_height, input_width=self.input_width,
                resize_height=self.output_height, resize_width=self.output_width,
                crop=self.crop) for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed...')

        for epoch in range(config.epoch):
            self.data = glob(os.path.join(config.data_dir, config.dataset, self.input_fname_pattern))
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in range(batch_idxs):
                batch_files = self.data[idx*config.batch_size: (idx+1)*config.batch_size]
                batch = [get_image( batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop,) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                # 更新D
                self.sess.run(d_opt, feed_dict={self.inputs: batch_images, self.z: batch_z})
                # 更新G
                self.sess.run(g_opt, feed_dict={self.z: batch_z})
                # 再来一次
                self.sess.run(g_opt, feed_dict={self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})
                
                counter += 1
                print('Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            self.save(config.checkpoint_dir, epoch)

    @property
    def model_dir(self):
        return '{}_{}_{}'.format(self.dataset_name, self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = 'DCGAN.model'
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print('[*] Reading checkpoints...')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)')).group(0))
            print('[*] Succecess to read {}'.format(ckpt_name))
            return True, counter
        else:
            print('[*] Failed to find a checkpoint')
            return False, 0


class Config(object):
    def __init__(self, learning_rate, beta1, epoch):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.epoch = epoch
        self.train_size = train_size
        self.batch_size = batch_size


if __name__ == '__main__':
    sess = 'fake'
    dcgan = DCGAN(sess)
