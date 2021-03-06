import os
import glob
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ops import batch_norm, lrelu, conv2d, linear, get_conved_size, deconv2d
from utils import imread, get_image, merge, to_rgb, get_layout

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
                        data_dir='./data',
                        learning_rate=0.0002, beta1=0.5, epoch=25, train_size=np.inf, log_every=5, log_list=[]):
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
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.epoch = epoch
        self.train_size = train_size
        self.log_every = log_every
        self.log_list = log_list


class DCGAN(object):
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
            h1 = tf.nn.relu(self.bn_g1(h1, train=True))
            h2 = deconv2d(h1, [self.cfg.batch_size, o_h2, o_w2, self.cfg.gf_dim*2], scope='g_h2')
            h2 = tf.nn.relu(self.bn_g2(h2, train=True))
            h3 = deconv2d(h2, [self.cfg.batch_size, o_h1, o_w1, self.cfg.gf_dim*1], scope='g_h3')
            h3 = tf.nn.relu(self.bn_g3(h3, train=True))
            h4 = deconv2d(h3, [self.cfg.batch_size, o_h0, o_w0, self.c_dim], scope='g_h4')

            return tf.nn.tanh(h4)

    def sampler(self, z):
        # 与Generator不同在于: 1)设置了变量重用 2)在batch_norm时设置train=False
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            o_h0, o_w0 = self.cfg.output_height, self.cfg.output_width
            o_h1, o_w1 = get_conved_size(o_h0, 2), get_conved_size(o_w0, 2)
            o_h2, o_w2 = get_conved_size(o_h1, 2), get_conved_size(o_w1, 2)
            o_h3, o_w3 = get_conved_size(o_h2, 2), get_conved_size(o_w2, 2)
            o_h4, o_w4 = get_conved_size(o_h3, 2), get_conved_size(o_w3, 2)
            
            z_ = linear(z, self.cfg.gf_dim*8*o_h4*o_w4, scope='g_h0_lin')
            h0 = tf.reshape(z_, [-1, o_h4, o_w4, self.cfg.gf_dim*8])
            h0 = tf.nn.relu(self.bn_g0(h0, train=False))
            h1 = deconv2d(h0, [self.cfg.batch_size, o_h3, o_w3, self.cfg.gf_dim*4], scope='g_h1')
            h1 = tf.nn.relu(self.bn_g1(h1, train=False))
            h2 = deconv2d(h1, [self.cfg.batch_size, o_h2, o_w2, self.cfg.gf_dim*2], scope='g_h2')
            h2 = tf.nn.relu(self.bn_g2(h2, train=False))
            h3 = deconv2d(h2, [self.cfg.batch_size, o_h1, o_w1, self.cfg.gf_dim*1], scope='g_h3')
            h3 = tf.nn.relu(self.bn_g3(h3, train=False))
            h4 = deconv2d(h3, [self.cfg.batch_size, o_h0, o_w0, self.c_dim], scope='g_h4')
            
            return tf.nn.tanh(h4)
    
    def build_model(self):
        if self.cfg.crop:
            image_dims = [self.cfg.output_height, self.cfg.output_width, self.c_dim]
        else:
            image_dims = [self.cfg.input_height, self.cfg.input_width, self.c_dim]
        
        self.inputs = tf.placeholder(tf.float32, [self.cfg.batch_size]+image_dims, name='real_images')
        
        self.z = tf.placeholder(tf.float32, [None, self.cfg.z_dim], name='z')
        
        # 因为self.discriminator会被调用两次, 所以要添加reuse参数
        self.G = self.generator(self.z)
        self.D1, self.D1_logits = self.discriminator(self.inputs, reuse=False)
        #self.sampler = self.sampler(self.z)
        self.D2, self.D2_logits = self.discriminator(self.G, reuse=True)
        
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2)))
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        self.saver = tf.train.Saver()
    
    def train(self, config=None):
        if type(config) == type(None):
            config = self.cfg
        d_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        tf.global_variables_initializer().run()

        start_time = time.time()
        could_load, counter = self.load(self.cfg.checkpoint_dir)
        if could_load:
            print('[*] Load SUCCESS')
        else:
            print('[!] Load FAILED...')
        
        for epoch in range(1, config.epoch+1):
            data = glob.glob(os.path.join(config.data_dir, config.dataset_name, self.cfg.input_fname_pattern))
            np.random.shuffle(data)
            batch_idxs = min(len(data), config.train_size) // config.batch_size
            print('Data loaded: epoch', epoch)
            
            for idx in range(batch_idxs):
                #print('start idx:\t', idx)
                batch_files = data[idx*config.batch_size: (idx+1)*config.batch_size]
                batch = [get_image( batch_file,
                                input_height=self.cfg.input_height,
                                input_width=self.cfg.input_width,
                                resize_height=self.cfg.output_height,
                                resize_width=self.cfg.output_width,
                                crop=self.cfg.crop,) for batch_file in batch_files]
                batch_images = np.array(batch)#.astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.cfg.z_dim])#.astype(np.float32)
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # 更新D
                self.sess.run([d_opt, extra_update_ops], feed_dict={self.inputs: batch_images, self.z: batch_z})
                # 更新G
                self.sess.run([g_opt, extra_update_ops], feed_dict={self.inputs: batch_images, self.z: batch_z})
                # 再来一次
                self.sess.run([g_opt, extra_update_ops], feed_dict={self.inputs: batch_images, self.z: batch_z})
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})
                
                if idx % 10 == 0:
                    print('Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            if epoch % self.cfg.log_every == 0:
                self.save(config.checkpoint_dir, epoch+counter)
            elif epoch+counter in self.cfg.log_list:
                self.save(config.checkpoint_dir, epoch+counter)
        self.save(config.checkpoint_dir, config.epoch+counter)
    
    def preview(self, select=True):
        target = None
        if select:
            target = self.list_ckpt()
        self.load(self.cfg.checkpoint_dir, target)
        _z = np.random.uniform(-1, 1, [self.cfg.batch_size, self.cfg.z_dim])#.astype(np.float32)
        #_z = np.random.uniform(-1, 1, [63, self.cfg.z_dim]).astype(np.float32)
        z = tf.placeholder(tf.float32, [None, self.cfg.z_dim])
        ret = self.sampler(z)
        imgs = self.sess.run(ret, feed_dict={z: _z})
        img = merge(imgs, get_layout(_z.shape[0]))
        img = to_rgb(img)
        print('Shape of img:', img.shape)
        plt.figure(figsize=(16,9))
        plt.imshow(img)
        plt.show()
        '''
        data = glob.glob(os.path.join(self.cfg.data_dir, self.cfg.dataset_name, self.cfg.input_fname_pattern))
        np.random.shuffle(data)
        batch = [get_image( batch_file,
                            input_height=self.cfg.input_height,
                            input_width=self.cfg.input_width,
                            resize_height=self.cfg.output_height,
                            resize_width=self.cfg.output_width,
                            crop=self.cfg.crop,) for batch_file in data[:self.cfg.batch_size]]
        img = merge(np.array(batch), get_layout(_z.shape[0]))
        img = to_rgb(img)
        plt.figure(figsize=(16,9))
        plt.imshow(img)
        plt.show()
        '''

    def list_ckpt(self):
        target_dir = os.path.join(self.cfg.checkpoint_dir, self.model_dir)
        names = []
        for root, dirs, files in os.walk(target_dir, topdown=False):
            for f in files:
                if len(f)>5 and f[-6:]=='.index':
                    names.append(f[:-6])
        if not names:
            return None
        names = sorted(names, key=lambda x: int(x.split('-')[-1]))
        print('Please select a ckpt:')
        for i in range(len(names)):
            print('[{}] {}'.format(i, names[i]))
        select = int(input())
        print('select:\n',names[select])
        return names[select]



    @property
    def model_dir(self):
        return '{}_{}_{}'.format(self.cfg.dataset_name, self.cfg.output_height, self.cfg.output_width)
    
    def save(self, checkpoint_dir, step):
        model_name = 'DCGAN.model'
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
    
    def load(self, checkpoint_dir, target=None):
        import re
        print('[*] Loading checkpoints...')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if target:
            ckpt_name = target
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

            
if __name__ == '__main__':
    training = False
    cfg = Config(   input_height=96, 
                    input_width=96, 
                    output_height=64, 
                    output_width=64,
                    batch_size = 64,
                    dataset_name='faces', 
                    epoch=200, log_every=2, log_list=[1, 5, 10, 20, 50, 100, 150, 200, 300, 500])
    
    with tf.Session() as sess:
        dcgan = DCGAN(sess, cfg)
        if training == True:
            dcgan.train()
        else:
            dcgan.preview()
            '''
            for img in res:
                plt.imshow(img)
                plt.show()
                '''
