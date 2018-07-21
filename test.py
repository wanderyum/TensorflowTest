# https://github.com/AYLIEN/gan-intro/blob/master/gan.py

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

argDict = { 'batch_size': 8,
            'hidden_size': 4,
            'minibatch': True,
            'anim_path': None,
            'num_steps': 5000,
            'log_every': 500,
            'anim_every': 1
            }

class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5
    
    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        #samples.sort()
        return samples

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim*2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim*2, 'd1'))
    
    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim*2, scope='d2'))
    
    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3

def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels*kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=step, var_list=var_list)
    
    return optimizer
    
def log(x):
    '''
    Sometimes discriminator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))

class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params['batch_size'], 1))
            self.G = generator(self.z, params['hidden_size'])
            
        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params['batch_size'], 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(self.x, params['hidden_size'], params['minibatch'])
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G, params['hidden_size'], params['minibatch'])
        
        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1-self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))
        
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)
        
def train(model, data, gen, params):
    anim_frames = []
    
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        
        for step in range(params['num_steps'] + 1):
            # update discriminator
            x = data.sample(params['batch_size'])
            z = gen.sample(params['batch_size'])
            loss_d, _ = sess.run([model.loss_d, model.opt_d],{
                        model.x: np.reshape(x, (params['batch_size'], 1)),
                        model.z: np.reshape(z, (params['batch_size'], 1))
                        })
            
            # update generator
            z = gen.sample(params['batch_size'])
            loss_g, _ = sess.run([model.loss_g, model.opt_g],{
                        model.z: np.reshape(z, (params['batch_size'], 1))
                        })
            if step % params['log_every'] == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
                samps = samples(model, sess, data, gen.range, params['batch_size'])
                plot_distributions(samps, gen.range)
            if params['anim_path'] and (step % params['anim_every'] == 0):
                anim_frames.append(samples(model, sess, data, gen.range, params['batch_size']))

        if params['anim_path'] and (step % params['anim_every'] == 0):
            anim_frames.append(samples(model, sess, data, gen.range, params['batch_size']))
        else:
            samps = samples(model, sess, data, gen.range, params['batch_size'])
            plot_distributions(samps, gen.range)
            
def samples(model, sess, data, sample_range, batch_size, num_points=10000, num_bins=100):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)
    
    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size*i:batch_size*(i+1)] = sess.run(model.D1,{
                                            model.x: np.reshape(xs[batch_size*i:batch_size*(i+1)], (batch_size, 1))})
    
    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)
    
    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size*i: batch_size*(i+1)] = sess.run(
                                        model.G,
                                        {model.z: np.reshape(zs[batch_size*i: batch_size*(i+1)], (batch_size,1))})
    pg, _ = np.histogram(g, bins=bins, density=True)
    
    return db, pd, pg

def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()



def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args)

if __name__ == '__main__':
    main(argDict)
            
            
