import tensorflow as tf
import math

def get_conved_size(size, stride):
    return int(math.ceil(size / stride))

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or 'Linear'):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(bias_start) )

    if with_w:
        return tf.matmul(input_, w)+b, w, b
    else:
        return tf.matmul(input_, w)+b


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.layers.batch_normalization(x,
                                momentum=self.momentum,
                                epsilon=self.epsilon,
                                scale=True,
                                training=train,
                                name=self.name)

def conv2d(input_, output_dim,
        kernel_height=5, kernel_width=5, 
        stride_vertical=2, stride_horizontal=2,
        stddev=0.02, scope='conv2d'):
    '''
    输入: (输入, 输出维数)
    输出: (加bias的卷积层)
    '''
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [kernel_height, kernel_width, input_.get_shape()[-1], output_dim], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_vertical, stride_horizontal, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
        kernel_height=5, kernel_width=5,
        stride_vertical=2, stride_horizontal=2,
        stddev=0.02, scope='deconv2d', with_w=False):
    with tf.variable_scope(scope) as scope:
        # filter: [height, width, output_cjannels, inchannels]
        w = tf.get_variable('w', [kernel_height, kernel_width, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, stride=[1, stride_vertical, stride_horizontal, 1])
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

        if with_w:
            return deconv, w, bias
        else:
            return deconv

