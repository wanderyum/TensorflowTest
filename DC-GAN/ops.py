import tensorflow as tf

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

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
        stddev=0.02, name='conv2d')
    '''
    输入: (输入, 输出维数)
    输出: (加bias的卷积层)
    '''
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, input_.get_shape()[-1], output_dim], 
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_vertical, stride_horizontal, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

        return conv



