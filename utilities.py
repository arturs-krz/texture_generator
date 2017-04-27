import numpy as np
import tensorflow as tf

def leaky_relu(input, alpha):
    return tf.maximum(tf.multiply(input, alpha), input)

# shape = [size, size, in_channels, out_channels]
def weight_var(shape, glorot=True):

    if glorot:
        minval = -np.sqrt(6. / (shape[2] + shape[3]))
        maxval = np.sqrt(6. / (shape[2] + shape[3]))
        initial = tf.random_uniform(shape, minval=minval, maxval=maxval)
    else:
        # atgriež random vērtības no normālsadalījuma max 2 standartnoviržu attālumā
        initial = tf.truncated_normal(shape, stddev=0.2, mean=0.5)
    
    return tf.Variable(initial)


def bias_var(num_filters, input_channels, glorot=True):

    if glorot:
        minval = -np.sqrt(6. / (num_filters + input_channels))
        maxval = np.sqrt(6. / (num_filters + input_channels))
        initial = tf.random_uniform([num_filters], minval=minval, maxval=maxval)
    else:
        initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv(input, num_filters, filter_size, stride_len, activation='leaky_relu', name='conv'):
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        weights = weight_var(shape=[filter_size, filter_size, input_shape[3], num_filters])
        bias = bias_var(num_filters, input_shape[3], glorot=True)
        tf.summary.histogram('conv_weights', weights)

        output = tf.nn.conv2d(
            input=input, 
            filter=weights, 
            strides=[1, stride_len, stride_len, 1],
            padding="SAME"
        )
        # output = instance_norm(output)
        output = spatial_batch_norm(tf.nn.bias_add(output, bias))
        
        tf.summary.histogram('conv_output', output)

        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'leaky_relu':
            output = leaky_relu(output, alpha=0.01)
        
        return output

def residual_conv(input, filter_size, name='residual'):
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        residual_layer = conv(input, input_shape[3], filter_size, 1, name=name+'_residual')
        return input + conv(residual_layer, input_shape[3], filter_size, 1, activation='relu')

def conv_transpose(input, num_filters, filter_size, stride_len, name='conv_transpose'):
    """
    Output shape = [batch, x*stride, y*stride, num_filters]
    """
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        weights = weight_var(shape=[filter_size, filter_size, num_filters, input_shape[3]])
        tf.summary.histogram('transpose_weights', weights)

        shape = tf.stack([input_shape[0], input_shape[1] * stride_len, input_shape[2] * stride_len, num_filters])
        output = tf.nn.conv2d_transpose(input, weights, shape, [1, stride_len, stride_len, 1], padding='SAME')
        # output = instance_norm(output)

        tf.summary.histogram('transpose_output', output)

        return tf.nn.relu(output)

def instance_norm(input):
    input_shape = input.get_shape().as_list()
    var_shape = [input_shape[3]]
    mu, sigma_sq = tf.nn.moments(input, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (input - mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift    

def gram_matrix(activation_layer):
    layer_shape = activation_layer.get_shape().as_list()
    
    # N filters / feature maps
    N = layer_shape[3]
    # M = x * y
    M = layer_shape[1] * layer_shape[2]

    F = tf.reshape(activation_layer, shape=[N, -1])
    FT = tf.transpose(F)
    G = tf.matmul(F,FT) / M
    return G


def gram_loss(target_gram, generated, layer_weight=1.0):
    layer_shape = generated.get_shape().as_list()
    # N filters / feature maps
    N = layer_shape[3]
    # M = x * y
    M = layer_shape[1] * layer_shape[2]

    F = tf.reshape(generated, shape=[N, -1])
    FT = tf.transpose(F)
    G = tf.matmul(F,FT) / M

    gram_diff = G - target_gram
    loss = layer_weight/4. * tf.reduce_sum(tf.pow(gram_diff,2)) / (N**2)
    # gradient = tf.reshape(layer_weight * tf.transpose(tf.matmul(FT, gram_diff)) / (M * N**2), shape=layer_shape)
    # return [loss, gradient]
    return loss

def spatial_batch_norm(input_layer, name='spatial_batch_norm'):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
    variance_epsilon = 0.01
    inv = tf.rsqrt(variance + variance_epsilon)
    num_channels = input_layer.get_shape().as_list()[3]
    scale = tf.Variable(tf.random_uniform([num_channels]))
    offset = tf.Variable(tf.random_uniform([num_channels]))
    return_val = tf.subtract(tf.multiply(tf.multiply(scale, inv), tf.subtract(input_layer, mean)), offset)
    return return_val

def join_resolutions(low, high):
    # resize low res image to higher res size
    lower_norm = spatial_batch_norm(tf.image.resize_nearest_neighbor(low, high.get_shape().as_list()[1:3]))
    higher_norm = spatial_batch_norm(high)
    return tf.concat([lower_norm, higher_norm], 3)