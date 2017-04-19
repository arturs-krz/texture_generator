import tensorflow as tf

# shape = [size, size, in_channels, out_channels]
def weight_var(shape):
    # atgriež random vērtības no normālsadalījuma max 2 standartnoviržu attālumā
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv(input, num_filters, filter_size, stride_len, activation='relu', name='conv'):
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        weights = weight_var(shape=[filter_size, filter_size, input_shape[3], num_filters])
        tf.summary.histogram('conv_weights', weights)

        output = tf.nn.conv2d(
            input=input, 
            filter=weights, 
            strides=[1, stride_len, stride_len, 1], 
            padding="SAME")
        
        tf.summary.histogram('conv_output', output)

        if activation == 'relu':
            output = tf.nn.relu(output)
        
        return output

def residual_conv(input, filter_size, name='residual'):
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        residual_layer = conv(input, input_shape[3], filter_size, 1, name=name+'_residual')
        return input + conv(residual_layer, input_shape[3], filter_size, 1, activation='relu')


def conv_transpose(input, num_filters, filter_size, stride_len, name='conv_transpose'):
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        weights = weight_var(shape=[filter_size, filter_size, num_filters, input_shape[3]])
        tf.summary.histogram('transpose_weights', weights)

        shape = tf.stack([input_shape[0], input_shape[1] * stride_len, input_shape[2] * stride_len, num_filters])
        output = tf.nn.conv2d_transpose(input, weights, shape, [1, stride_len, stride_len, 1], padding='SAME')

        tf.summary.histogram('transpose_output', output)

        return tf.nn.relu(output)