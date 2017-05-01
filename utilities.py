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
            padding="SAME",
            name=name
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
    # shuffled - batch x channels x width x height
    shuffled = tf.transpose(activation_layer, perm=[0, 3, 1, 2])
    layer_shape = activation_layer.get_shape().as_list()
    
    # N filters / feature maps
    N = layer_shape[1]
    # M = x * y
    # M = layer_shape[2] * layer_shape[3]

    F = tf.reshape(activation_layer, shape=[N, -1])
    FT = tf.transpose(F, perm=[1, 0])
    # G = tf.matmul(F,FT) / M
    G = tf.matmul(F, FT)
    return G


def gram_loss(target_gram, generated, layer_weight=1.0):
    layer_shape = generated.get_shape().as_list()
    # N filters / feature maps
    N = layer_shape[3]
    # M = x * y
    M = layer_shape[1] * layer_shape[2]
    G = gram_matrix(generated)

    gram_diff = G - target_gram
    # loss = layer_weight/4. * tf.reduce_sum(tf.pow(gram_diff,2)) / (N**2)
    loss = tf.divide(tf.reduce_sum(tf.pow(gram_diff, 2)), 4 * (N ** 2) * (M ** 2))
    # gradient = tf.reshape(layer_weight * tf.transpose(tf.matmul(FT, gram_diff)) / (M * N**2), shape=layer_shape)
    # return [loss, gradient]
    return loss


def style_loss(layers, target_activations):
    """
    [0:num_style, :, :, :] holds i style images,
    [num_style : num_style+num_content, :, :, :] holds j content images,
    [num_style+num_content : num_style+num_content+num_synth, :, :, :] holds k synthesized images
    """
    activations = [activations_for_layer(layer) for layer in layers]
    gramians = [gramian_for_layer(layer, target_activations[i]) for i, layer in enumerate(layers)]

    print(gramians)

    # Slices are for style and synth image
    gramian_diffs = [
        tf.subtract(
            tf.tile(tf.slice(g, [0, 0, 0], [1, -1, -1]), [1, 1, 1]),
            tf.slice(g, [1, 0, 0], [1, -1, -1]))
        for g in gramians]
    
    # gramian_diffs = [(target_grams[i] - gram) for i, gram in enumerate(gramians)]

    Ns = [g.get_shape().as_list()[2] for g in gramians]
    Ms = [a.get_shape().as_list()[1] * a.get_shape().as_list()[2] for a in activations]
    scaled_diffs = [tf.square(g) for g in gramian_diffs]
    style_loss = tf.divide(
        tf.add_n([tf.divide(tf.reduce_sum(x), 4 * (N ** 2) * (M ** 2)) for x, N, M in zip(scaled_diffs, Ns, Ms)]),
        len(layers))
    return style_loss




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

def gramian(activations):
    # Takes (batches, channels, width, height) and computes gramians of dimension (batches, channels, channels)
    activations_shape = activations.get_shape().as_list()
    """
    Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    the entire gramian in a single matrix multiplication.
    """
    vectorized_activations = tf.reshape(activations,
                                        [activations_shape[0], activations_shape[1], -1])
    print(activations)
    print(vectorized_activations)
    transposed_vectorized_activations = tf.transpose(vectorized_activations, perm=[0, 2, 1])
    mult = tf.matmul(vectorized_activations, transposed_vectorized_activations)
    return mult

def gramian_for_layer(layer, target_layer):
    """
    Returns a matrix of cross-correlations between the activations of convolutional channels in a given layer.
    """
    activations = tf.concat([target_layer, activations_for_layer(layer)], 0)

    # Reshape from (batch, width, height, channels) to (batch, channels, width, height)
    shuffled_activations = tf.transpose(activations, perm=[0, 3, 1, 2])
    return gramian(shuffled_activations)

def activations_for_layer(layer, ref=False):
    """
    :param layer: A tuple that indexes into the convolutional blocks of the VGG Net
    """
    return tf.get_default_graph().get_tensor_by_name("generator/{}/{}/Relu:0".format('vgg_ref' if ref else 'vgg', layer[0]))
    