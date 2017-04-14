import tensorflow as tf

def weight_var(shape):
    # atgriež random vērtības no normālsadalījuma max 2 standartnoviržu attālumā
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_transpose(input, output_shape, filter_w=5, filter_h=5, stride_w=2, stride_h=2):
    input_shape = input.get_shape().as_list()

    # W x H x output_channels x input_channels
    filter = weight_var(shape=[filter_w, filter_h, output_shape[-1], input_shape[-1]])
    deconv = tf.nn.conv2d_transpose(
        value=input,
        filter=filter,
        output_shape=output_shape,
        strides=[1,stride_w,stride_h,1]
    )

    return deconv