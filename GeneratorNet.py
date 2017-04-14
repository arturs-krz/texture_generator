import numpy as np
import tensorflow as tf
from utilities import *


class GeneratorNet:
    def __init__(self):
        self.layers = []

    def build(self):

        # Starting data - random 4x4 noise (x3 color channels)
        self.init_noise = tf.random_normal(shape=[1, 224, 224, 3])
       
        conv_test_filter1 = weight_var(shape=[16, 16, 3, 3])
        conv_test_bias1 = bias_var(shape=[3])
        self.conv_test1 = tf.nn.elu(tf.nn.conv2d(
            input=self.init_noise,
            filter=conv_test_filter1, strides=[1, 1, 1, 1], padding="SAME") + conv_test_bias1)

        conv_test_filter2 = weight_var(shape=[8, 8, 3, 3])
        conv_test_bias2 = bias_var(shape=[3])
        self.conv_test2 = tf.nn.elu(tf.nn.conv2d(input=self.conv_test1, filter=conv_test_filter2, strides=[1, 1, 1, 1], padding="SAME") + conv_test_bias2)

        conv_test_filter3 = weight_var(shape=[3, 3, 3, 3])
        conv_test_bias3 = bias_var(shape=[3])
        self.conv_test3 = tf.nn.relu(tf.nn.conv2d(
            input=self.conv_test2,
            filter=conv_test_filter3, strides=[1, 1, 1, 1], padding="SAME") + conv_test_bias3)


        conv_test_filter4 = weight_var(shape=[3, 3, 3, 3])
        conv_test_bias4 = bias_var(shape=[3])
        self.conv_test4 = tf.nn.elu(tf.nn.conv2d(input=self.conv_test3, filter=conv_test_filter4, strides=[1, 1, 1, 1], padding="SAME") + conv_test_bias4)

        conv_test_filter5 = weight_var(shape=[2, 2, 3, 3])
        conv_test_bias5 = bias_var(shape=[3])
        self.conv_test5 = tf.nn.relu(tf.nn.conv2d(input=self.conv_test4, filter=conv_test_filter5, strides=[1, 1, 1, 1], padding="SAME") + conv_test_bias5)


        self.result = tf.nn.tanh(self.conv_test5)
        # self.init_noise = tf.random_normal(shape=[1, 7, 7, 384])

        # # b1 = bias_var(shape=[128])
        # b2 = bias_var(shape=[192])
        # b3 = bias_var(shape=[96])
        # b4 = bias_var(shape=[48])
        # b5 = bias_var(shape=[12])
        # b6 = bias_var(shape=[3])

        # # self.conv1 = tf.nn.relu(conv2d_transpose(self.init_noise, [1, 7, 7, 128]) + b1)

        # self.conv2 = tf.nn.relu(conv2d_transpose(self.init_noise, [1, 14, 14, 192]) + b2)
        # self.conv3 = tf.nn.relu(conv2d_transpose(self.conv2, [1, 28, 28, 96]) + b3)
        # self.conv4 = tf.nn.relu(conv2d_transpose(self.conv3, [1, 56, 56, 48]) + b4)
        # self.conv5 = tf.nn.relu(conv2d_transpose(self.conv4, [1, 112, 112, 12]) + b5)
        # self.conv6 = tf.nn.relu(conv2d_transpose(self.conv5, [1, 224, 224, 3]) + b6)

        # self.result = tf.nn.tanh(self.conv6)

        self.t_vars = tf.trainable_variables()
        # self.conv2_1 = tf.nn.conv2d(
        #     input=tf.nn.conv2d_transpose(
        #         value=self.conv1_1, 
        #         filter=transpose1_filter,
        #         output_shape=[1, 8, 8, 3],
        #         strides=[1, 1, 1, 1]
        #     ),
        #     filter=conv2_filter,
        #     strides=[1, 1, 1, 1],
        #     padding="SAME"
        # )
        

    def run(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)

        res = self.result.eval()
        return res
