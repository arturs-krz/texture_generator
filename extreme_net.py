import tensorflow as tf
import numpy as np

from utilities import *

class ExtremeNet:
    def __init__(self, input):
        self.conv1_1 = conv(input, 48, 3, 1, activation='elu', name='extreme_conv1_1')
        self.conv1_2 = conv(self.conv1_1, 48, 3, 1, activation='elu', name='extreme_conv1_2')

        self.conv2_1 = conv(self.conv1_2, 80, 2, 2, activation='elu', name='extreme_conv2_1', padding='VALID')
        self.conv2_2 = conv(self.conv2_1, 80, 3, 1, activation='elu', name='extreme_conv2_2')

        self.conv3_1 = conv(self.conv2_2, 112, 2, 2, activation='elu', name='extreme_conv3_1', padding='VALID')
        self.conv3_2 = conv(self.conv3_1, 112, 3, 1, activation='elu', name='extreme_conv3_2')

        self.conv4_1 = conv(self.conv3_2, 176, 2, 2, activation='elu', name='extreme_conv4_1', padding='VALID')
        self.conv4_2 = conv(self.conv4_1, 176, 3, 1, activation='elu', name='extreme_conv4_2')
        self.conv4_3 = conv(self.conv4_2, 176, 3, 1, activation='elu', name='extreme_conv4_3')
