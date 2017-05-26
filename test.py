import skimage
import numpy as np
import tensorflow as tf
import os.path
import os
import sys, getopt

import vgg16
import extreme_net
import utils
from PIL import Image

from utilities import *

descriptor = 'VGG_tfmodel'
restore = True
image_name = "pebbles"
activation = "leaky_relu"
batch_size = 1
iterations = 2000
alpha = 0.01
savediff = 0

opts, args = getopt.getopt(sys.argv[1:], "ni:t:b:c:l:a:d:", ["norestore", "iterations=","target=","batch=","continue=","learnrate=","activation=","descriptor="])
for opt, arg in opts:
    if opt in ("-n", "--norestore"):
        restore = False
    elif opt in ("-i", "--iterations"):
        iterations = int(arg)
    elif opt in ("-t", "--target"):
        image_name = arg
    elif opt in ("-b", "--batch"):
        batch_size = int(arg)
    elif opt in ("-c", "--continue"):
        savediff = int(arg)
    elif opt in ("-l", "--learnrate"):
        alpha = float(arg)
    elif opt in ("-a", "--activation"):
        activation = arg    
    elif opt in ("-d", "--descriptor"):
        descriptor = arg

with tf.device('/gpu:0'):
# with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.get_default_graph().name_scope('generator'):

            # Uļjanova arhitektūra
            # init_noise = [
            #     tf.placeholder("float", shape=[batch_size,14,14,3]),
            #     tf.placeholder("float", shape=[batch_size,28,28,3]),
            #     tf.placeholder("float", shape=[batch_size,56,56,3]),
            #     tf.placeholder("float", shape=[batch_size,112,112,3]), 
            #     tf.placeholder("float", shape=[batch_size,224,224,3]),
            # ] 

            # current_aggregate = init_noise[0]
            # current_channels = 8
            # for index, noise_layer in enumerate(init_noise[1:]):  # skip first
            #     low_conv1 = conv(current_aggregate, current_channels, 3, 1, name='gen_low_conv{}_1'.format(index), activation=activation)
            #     low_conv2 = conv(low_conv1, current_channels, 3, 1, name='gen_low_conv{}_2'.format(index), activation=activation)
            #     low_conv3 = conv(low_conv2, current_channels, 1, 1, name='gen_low_conv{}_3'.format(index), activation=activation)

            #     high_conv1 = conv(noise_layer, 8, 3, 1, name='gen_high_conv{}_1'.format(index), activation=activation)
            #     high_conv2 = conv(high_conv1, 8, 3, 1, name='gen_high_conv{}_2'.format(index), activation=activation)
            #     high_conv3 = conv(high_conv2, 8, 1, 1, name='gen_high_conv{}_3'.format(index), activation=activation)

            #     current_channels += 8
            #     current_aggregate = join_resolutions(low_conv3, high_conv3)
                
            # result_conv1 = conv(current_aggregate, current_channels, 3, 1, name='gen_result_1', activation=activation)
            # result_conv2 = conv(result_conv1, current_channels, 3, 1, name='gen_result_2', activation=activation)
            # result_conv3 = conv(result_conv2, current_channels, 1, 1, name='gen_result_3', activation=activation)

            # result = conv(result_conv3, 3, 1, 1, name='gen_final', activation=activation)

            single_noise = tf.placeholder("float", shape=[1,224,224,16])
            conv1_1 = conv(single_noise, 16, 3, 1, name='gen_conv1_1')
            conv1_2 = conv(conv1_1, 16, 3, 1, name='gen_conv1_2')
            # conv2_1 = conv(conv1_2, 16, 3, 1, name='gen_conv2_1')
            # conv2_2 = conv(conv2_1, 16, 3, 1, name='gen_conv2_2')
            result = conv(conv1_2, 3, 1, 1, name='gen_final')


            print('Result shape: ', result.get_shape())

            tf.summary.image('Output image', result)


            
            # used_layers = [
            #     ('conv1_1', 0.2),
            #     ('conv2_1', 0.2),
            #     ('conv3_1', 0.2),
            #     ('conv4_1', 0.2),
            #     ('conv5_1', 0.2)
            # ]
            
            image_path = "data/{}.jpg".format(image_name)

            img1 = utils.load_image(image_path)
            target_image = tf.to_float(tf.constant(img1.reshape((1, 224, 224, 3))))

            if descriptor == 'VGG_npy':

                used_layers = [
                    ('conv1_1', 0.10),
                    ('conv2_1', 0.15),
                    ('conv3_1', 0.20),
                    ('conv4_1', 0.25),
                    ('conv5_1', 0.30)
                ]

                vgg_ref = vgg16.Vgg16()
                with tf.name_scope("ref_vgg"):
                    vgg_ref.build(target_image)

                target_activations = [sess.run(getattr(vgg_ref, layer[0])) for layer in used_layers]

                # batched_result = get_random_batch(result, batch_size=batch_size)

                descriptor_net = vgg16.Vgg16()
                with tf.name_scope("content_vgg"):            
                    descriptor_net.build(result)
                total_loss = tf.add_n([gram_loss(target_activations[i], getattr(descriptor_net, layer[0]), layer_weight=layer[1], batch_size=batch_size) for i, layer in enumerate(used_layers)])

            elif descriptor == 'VGG_tfmodel':
                used_layers = [
                    ('conv1_1', 0.10),
                    ('conv2_1', 0.15),
                    ('conv3_1', 0.20),
                    ('conv4_1', 0.25),
                    ('conv5_1', 0.30)
                ]

                with open("data/vgg16.tfmodel", mode='rb') as f:
                    file_content = f.read()
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file_content)
                tf.import_graph_def(graph_def, input_map={"images": target_image}, name='vgg_ref')
            
                target_activations = [sess.run(activations_for_layer(layer, ref=True)) for layer in used_layers]

                tf.import_graph_def(graph_def, input_map={"images": result}, name='vgg')
                total_loss = tf.add_n([gram_loss(target_activations[i], activations_for_layer(layer), layer_weight=layer[1], batch_size=batch_size) for i, layer in enumerate(used_layers)]) 

            elif descriptor == 'ExtremeNet':
                used_layers = [
                    ('conv1_1', 1.0),
                    ('conv1_2', 1.0),
                    ('conv2_1', 1.0),
                    ('conv2_2', 1.0),
                    ('conv3_1', 1.0),
                    ('conv3_2', 1.0),
                    ('conv4_1', 1.0),
                    ('conv4_2', 1.0),
                    ('conv4_3', 1.0)
                ]

                extreme_ref = extreme_net.ExtremeNet(target_image, sess)
                target_activations = [sess.run(getattr(extreme_ref, layer[0])) for layer in used_layers]

                descriptor_net = extreme_net.ExtremeNet(result, sess)
                print([getattr(descriptor_net, layer[0]) for layer in used_layers])
                total_loss = tf.add_n([gram_loss(target_activations[i], getattr(descriptor_net, layer[0]), layer_weight=layer[1], batch_size=batch_size) for i, layer in enumerate(used_layers)])
            
            
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp')

            tvars = tf.trainable_variables()
            t_vars = [var for var in tvars if 'gen_' in var.name]
            print("Found {} trainable variables".format(len(t_vars)))
            train_step = optimizer.minimize(total_loss, var_list=t_vars)

            tf.summary.scalar('loss', total_loss)
            writer = tf.summary.FileWriter('.tmp/logs/', graph=tf.get_default_graph())

            summary_op = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            sess.run(init)
            if restore:
                print("Checking for model_{}.ckpt".format(image_name))
                if os.path.isfile("data/model_{}.ckpt.index".format(image_name)):
                    print("Loading existing model...")
                    saver.restore(sess, "data/model_{}.ckpt".format(image_name))
            
            if not os.path.exists("./output/{}/".format(image_name)):
                os.makedirs("./output/{}/".format(image_name))

            print("Batch size: {}".format(batch_size))
            for i in range(iterations):
                
                # batch = [
                #     np.random.rand(batch_size, 14, 14, 3),
                #     np.random.rand(batch_size, 28, 28, 3),
                #     np.random.rand(batch_size, 56, 56, 3),
                #     np.random.rand(batch_size, 112, 112, 3),
                #     np.random.rand(batch_size, 224, 224, 3),
                # ]
                # feed={}
                # for index, layer in enumerate(init_noise):
                #     feed[layer] = batch[index]
                feed={single_noise: np.random.rand(1, 224, 224, 16)}
        
                train_step.run(session=sess, feed_dict=feed)
                summary, loss_value = sess.run([summary_op, total_loss], feed_dict=feed)
                writer.add_summary(summary, i)
                if i%10 == 0:
                    print("Iteration #{}: loss = {}".format(i, loss_value))
                # if i%50 == 0:
                    img = result.eval(session=sess, feed_dict=feed)[0,:,:,:].reshape((224, 224, 3))
                    img = np.clip(np.array(img) * 255.0, 0, 255).astype('uint8')
                    skimage.io.imsave("output/{}/iteration-{}.jpeg".format(image_name, i + savediff), img)

            saver.save(sess, "data/model_{}.ckpt".format(image_name))
            
            img = result.eval(session=sess, feed_dict=feed)[0,:,:,:].reshape((224, 224, 3))
            img = np.clip(np.array(img) * 255.0, 0, 255).astype('uint8')
            skimage.io.imsave("output/final.jpeg", img)