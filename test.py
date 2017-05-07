import skimage
import numpy as np
import tensorflow as tf
import os.path
import sys, getopt

# from VGG import vgg16
# from VGG import utils
import vgg16
# import vgg19
import utils
from PIL import Image

from utilities import *
# import GeneratorNet as gen

restore = True
image_name = "pebbles"
batch_size = 16
iterations = 2000

opts, args = getopt.getopt(sys.argv[1:], "ni:t:b:", ["norestore", "iterations=","target=","batch="])
for opt, arg in opts:
    if opt in ("-n", "--norestore"):
        restore = False
    elif opt in ("-i", "--iterations"):
        iterations = arg
    elif opt in ("-t", "--target"):
        image_name = arg
    elif opt in ("-b", "--batch"):
        batch_size = arg

# layer => shape = {1, width, height, filters}
def gram_matrix_old(layer, area, filters):
    # Saplacinam slāņa garums x platums vienā dimensijā.
    feature_map = tf.reshape(layer, (area, filters))
    # Reizinam ar savu transponēto matricu
    return tf.matmul(tf.transpose(feature_map), feature_map)

def layer_loss(reference_layer, generated_layer):
    # Slāņa garums x platums
    rshape = reference_layer.get_shape().as_list()

    # feature_map_area = reference_layer.shape[1] * reference_layer.shape[2]
    # Slāņa pēdējā dimensija
    # feature_map_filters = reference_layer.shape[3]

    feature_map_area = rshape[1] * rshape[2]
    feature_map_filters = rshape[3]

    # reference_gram = gram_matrix(reference_layer, feature_map_area, feature_map_filters)
    # generated_gram = gram_matrix(generated_layer, feature_map_area, feature_map_filters)

    # return (1 / (3 * feature_map_filters**2 * feature_map_area**2)) * tf.reduce_sum(tf.pow(generated_gram - reference_gram, 2))
    return (1 / (4 * feature_map_filters * feature_map_area)) * tf.reduce_sum(tf.pow(generated_layer - reference_layer,2))
    # return tf.reduce_sum(tf.pow(generated_gram - reference_gram, 2))
    
    # result = tf.reduce_sum(tf.pow(tf.subtract(generated_gram, reference_gram), 2))
    # result = tf.pow(tf.subtract(generated_gram, reference_gram), 2)
    # print(result)
    
    # return result

def get_loss(reference, generated):
    loss = sum([layer_loss(reference[i], generated[i]) for i in range(len(reference))])
    return loss
    

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

with tf.device('/gpu:0'):
# with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.get_default_graph().name_scope('generator'):
            # Starting data - random 4x4 noise (x3 color channels)

            # Uļjanova arhitektūra
            init_noise = [
                tf.placeholder("float", shape=[batch_size,14,14,3]),
                tf.placeholder("float", shape=[batch_size,28,28,3]),
                tf.placeholder("float", shape=[batch_size,56,56,3]),
                tf.placeholder("float", shape=[batch_size,112,112,3]), 
                tf.placeholder("float", shape=[batch_size,224,224,3]),
            ] 

            current_aggregate = init_noise[0]
            current_channels = 8
            for index, noise_layer in enumerate(init_noise[1:]):  # skip first
                low_conv1 = conv(current_aggregate, current_channels, 3, 1, name='gen_low_conv{}_1'.format(index))
                low_conv2 = conv(low_conv1, current_channels, 3, 1, name='gen_low_conv{}_2'.format(index))
                low_conv3 = conv(low_conv2, current_channels, 1, 1, name='gen_low_conv{}_3'.format(index))

                high_conv1 = conv(noise_layer, 8, 3, 1, name='gen_high_conv{}_1'.format(index))
                high_conv2 = conv(high_conv1, 8, 3, 1, name='gen_high_conv{}_2'.format(index))
                high_conv3 = conv(high_conv2, 8, 1, 1, name='gen_high_conv{}_3'.format(index))

                current_channels += 8
                current_aggregate = join_resolutions(low_conv3, high_conv3)
                
            result_conv1 = conv(current_aggregate, 3, 3, 1, name='gen_result_1')
            result_conv2 = conv(result_conv1, 3, 3, 1, name='gen_result_2')
            result_conv3 = conv(result_conv2, 3, 1, 1, name='gen_result_3')

            result = conv(result_conv3, 3, 1, 1, name='gen_final')
            print('Result shape: ', result.get_shape())

            tf.summary.image('Output image', result)


            # total_loss = tf.divide(tf.add_n([gram_loss(target_grams[layer[0]], getattr(vgg, layer[0]), layer_weight=layer[1]) for layer in used_layers]), len(used_layers))

            used_layers = [
                ('conv1_1', 0.10),
                ('conv2_1', 0.15),
                ('conv3_1', 0.20),
                ('conv4_1', 0.25),
                ('conv5_1', 0.30)
            ]
            
            image_path = "data/{}.jpg".format(image_name)

            img1 = utils.load_image(image_path)
            target_image = tf.to_float(tf.constant(img1.reshape((1, 224, 224, 3))))
            vgg_ref = vgg16.Vgg16()
            with tf.name_scope("ref_vgg"):
                vgg_ref.build(target_image)

            target_activations = [sess.run(getattr(vgg_ref, layer[0])) for layer in used_layers]

            # batched_result = get_random_batch(result, batch_size=batch_size)

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):            
                vgg.build(result)

            # total_loss = tf.divide(tf.add_n([gram_loss(target_activations[i], getattr(vgg, layer[0]), layer_weight=layer[1]) for i, layer in enumerate(used_layers)]), len(used_layers))
            total_loss = tf.add_n([gram_loss(target_activations[i], getattr(vgg, layer[0]), layer_weight=layer[1], batch_size=batch_size) for i, layer in enumerate(used_layers)])

            # input_ref = [
            #     utils.load_image(image_path, 14).reshape((1, 14, 14, 3)),
            #     utils.load_image(image_path, 28).reshape((1, 28, 28, 3)),
            #     utils.load_image(image_path, 56).reshape((1, 56, 56, 3)),
            #     utils.load_image(image_path, 112).reshape((1, 112, 112, 3)),
            #     utils.load_image(image_path, 224).reshape((1, 224, 224, 3))
            # ]

            # with open("data/vgg16.tfmodel", mode='rb') as f:
            #     file_content = f.read()
            # graph_def = tf.GraphDef()
            # graph_def.ParseFromString(file_content)
            # tf.import_graph_def(graph_def, input_map={"images": target_image}, name='vgg_ref')
            
            # target_activations = [sess.run(activations_for_layer(layer, ref=True)) for layer in used_layers]

            # tf.import_graph_def(graph_def, input_map={"images": result}, name='vgg')
            
            # total_loss = style_loss(used_layers, target_activations)
            
            # alpha - training rate
            alpha = 0.01
            
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp')

            tvars = tf.trainable_variables()
            t_vars = [var for var in tvars if 'gen_' in var.name]
            print("Found {} trainable variables".format(len(t_vars)))
            train_step = optimizer.minimize(total_loss, var_list=t_vars)
            # train_step = optimizer.minimize(total_loss)

            # grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, t_vars), 1)
            # train_step = opt_func.apply_gradients(zip(grads, t_vars))

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
            
            # batch_size = 1
            # batch = (0.6 * np.random.uniform(-20,20,(1,28,28,3)).astype("float32")) + (0.4 * input_ref)
            

            for i in range(iterations):
                # batch = (np.random.rand(1, 224, 224, 3)*32)+112
                # batch = batch1
                # batch = [
                #     (0.6 * np.random.uniform(-20, 20, (1, 14, 14, 3))) + (0.4 * input_ref[0]),
                #     (0.6 * np.random.uniform(-20, 20, (1, 28, 28, 3))) + (0.4 * input_ref[1]),
                #     (0.6 * np.random.uniform(-20, 20, (1, 56, 56, 3))) + (0.4 * input_ref[2]),
                #     (0.6 * np.random.uniform(-20, 20, (1, 112, 112, 3))) + (0.4 * input_ref[3]),
                #     (0.6 * np.random.uniform(-20, 20, (1, 224, 224, 3))) + (0.4 * input_ref[4])
                # ]
                # batch = [
                #     np.random.rand(1, 14, 14, 3),
                #     np.random.rand(1, 28, 28, 3),
                #     np.random.rand(1, 56, 56, 3),
                #     np.random.rand(1, 112, 112, 3),
                #     np.random.rand(1, 224, 224, 3),
                #     np.random.rand(1, 448, 448, 3),
                #     np.random.rand(1, 896, 896, 3)
                # ]
                batch = [
                    np.random.rand(batch_size, 14, 14, 3),
                    np.random.rand(batch_size, 28, 28, 3),
                    np.random.rand(batch_size, 56, 56, 3),
                    np.random.rand(batch_size, 112, 112, 3),
                    np.random.rand(batch_size, 224, 224, 3),
                ]
                feed={}
                for index, layer in enumerate(init_noise):
                    feed[layer] = batch[index]
        
                train_step.run(session=sess, feed_dict=feed)
                summary, loss_value = sess.run([summary_op, total_loss], feed_dict=feed)
                writer.add_summary(summary, i)
                if i%10 == 0:
                    print("Iteration #{}: loss = {}".format(i, loss_value))
                if i%50 == 0:
                    img = result.eval(session=sess, feed_dict=feed).reshape((224, 224, 3))
                    img = np.clip(np.array(img) * 255.0, 0, 255).astype('uint8')
                    skimage.io.imsave("output/iteration-%d.jpeg" % i, img)

            saver.save(sess, "data/model_{}.ckpt".format(image_name))
            
            img = result.eval(session=sess, feed_dict=feed).reshape((224, 224, 3))
            img = np.clip(np.array(img) * 255.0, 0, 255).astype('uint8')
            skimage.io.imsave("output/final.jpeg", img)