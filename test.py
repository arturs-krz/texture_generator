import skimage
import numpy as np
import tensorflow as tf

# from VGG import vgg16
# from VGG import utils
# import vgg16
import vgg19
import utils
from PIL import Image

from utilities import *
# import GeneratorNet as gen


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
        
        # tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
        # tex_weights = [1e9,1e9,1e9,1e9,1e9]
        used_layers = [
            ('conv1_1', 1.0),
            ('conv2_1', 1.0),
            ('conv3_1', 1.0),
            ('conv4_1', 1.0),
            ('conv5_1', 1.0)
        ]
        image_path = "data/pebbles.jpg"

        img1 = utils.load_image(image_path)
        batch1 = img1.reshape((1, 224, 224, 3))

        input_ref = utils.load_image(image_path, 28).reshape((1, 28, 28, 3))
        
        # batch = np.concatenate((batch1, batch2), 0)
        images = tf.placeholder("float", [1, 224, 224, 3])
        # feed_dict = {images: batch1}

        vgg_ref = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg_ref.build(images)
 
        # target_grams = [sess.run([getattr(vgg_ref, layer[0])], feed_dict={images: batch1 }) for layer in used_layers]
        target_grams = {}
        for layer in used_layers:
            # target_grams[layer[0]] = sess.run([getattr(vgg_ref, layer[0])], feed_dict={images: batch1})
            target_grams[layer[0]] = gram_matrix(getattr(vgg_ref, layer[0]))


        # gold_conv5_1, gold_conv3_1, gold_conv1_1, gold_conv4_2 = sess.run([vgg_ref.conv5_1, vgg_ref.conv3_1, vgg_ref.conv1_1, vgg_ref.conv4_2], feed_dict={images: batch1})        
        # gold_5_placeholder = tf.placeholder("float", vgg_ref.conv5_1.get_shape())
        # gold_3_placeholder = tf.placeholder("float", vgg_ref.conv3_1.get_shape())
        # gold_1_placeholder = tf.placeholder("float", vgg_ref.conv1_1.get_shape())

        # gold_4_content = tf.placeholder("float", vgg_ref.conv4_2.get_shape())

        
        # generator = gen.GeneratorNet()
        # generator.build()


        with tf.name_scope('generator'):
            # Starting data - random 4x4 noise (x3 color channels)

            # Uļjanova arhitektūra
            init_noise = [
                tf.placeholder("float", shape=[1,14,14,3]),
                tf.placeholder("float", shape=[1,28,28,3]),
                tf.placeholder("float", shape=[1,56,56,3]),
                tf.placeholder("float", shape=[1,112,112,3]), 
                tf.placeholder("float", shape=[1,224,224,3]), 
            ] 

            current_aggregate = init_noise[0]
            current_channels = 8
            for noise_layer in init_noise[1:]:  # skip first
                low_conv1 = conv(current_aggregate, current_channels, 3, 1)
                low_conv2 = conv(low_conv1, current_channels, 3, 1)
                low_conv3 = conv(low_conv2, current_channels, 1, 1)

                high_conv1 = conv(noise_layer, 8, 3, 1)
                high_conv2 = conv(high_conv1, 8, 3, 1)
                high_conv3 = conv(high_conv2, 8, 1, 1)

                current_channels += 8
                current_aggregate = join_resolutions(low_conv3, high_conv3)
                
            result_conv1 = conv(current_aggregate, 3, 3, 1)
            result_conv2 = conv(result_conv1, 3, 3, 1)
            result_conv3 = conv(result_conv2, 3, 1, 1)

            result = conv(result_conv3, 3, 1, 1)

            # h1 = conv(init_noise, 9, 9, 2, name="gen_conv1")
            # print(h1)
            # tf.summary.image('First layer', h1)

            # h2 = conv(h1, 3, 5, 1, name="gen_conv2")
            # print(h2)
            # tf.summary.image('Second layer', h2)

            # h3 = conv(h2, 3, 3, 1, name="gen_conv3")
            # print(h3)
            # conv1 = conv(init_noise, 32, 9, 1, activation='relu', name='gen_conv1')
            # conv2 = conv(conv1, 64, 3, 2, activation='relu', name='gen_conv2')
            # conv3 = conv(conv2, 128, 3, 2, activation='relu', name='gen_conv3')

            # residual1 = residual_conv(conv3, 3, name='gen_residual1')
            # residual2 = residual_conv(residual1, 3, name='gen_residual2')
            # residual3 = residual_conv(residual2, 3, name='gen_residual3')

            # transpose1 = conv_transpose(residual3, 64, 3, 2, name='gen_transpose1')
            # transpose2 = conv_transpose(transpose1, 32, 3, 2, name='gen_transpose2')
            # transpose3 = conv_transpose(transpose2, 3, 3, 1, name='gen_transpose3')

            # result = h3
            # result = conv(h1, 3, 3, 1, name='gen_conv')
            # transpose1 = conv_transpose(init_noise, 9, 7, 4, name='gen_transpose1')
            # tf.summary.image('First layer', transpose1)

            # transpose2 = conv_transpose(transpose1, 6, 3, 2, name='gen_transpose2')
            # tf.summary.image('Second layer', transpose2)

            # transpose3 = conv_transpose(transpose2, 3, 4, 2, name='gen_transpose3')
            # tf.summary.image('Third layer', transpose3)

            # conv1 = conv(transpose2, 3, 3, 1, name='gen_conv1')

            # print(transpose1)
            # print(transpose2)
            # print(transpose3)
            # print(conv1)
            # conv2 = conv(conv1, 3, 3, 1, name='gen_conv2')

            # transpose3 = conv_transpose(transpose2, 3, 3, 2, name='gen_transpose3')
            # result = conv1
            tf.summary.image('Output image', result)

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):            
            vgg.build(result)

        # print(vgg.conv5_1.eval(session=sess))

        # loss = get_loss(reference=[gold_3_placeholder], generated=[vgg.conv3_1])
        

        # loss = get_loss(reference=[gold_1_placeholder, gold_3_placeholder, gold_5_placeholder], generated=[vgg.conv1_2, vgg.conv3_1, vgg.conv5_1])
        # loss = get_loss(reference=[gold_4_content], generated=[vgg.conv4_2])
        # Random loss function
        # loss = tf.reduce_sum(0.5*tf.reduce_mean(tf.pow(gold_5_placeholder - vgg.conv5_1, 2)) + 0.3*tf.reduce_mean(tf.pow(gold_3_placeholder - vgg.conv3_1, 2)) + 0.2*tf.reduce_mean(tf.pow(gold_1_placeholder - vgg.conv1_2, 2)))
        # loss = tf.reduce_sum(0.7*layer_loss(gold_3_placeholder,vgg.conv3_1) + 0.3*layer_loss(gold_1_placeholder,vgg.conv1_1))
        # loss = tf.reduce_mean(tf.pow(gold_1_placeholder - vgg.conv1_1, 2))
        # total_loss = tf.zeros([])
        # # total_grad = tf.zeros([])
        # for layer in used_layers:
        #     loss = gram_loss(target_grams[layer[0]], getattr(vgg, layer[0]), layer_weight=layer[1])
        #     total_loss += loss
        #     total_grad += grad

        total_loss = tf.divide(tf.add_n([gram_loss(target_grams[layer[0]], getattr(vgg, layer[0]), layer_weight=layer[1]) for layer in used_layers]), len(used_layers))

        # alpha - training rate
        alpha = 0.01
        # train_step = tf.train.AdamOptimizer(alpha).minimize(loss, var_list=generator.t_vars)
        # train_step = tf.train.AdamOptimizer(alpha).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        train_step = optimizer.minimize(total_loss)
        # t_vars = tf.trainable_variables()
        # t_vars = [var for var in tvars if 'gen_' in var.name]

        # grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, t_vars), 1)
        # train_step = opt_func.apply_gradients(zip(grads, t_vars))

        tf.summary.scalar('loss', total_loss)
        writer = tf.summary.FileWriter('.tmp/logs/', graph=tf.get_default_graph())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        sess.run(init)
        
        iterations = 1000
        # batch_size = 1
        # batch = (0.6 * np.random.uniform(-20,20,(1,28,28,3)).astype("float32")) + (0.4 * input_ref)
        # batch = [
        #     np.random.rand(1, 14, 14, 3),
        #     np.random.rand(1, 28, 28, 3),
        #     np.random.rand(1, 56, 56, 3),
        #     np.random.rand(1, 112, 112, 3),
        #     np.random.rand(1, 224, 224, 3)
        # ]

        for i in range(iterations):
            # batch = (np.random.rand(1, 224, 224, 3)*32)+112
            # batch = batch1
            batch = [
                np.random.uniform(-20., 20., (1, 14, 14, 3)),
                np.random.uniform(-20., 20., (1, 28, 28, 3)),
                np.random.uniform(-20., 20., (1, 56, 56, 3)),
                np.random.uniform(-20., 20., (1, 112, 112, 3)),
                np.random.uniform(-20., 20., (1, 224, 224, 3))
            ]
            feed={images: batch1}
            for index, layer in enumerate(init_noise):
                feed[layer] = batch[index]
    
            train_step.run(session=sess, feed_dict=feed)
            summary, loss_value = sess.run([summary_op, total_loss], feed_dict=feed)
            writer.add_summary(summary, i)
            if i%10 == 0:
                print("Iteration #{}: loss = {}".format(i, loss_value))
          
        # Kad iterācijas izgājušas, uzģenerējam un saglabājam bildi ar esošajām vērtībām
        # img = result.eval(session=sess, feed_dict={init_noise: (0.6 * np.random.uniform(-20,20,(1,28,28,3)).astype("float32")) + (0.4 * input_ref)})
        # img = result.eval(session=sess)
        # img = Image.fromarray(np.asarray(img)[0], "RGB")
        # img.save('output/result.bmp')
        # img.show()
          
          # ------
          
          
            # if not i%10:    
            #     print('Iteration #{}: error = {}'.format(i,1 - accuracy.eval(session=sess,feed_dict={
            #         x: x_batch, gold_y: y_batch, keep_prob: 1.0
            #     })))






        # gram = gram_matrix(gold_conv5_2)

        # gram = sess.run([gram], feed_dict=feed_dict)
        # print(gram)
        
        

        # img = generator.run(sess)
        
        # generated = generator.result.reshape([1, 224, 224, 3])
        

        
        




        # img = Image.fromarray(np.asarray(img)[0], "RGB")
        # img.show()

        
        # prob, conv1_1, conv1_2 = sess.run([vgg.prob, vgg.conv1_1, vgg.conv1_2], feed_dict=feed_dict)
        # gram_matrix(conv1_2)
        # utils.print_prob(prob[0], './VGG/synset.txt')
        # utils.print_prob(prob[1], './synset.txt')
        
        