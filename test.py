import skimage
import numpy as np
import tensorflow as tf

# from VGG import vgg16
# from VGG import utils
import vgg16
import utils
from PIL import Image

from utilities import *
# import GeneratorNet as gen


# layer => shape = {1, width, height, filters}
def gram_matrix(layer, area, filters):
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

    reference_gram = gram_matrix(reference_layer, feature_map_area, feature_map_filters)
    generated_gram = gram_matrix(generated_layer, feature_map_area, feature_map_filters)

    print(reference_gram)
    print(generated_gram)

    # TODO: Izpētīt, kas te fuckin notiek
    return (1 / (4 * feature_map_filters**2 * feature_map_area**2)) * tf.reduce_sum(tf.pow(generated_gram - reference_gram, 2))
    
    # result = tf.reduce_sum(tf.pow(tf.subtract(generated_gram, reference_gram), 2))
    # result = tf.pow(tf.subtract(generated_gram, reference_gram), 2)
    # print(result)
    
    # return result

def get_loss(reference, generated):
    # loss = sum([layer_loss(sess.run([reference[i], generated[i]])) for i in range(len(reference))])
    loss = sum([layer_loss(reference[i], generated[i]) for i in range(len(reference))])
    return loss
    

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

with tf.device('/gpu:0'):
# with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        img1 = utils.load_image("data/pebbles_small.jpg")
        batch1 = img1.reshape((1, 224, 224, 3))
        # batch = np.concatenate((batch1, batch2), 0)
        images = tf.placeholder("float", [1, 224, 224, 3])
        # feed_dict = {images: batch1}

        vgg_ref = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg_ref.build(images)
 
        gold_conv5_1, gold_conv3_1, gold_conv1_2 = sess.run([vgg_ref.conv5_1, vgg_ref.conv3_1, vgg_ref.conv1_2], feed_dict={images: batch1})        
        gold_5_placeholder = tf.placeholder("float", vgg_ref.conv5_1.get_shape())
        gold_3_placeholder = tf.placeholder("float", vgg_ref.conv3_1.get_shape())
        gold_1_placeholder = tf.placeholder("float", vgg_ref.conv1_2.get_shape())

        
        # generator = gen.GeneratorNet()
        # generator.build()


        with tf.name_scope('generator'):
            # Starting data - random 4x4 noise (x3 color channels)
            # init_noise = tf.random_normal(shape=[1, 224, 224, 3])
            init_noise = tf.placeholder("float", shape=[1,224,224,3])
            tf.summary.histogram('Init noise', init_noise)
        
            conv1 = conv(init_noise, 32, 9, 1, activation=None, name='conv1')
            conv2 = conv(conv1, 64, 3, 2, activation=None, name='conv2')
            conv3 = conv(conv2, 128, 3, 2, activation=None, name='conv3')

            conv4 = conv(conv3, 128, 3, 2, activation='relu', name='conv4_relu')

            residual1 = residual_conv(conv4, 3, name='residual1')
            residual2 = residual_conv(residual1, 3, name='residual2')
            residual3 = residual_conv(residual2, 3, name='residual3')

            transpose1 = conv_transpose(residual3, 64, 3, 2, name='transpose1')
            transpose2 = conv_transpose(transpose1, 32, 3, 2, name='transpose2')
            transpose3 = conv_transpose(transpose2, 3, 9, 2, name='transpose3')
            transpose4 = conv_transpose(transpose3, 3, 3, 1, name='transpose4')

            result = tf.nn.tanh(transpose4) * 150 + 255./2
            tf.summary.image('Output image', result)

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):            
            vgg.build(result)

        # print(vgg.conv5_1.eval(session=sess))

        # loss = get_loss(reference=[gold_1_placeholder], generated=[vgg.conv1_2])
        # loss = get_loss(reference=[gold_conv1_2, gold_conv3_1, gold_conv5_1], generated=[vgg.conv1_2, vgg.conv3_1, vgg.conv5_1])
        # loss = get_loss(reference=[gold_1_placeholder, gold_3_placeholder, gold_5_placeholder], generated=[vgg.conv1_2, vgg.conv3_1, vgg.conv5_1])


        loss = tf.reduce_mean(tf.pow(gold_3_placeholder - vgg.conv3_1, 2))
        print(loss)

        # alpha - training rate
        alpha = 0.001
        # train_step = tf.train.AdamOptimizer(alpha).minimize(loss, var_list=generator.t_vars)
        # train_step = tf.train.AdamOptimizer(alpha).minimize(loss)
        opt_func = tf.train.AdamOptimizer(alpha)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        train_step = opt_func.apply_gradients(zip(grads, tvars))

        tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter('.tmp/logs/', graph=tf.get_default_graph())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        sess.run(init)
        
        iterations = 1000
        # batch_size = 1
        
        for i in range(iterations):
            batch = np.random.rand(1, 224, 224, 3)
            feed={init_noise: batch, gold_5_placeholder: gold_conv5_1, gold_3_placeholder: gold_conv3_1, gold_1_placeholder: gold_conv1_2}    
    
            train_step.run(session=sess, feed_dict=feed)
            summary, loss_value = sess.run([summary_op, loss], feed_dict=feed)
            writer.add_summary(summary, i)
            if i%10 == 0:
                print("Iteration #{}: loss = {}".format(i, loss_value))
          
        # Kad iterācijas izgājušas, uzģenerējam un saglabājam bildi ar esošajām vērtībām
        img = result.eval(session=sess, feed_dict={init_noise: np.random.rand(1,224,224,3)})
        # img = result.eval(session=sess)
        img = Image.fromarray(np.asarray(img)[0], "RGB")
        img.save('output/result.bmp')
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
        
        