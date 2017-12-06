# coding: utf-8
import numpy as np
import tensorflow as tf
import scipy.misc
import Generative_Network
import Discriminative_Network
import matplotlib.pyplot as plt
import skimage.color as color

palettepath = './lab_palette.npy'

quantized_lab = np.load(palettepath)

net_path = '../imagenet-vgg-verydeep-19.mat'


def g_network(image, quantized_lab, reuse=False,):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    '''
    在重复使用的时候, 一定要在代码中强调 scope.reuse_variables(), 否则系统将会报错, 以为你只是单纯的不小心重复使用到了一个变量
    '''
    prob = Generative_Network.net(image)
    fake_image = Generative_Network.mapping(prob, quantized_lab)
    return fake_image

def d_network(image, weights, reuse = False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    prob, logits = Discriminative_Network.discriminator(image, weights)

    return prob, logits

'''

Testing Data

'''
CROP_SIZE = 128  # 256
img_path = '../coast_bea3.jpg'
img = scipy.misc.imread(img_path, mode = 'RGB')

# plt.figure(1)
# plt.imshow(img)

img = scipy.misc.imresize(img, (CROP_SIZE, CROP_SIZE)) # Image Resizing
plt.figure(2)
plt.imshow(img)
weights, mean_pixel = Discriminative_Network.load_net(net_path)
img = Discriminative_Network.preprocess(img, mean_pixel)
input_img = np.array([img]) # Add to batch


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [1, CROP_SIZE, CROP_SIZE, 3])

G_sample = g_network(X, quantized_lab)
D_real, D_logit_real = d_network(X, weights)
D_fake, D_logit_fake = d_network(G_sample, weights)

with tf.name_scope('D_loss'):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake

with tf.name_scope('G_loss'):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

tvar = tf.trainable_variables()
dvar = [var for var in tvar if 'discriminator' in var.name] # Find variable in DN
gvar = [var for var in tvar if 'generator' in var.name] # Find variable in GB
# print ('dvar:',dvar)
# print ('gvar:',gvar)

with tf.name_scope('train'):
    # adam_train = tf.train.AdamOptimizer.__init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    d_train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(D_loss, var_list=dvar)
    g_train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(G_loss, var_list=gvar)
    # d_train = adam_train.minimize(D_loss, var_list=dvar)
    # g_train = adam_train.minimize(G_loss, var_list=gvar)

samples = []
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(100):

        if i % 10 == 0:
            samples = sess.run(G_sample, feed_dict={X: input_img})

        _, D_loss_curr = sess.run([d_train, D_loss], feed_dict={X: input_img})
        _, G_loss_curr = sess.run([g_train, G_loss], feed_dict={X: input_img})

        if i % 10 == 0:
            print(D_loss_curr, G_loss_curr)

        # if i == 99:
            # print( (samples[0]).shape )
            # print( np.amax(samples[0], axis=1) )
            # img_rgb_out = (255*np.clip( color.lab2rgb( (samples[0]).astype(np.float64, copy=False)) ,0,1 )).astype('uint8') # convert back to rgb
            # plt.figure(3)
            # plt.imshow(img_rgb_out)
        L_ch = np.clip(samples.flatten()[0::3] ,0,100)
        a_ch = samples.flatten()[1::3]
        b_ch = samples.flatten()[2::3]
        print ( max(L_ch),min(L_ch) )
        print ( max(a_ch),min(a_ch) )
        print ( max(b_ch),min(b_ch) )



# sess.close()
# plt.figure(4)
# plt.imshow(input_img[0])
# plt.show()



