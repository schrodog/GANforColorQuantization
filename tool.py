from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import random
import collections
import math
import time
import json
import glob

parser = argparse.ArgumentParser()
# file
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

# model parameters
# parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])   # specify do what
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# Summary freq options
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

# parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
# parser.add_argument("--max_epochs", type=int, help="number of training epochs")
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256


Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

def preprocess(image):
    with tf.name_scope("preprocess"):
        return image*2 - 1  # [0, 1] => [-1, 1]

def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2   # [-1, 1] => [0, 1]

def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100], a_chan/b_chan: color channels with input range ~[-110, 110], not exact => all to [-1,1] range
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb

###

def conv_init_vars(net, out_channels, filter_size, transpose=False):
    '''
    According to the previous output, intialize the weight matrix.
    '''
    _, rows, cols, in_channels = [i.value for i in net.get_shape()] ### Obtain in_channels

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    # weights shape = [Kernal size, kernal size, output kernal, input kernal]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev = 0.1, seed=1), dtype=tf.float32)
    return weights_init


def batch_norm(net, train=True):
    '''
    Apply Batch Normalization Function
    BN: Forward norm and then inverse norm.
    formula: y = scale*[(x-mu)/sqrt(variance+epsilon)]+shift
    '''

    batch, rows, cols, channels = [i.value for i in net.get_shape()] ### Shape Meaning: [batchsize, height, width, kernels]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True) ### Calculate the mean and variance of x.Output: One-dimension
    shift = tf.Variable(tf.zeros(var_shape)) ### Inverse Norm
    scale = tf.Variable(tf.ones(var_shape)) ### Inverse Norm
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift ### Applied Batch Normalization


def conv_layer(net, num_filters, filter_size, strides, relu=True):
    '''
    Apply convolution operation (with relu)
    '''
    weights_init = conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    if relu:
        net = tf.nn.relu(net)
    return net

def conv_layer_dila(net, num_filters, filter_size, rate, relu=True):
    '''
    Apply dilation convolution operation (with relu)
    在己有像素上，skip一些pixel/input unchange; 對conv的kernel參數中插一些0的weight => 空間範圍變大
    '''
    weights_init = conv_init_vars(net, num_filters, filter_size)
    #strides_shape = [1, strides, strides, 1]
    net = tf.nn.atrous_conv2d(net, weights_init, rate, 'SAME') # Dialation Convolution

    if relu:
        net = tf.nn.relu(net)
    return net

def conv_tranpose_layer(net, num_filters, filter_size, strides):
    '''
    Inverse of convolution operation
    '''
    weights_init = conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    return tf.nn.relu(net)


def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        # srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn), normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask
            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")
    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically, having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])

        # split color(B) and brightness(A) from image, no B image exists here
        lab = rgb_to_lab(raw_input)
        L_chan, a_chan, b_chan = preprocess_lab(lab)
        a_images = tf.expand_dims(L_chan, axis=2)
        b_images = tf.stack([a_chan, b_chan], axis=2)

    inputs, targets = [a_images, b_images] # A to B

    # synchronize seed for image operations so that we do the same operations to both input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)
    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

###

def net(image):  ### 输入的图像不用normalization
    normalization = batch_norm(image, train=True)
    conv1_1_relu = conv_layer(normalization, 64, 3, 1, relu=True)
    conv1_2_relu = conv_layer(conv1_1_relu, 64, 3, 1, relu=True)
    conv1_2norm = batch_norm(conv1_2_relu, train=True)
    '''
    > conv2_1 > relu2_1 > conv2_2 (Stride:2) > relu2_2 > conv2_2norm
    '''
    conv2_1_relu = conv_layer(conv1_2norm, 128, 3, 1, relu=True)
    conv2_2_relu = conv_layer(conv1_1_relu, 128, 3, 2, relu=True)
    conv2_2norm = batch_norm(conv2_2_relu, train=True)

    '''
    > conv3_1 > relu3_1 > conv3_2 > relu3_2 > conv3_3 (Stride:2)> relu3_3 > conv3_3norm
    '''

    conv3_1_relu = conv_layer(conv2_2norm, 256, 3, 1, relu=True)
    conv3_2_relu = conv_layer(conv3_1_relu, 256, 3, 1, relu=True)
    conv3_3_relu = conv_layer(conv3_2_relu, 256, 3, 2, relu=True)
    conv3_3norm = batch_norm(conv3_3_relu, train=True)
    '''
    conv4_1 (Stride:1,pad:1 dilation: 1)> relu4_1 > conv4_2(same) > relu4_2 > conv4_3(same) > relu4_3 > conv4_3_norm
    tf.nn.atrous_conv2d(net, weights_init, rate, 'SAME')
    conv_layer_dila(net, num_filters, filter_size, rate, relu=True)
    '''
    conv4_1_relu = conv_layer(conv3_3norm, 512, 3, 2, relu=True)
    conv4_2_relu = conv_layer_dila(conv4_1_relu, 512, 3, 1, relu=True)
    conv4_3_relu = conv_layer_dila(conv4_2_relu, 512, 3, 1, relu=True)
    conv4_3norm = batch_norm(conv4_3_relu, train=True)

    '''
    conv5_1(Stride:1,pad:2 dilation: 2) > relu5_1 > conv5_2(same) > relu5_2 > conv5_3 > relu5_3 > conv5_3_norm
    '''

    conv5_1_relu = conv_layer_dila(conv4_3norm, 512, 3, 2, relu=True)
    conv5_2_relu = conv_layer_dila(conv5_1_relu, 512, 3, 2, relu=True)
    conv5_3_relu = conv_layer_dila(conv5_2_relu, 512, 3, 2, relu=True)
    conv5_3norm = batch_norm(conv5_3_relu, train=True)

    '''
    conv6_1 (Stride:1,pad:2 dilation: 2)> relu6_1 > conv6_2(same) > relu6_2 > conv6_3(same) > relu6_3 > conv6_3_norm
    '''

    conv6_1_relu = conv_layer_dila(conv5_3norm, 512, 3, 2, relu=True)
    conv6_2_relu = conv_layer_dila(conv6_1_relu, 512, 3, 2, relu=True)
    conv6_3_relu = conv_layer_dila(conv6_2_relu, 512, 3, 2, relu=True)
    conv6_3norm = batch_norm(conv6_3_relu, train=True)

    '''
    conv7_1(Stride:1,pad:1 dilation: 1) > relu7_1 > conv7_2 > relu7_2 > conv7_3 > relu7_3 > conv7_3_norm
    '''

    conv7_1_relu = conv_layer_dila(conv6_3norm, 512, 3, 1, relu=True)
    conv7_2_relu = conv_layer_dila(conv7_1_relu, 512, 3, 1, relu=True)
    conv7_3_relu = conv_layer_dila(conv7_2_relu, 512, 3, 1, relu=True)
    conv7_3norm = batch_norm(conv7_3_relu, train=True)

    '''
    conv8_1(256, kernal:4 stride:2 pad:1 dilation:1) > relu8_1 > conv8_2(kernal:3 stride:1) > relu8_2 > conv8_3
    '''
    conv8_1_relu = conv_tranpose_layer(conv7_3norm, 256, 4, 2)
    conv8_2_relu = conv_layer(conv8_1_relu, 256, 3, 1, relu=True)
    conv8_3_relu = conv_layer(conv8_2_relu, 256, 3, 1, relu=True)

    return conv8_3_relu

def ab2lab(image):
    # scale by 2.606: reheat the output distribution
    conv8_313_rh = tf.scalar_mul(2.606, image)

    # convert softmax to probab distribution
    class8_313_rh = tf.nn.softmax(conv8_313_rh)

    # populate cluster centers as 1x1 convolution kernel
    quantized_ab = np.load('./resources/pts_in_hull.npy') # load cluster centers
	filter_8_313 = tf.Variable(np.transpose(quantized, [1,0]), dtype='float32')
    filter_8_313 = tf.reshape(filter_8_313, [1, 1, 313, 2])
    class8_ab = tf.nn.conv2d(class8_313_rh, filter_8_313, strides=[1,1,1,1], padding='VALID')
    

###
def create_model(inputs, targets):
    '''
    VGG16 :
    > conv1_1 > relu1_1 > conv1_2 > relu1_2 > pool1
    ===
    > conv2_1 > relu2_1 > conv2_2 > relu2_2 > pool2
    ===
    > conv3_1 > relu3_1 > conv3_2 > relu3_2 > conv3_3 > relu3_3 > pool3
    ===
    > conv4_1 > relu4_1 > conv4_2 > relu4_2 > conv4_3 > relu4_3 > pool4
    ===
    > conv5_1 > relu5_1 > conv5_2 > relu5_2 > conv5_3 > relu5_3 > pool5
    ===
    > FC-4096 > FC-4096 > FC-100 > soft-max
    '''

    def discriminator(input1, input2):
    	# 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    	input=tf.concat([input1,input2],axis=3)

        # block1 : [batch, 256, 256, 6] => [batch, 128, 128, 64]  
        conv1_1_relu=conv(input,64)
        conv1_2_relu=conv(conv1_1_relu,64)
        pool1=pool(conv1_2_relu)
        	
        # block2 : [batch, 128, 128, 64] => [batch, 64, 64, 128]  
        conv2_1_relu=conv(pool1,128)
        conv2_2_relu=conv(conv2_1_relu,128)
    	pool2=pool(conv2_2_relu)

        # block3 : [batch, 64, 64, 128] => [batch, 32, 32, 256]  
        conv3_1_relu=conv(pool2,256)
        conv3_2_relu=conv(conv3_1_relu,256)
        conv3_3_relu=conv(conv3_2_relu,256)
        pool3=pool(conv3_3_relu)

        # block4 : [batch, 32, 32, 256] => [batch, 16, 16, 512]    
        conv4_1_relu=conv(pool3,512)
        conv4_2_relu=conv(conv4_1_relu,512)
        conv4_3_relu=conv(conv4_2_relu,512)
        pool4=pool(conv4_3_relu)

        #block5 : [batch, 16, 16, 512] => [batch, 8, 8, 512]   
        conv5_1_relu=conv(pool4,512)
        conv5_2_relu=conv(conv5_1_relu,512)
        conv5_3_relu=conv(conv5_2_relu,512)
        pool5=pool(conv5_3_relu)

     #    #block6 :FC layers
    	# fc1=FC_layer(pool5,4096)
    	# fc2=FC_layer(fc1,4096)
    	# fc3=FC_layer(fc2,1000)
            
    	# def FC_layer(input,out_channels):
    	# 	if len(input.get_shape)==4:			
    	#     	shape=int(np.prod(input.get_shape()[1:]))
    	#     else:
    	#     	shape=int(np.prod(input.get_shape()[-1]))
    	#     weight=tf.Variable(tf.truncated_normal([shape,out_channels],dtype=tf.float32,stddev=1e-1))
    	#     bias=tf.Variable(tf.constant(1.0,shape=[out_channels],dtype=tf.float32))
    	#     flat=tf.reshape(input,[-1,shape])
    	#     fc=tf.nn.bias_add(tf.matmul(flat,weight),bias)
    	#     fc=tf.nn.relu(fc)   
    	#     return fc

    	result= tf.sigmoid(pool5)
    	return result

    	def kernel_init(in_channels,out_channels):
    		weights=tf.Variable(tf.truncated_normal([3,3,in_channels,out_channels],dtype=tf.float32,stddev=1e-1))
    		biases=tf.Variable(tf.constant(0.0,shape=[out_channels],dtype=tf.float32))
    		return weights,biases

    	def conv(net,out_channels,relu=True):
    		in_channels=net.get_shape()[3]
    		weights,biases=kernel_init(in_channels,out_channels)
    	    conv=tf.nn.conv2d(net,weights,[1,1,1,1], padding="SAME")
    	    conv=tf.nn.bias_add(conv,biases)	   
    	    if relu:
    	        conv=tf.nn.relu(namedtuple)   
    	    return conv   

    	def pool(net):
            return tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs. they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1, abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets # return {name, step} list

# make html file for comparing input and outputs with ground truth
def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])
        index.write("</tr>")
    return index_path







