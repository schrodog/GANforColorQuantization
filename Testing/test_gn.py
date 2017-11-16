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

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


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

        # print(tf.Session().run(tf.reshape(lab_pixels, tf.shape(srgb))).shape)
        # print((tf.reshape(lab_pixels, tf.shape(srgb))).get_shape())
        # print(lab_pixels.get_shape())
        
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
    quantized_ab = np.load('../resources/pts_in_hull.npy') # load cluster centers
    filter_8_313 = tf.Variable(np.transpose(quantized_ab, [1,0]), dtype=tf.float32)
    filter_8_313 = tf.reshape(filter_8_313, [1, 1, 313, 2])
    class8_ab = tf.nn.conv2d(class8_313_rh, filter_8_313, strides=[1,1,1,1], padding='VALID')
    

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

def main():
    # CROP_SIZE = 256
    sess = tf.InteractiveSession()
    def convert(image, crop_size):
        return tf.image.resize_images(image, size=tf.constant([crop_size, crop_size]), method=tf.image.ResizeMethod.BICUBIC)
        
    # list_images = glob.glob(os.path.join('./imgs', '*.JPEG'))
    image = tf.image.decode_jpeg(tf.read_file('../imgs/ILSVRC2012_val_00041580.JPEG'))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape([380,500,3])
    # print(image.get_shape())
    lab = rgb_to_lab(image)
    # print(tf.shape(lab))
    L_ch, a_ch, b_ch = preprocess_lab(lab)
    
    a_images = tf.expand_dims(L_ch, axis=2)
    b_images = tf.stack([a_ch, b_ch], axis=2)
    inputs, targets = [convert(a_images,256) , convert(b_images,256) ]
    
    # start network
    outputs = ab2lab(net(inputs))
    output2 = np.transpose( (sess.run(outputs))[0,:,:,:], [2,0,1])
    rgb_image = augment(convert(output2, 224), convert(a_images, 224))
    
    plt.imshow()

main()