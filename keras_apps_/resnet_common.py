"""ResNet, ResNetV2, and ResNeXt models for Keras.

# Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks]
  (https://arxiv.org/abs/1603.05027) (ECCV 2016)
- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
  (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import sys
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from tensorflow.keras import backend
import tensorflow.python.keras.layers


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    #preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       #name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(x)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    #x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  #name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    #x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  #name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
    x = layers.Conv2D(filters, 1, use_bias=False, trainable=False,
                      kernel_initializer={'class_name': 'Constant',
                                          'config': {'value': kernel}},
                      name=name + '_2_gconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D((64 // groups) * filters, 1,
                      use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False,
                   name=name + '_block' + str(i))
    return x


def resnet(stack_fn,
           use_bias,
           model_name='resnet',
           include_top=True,
           input_shape=None,
           classes=8):


    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    #x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='post_bn')(x)
    A_k = layers.Activation('relu', name='last_conv')(x)

    F_k = layers.GlobalAveragePooling2D(name='avg_pool')(A_k)
    dense_layer = layers.Dense(classes, name='last_dense')

    Y_c=dense_layer(F_k)#
    x = Activation('softmax')(Y_c)
    optimizer = Adam(learning_rate=0.00001, decay=1e-6)
  
    #cam
    '''
    weights = dense_layer.get_weights()[0]#[:, 0]
    A_k_ = A_k[0]
    hp0 = A_k_ @ weights[:, 0][..., tf.newaxis]
    hp1 = A_k_ @ weights[:, 1][..., tf.newaxis]
    hp2 = A_k_ @ weights[:, 2][..., tf.newaxis]
    hp3 = A_k_ @ weights[:, 3][..., tf.newaxis]
    hp4 = A_k_ @ weights[:, 4][..., tf.newaxis]
    hp5 = A_k_ @ weights[:, 5][..., tf.newaxis]
    hp6 = A_k_ @ weights[:, 6][..., tf.newaxis]
    hp7 = A_k_ @ weights[:, 7][..., tf.newaxis]

    hp0 = tf.maximum(hp0, 0) / tf.math.reduce_max(hp0)
    hp1 = tf.maximum(hp1, 0) / tf.math.reduce_max(hp1)
    hp2 = tf.maximum(hp2, 0) / tf.math.reduce_max(hp2)
    hp3 = tf.maximum(hp3, 0) / tf.math.reduce_max(hp3)
    hp4 = tf.maximum(hp4, 0) / tf.math.reduce_max(hp4)
    hp5 = tf.maximum(hp5, 0) / tf.math.reduce_max(hp5)
    hp6 = tf.maximum(hp6, 0) / tf.math.reduce_max(hp6)
    hp7 = tf.maximum(hp7, 0) / tf.math.reduce_max(hp7)

    hp0 = tf.stack(hp0[tf.newaxis,...])
    hp1 = tf.stack(hp1[tf.newaxis,...])
    hp2 = tf.stack(hp2[tf.newaxis,...])
    hp3 = tf.stack(hp3[tf.newaxis,...])
    hp4 = tf.stack(hp4[tf.newaxis,...])
    hp5 = tf.stack(hp5[tf.newaxis,...])
    hp6 = tf.stack(hp6[tf.newaxis,...])
    hp7 = tf.stack(hp7[tf.newaxis,...])

    model = models.Model(inputs = img_input, outputs = [x, hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7])
    model.compile(loss=['categorical_crossentropy', 'mse','mse','mse','mse','mse','mse','mse','mse'], 
        optimizer=optimizer,metrics=['accuracy'])
    '''
    # Create model
    model = models.Model(img_input, x, name=model_name)
    model.compile(loss=['categorical_crossentropy'], optimizer=optimizer,metrics=['accuracy'])
 
    print(model.summary())
    return model

def ResNet152V2(input_shape,classes):

    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 3, name='conv3')
        x = stack2(x, 256, 3, name='conv4')
        x = UpSampling2D(size=2)(x)
        x = stack2(x, 256, 3, stride1=1, name='conv5')
        return x

    return resnet(stack_fn, True, 'resnet152v2',True, input_shape,classes)

