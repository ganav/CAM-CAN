"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

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



def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    #x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  #name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def transition_block2(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    #x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  #name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    return x

def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    #x1 = layers.BatchNormalization(axis=bn_axis,
                                   #epsilon=1.001e-5,
                                   #name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    #x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   #name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet201(blocks,
             include_top=True,
             input_shape=None,
             classes=1000):




    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    #x = layers.BatchNormalization(
        #axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block2(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    #x = layers.BatchNormalization(
        #axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
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
    hp8 = A_k_ @ weights[:, 8][..., tf.newaxis]
    hp9 = A_k_ @ weights[:, 9][..., tf.newaxis]
    hp0 = tf.maximum(hp0, 0) / tf.math.reduce_max(hp0)
    hp1 = tf.maximum(hp1, 0) / tf.math.reduce_max(hp1)
    hp2 = tf.maximum(hp2, 0) / tf.math.reduce_max(hp2)
    hp3 = tf.maximum(hp3, 0) / tf.math.reduce_max(hp3)
    hp4 = tf.maximum(hp4, 0) / tf.math.reduce_max(hp4)
    hp5 = tf.maximum(hp5, 0) / tf.math.reduce_max(hp5)
    hp6 = tf.maximum(hp6, 0) / tf.math.reduce_max(hp6)
    hp7 = tf.maximum(hp7, 0) / tf.math.reduce_max(hp7) 
    hp8 = tf.maximum(hp8, 0) / tf.math.reduce_max(hp8)
    hp9 = tf.maximum(hp9, 0) / tf.math.reduce_max(hp9) 
    hp0 = tf.stack(hp0[tf.newaxis,...])
    hp1 = tf.stack(hp1[tf.newaxis,...])
    hp2 = tf.stack(hp2[tf.newaxis,...])
    hp3 = tf.stack(hp3[tf.newaxis,...])
    hp4 = tf.stack(hp4[tf.newaxis,...])
    hp5 = tf.stack(hp5[tf.newaxis,...])
    hp6 = tf.stack(hp6[tf.newaxis,...])
    hp7 = tf.stack(hp7[tf.newaxis,...])
    hp8 = tf.stack(hp8[tf.newaxis,...])
    hp9 = tf.stack(hp9[tf.newaxis,...])

    model = models.Model(inputs = img_input, outputs = [x, hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9])
    model.compile(loss=['categorical_crossentropy', 'mse','mse','mse','mse','mse','mse','mse','mse','mse','mse'], 
        optimizer=optimizer,metrics=['accuracy'])
    
    '''
    model = models.Model(img_input, x, name='densenet201')
    model.compile(loss=['categorical_crossentropy'], optimizer=optimizer,metrics=['accuracy'])
    
    print(model.summary())

    return model
