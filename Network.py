#This CAM-CAN source was made by Ganbayar Batchuluun 
#if you have any question feel free to ask ganabata87@gmail.com


'''
1. ResNet152V2
2. InceptionNet v4, 
3. InceptionResNetV2
4. DenseNet201, 2017 
5. EfficientNetB7, 2019

'''
# Modules
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import sys
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer

def generator(shape,n_class):
    #record gradient between F_k and Y_c
    
    inp = Input(shape = shape)

    model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(inp)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    A_k = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = "same",name='last_conv')(model)
           
    F_k = GlobalAveragePooling2D(name='gap')(A_k)
    #A_k = Flatten()(A_k)
    dense_layer = Dense(n_class,name='last_dense')
    Y_c=dense_layer(F_k)#
    model = Activation('softmax')(Y_c)

    #cam
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

    #hp0 = tf.squeeze(hp0)
    #hp1 = tf.squeeze(hp1)
    #hp2 = tf.squeeze(hp2)
    #hp3 = tf.squeeze(hp3)
    #hp4 = tf.squeeze(hp4)
    #hp5 = tf.squeeze(hp5)
    #hp6 = tf.squeeze(hp6)
    #hp7 = tf.squeeze(hp7)

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

    model = Model(inputs = inp, outputs = [model, hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7])
    optimizer = Adam(learning_rate=0.00001, decay=1e-6)
    model.compile(loss=['categorical_crossentropy', 'mse','mse','mse','mse','mse','mse','mse','mse'], 
        optimizer=optimizer,metrics=['accuracy'])
    return model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
    
def discriminator(image_shape,n_class):

    hp0=Input(shape = image_shape)
    hp1=Input(shape = image_shape)
    hp2=Input(shape = image_shape)
    hp3=Input(shape = image_shape)
    hp4=Input(shape = image_shape)
    hp5=Input(shape = image_shape)
    hp6=Input(shape = image_shape)
    hp7=Input(shape = image_shape)

    inp = concatenate([hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7],axis=-1)
                
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(inp)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(model)

    model = Flatten()(model)
    model = LeakyReLU(alpha = 0.2)(model)
    model = Dense(n_class,activation='softmax')(model)
        
    discriminator_model = Model(inputs = [hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7], outputs = model)
        
    return discriminator_model