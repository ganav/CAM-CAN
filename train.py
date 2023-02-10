#This CAM-CAN source was made by Ganbayar Batchuluun 
#if you have any question feel free to ask ganabata87@gmail.com

import Network, math
import  Utils
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.layers import TimeDistributed
from tqdm import tqdm
import numpy as np
import argparse,csv, time,cv2
from tensorflow.keras.optimizers import Adam,SGD
from keras_apps_ import efficientNet, resnet_common, densenet, inception_resnet_v2,inception_v4

np.random.seed(10)

out_size = 14
image_shape = (200,200,1)
image_shape2 = (out_size,out_size,1)

def XDGan_network(discriminator, generator, optimizer):
    discriminator.trainable = False
    inp=Input(shape = image_shape)

    x,hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7 = generator(inp)

    xdgan_output = discriminator([hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7])

    xdgan = Model(inputs=inp, outputs=xdgan_output)
    xdgan.compile(loss=["categorical_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    return xdgan


#train function without CAM-CAN
def train(epochs, batch_size, output_dir, model_save_dir, ext,n_class):

    #generator = Network.generator(image_shape,n_class)
    #generator = efficientNet.EfficientNet(2.0, 3.1, 600, 0.5,input_shape=image_shape,classes=n_class)#(last_conv output is 14,14)
    generator = resnet_common.ResNet152V2(image_shape,n_class)#(last_conv output is 14,14)
    #generator = densenet.DenseNet201([3, 3, 5, 5],True, image_shape,n_class)#(last_conv output is 12,12)
    #generator = inception_resnet_v2.InceptionResNetV2(True,image_shape,n_class)#(last_conv output is 10,10)
    #generator = inception_v4.create_inception_v4(image_shape,n_class)#(last_conv output is 12,12)
    #sys.exit()
    
    data_csv = []
    img0 = np.zeros((out_size,out_size), dtype=np.uint8)
    img0=img0[np.newaxis,...]

    x_train, y_train,labels2 = Utils.load_training_data('train', ext,n_class,out_size)
    x_test, y_test,labels = Utils.load_training_data('test', ext,n_class,out_size)
            
    batch_count = int(x_train.shape[0] / batch_size)
    batch_count2 = int(x_test.shape[0] / batch_size)

    indx3 = tf.one_hot(n_class, n_class+1)
    indx3 = indx3[np.newaxis,...]
    indx3 = np.array(indx3)

    my_list = [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500]

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        loss = 0
        loss1 = 0
        loss2 = 0

        for num in tqdm(range(batch_count)):

            #one hot
            indx = tf.one_hot(labels2[num], n_class)
            indx = indx[np.newaxis,...]
            indx = np.array(indx)

            prob = generator.train_on_batch(x_train[num],indx)
            loss1 = loss1 + prob[0]
            Utils.plot_generated_images(output_dir, generator, x_test[1], x_test[0],e,num)

        for num in tqdm(range(batch_count2)):
            res = generator.predict(x_test[num])
            r=np.argmax(res)

            if r == labels[num]:
                loss2 = loss2 + 1
                        
        '''res= generator.predict(x_test[0])
        print(np.floor(res*100),' - ',labels[0])
        res = generator.predict(x_test[3])
        print(np.floor(res*100),' - ',labels[3])
        res = generator.predict(x_test[9])
        print(np.floor(res*100),' - ',labels[9])'''

        print('Accuracy: ',loss2/batch_count2)
        data_csv.append([loss2/batch_count2,loss1/batch_count])
        

        if e in my_list:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)

    with open(model_save_dir+'log_file.csv', 'w',newline='') as fout:#save training loss
        writer = csv.writer(fout)
        writer.writerows(map(lambda x: [x], data_csv))

#train function with CAM-CAN
''' 
def train(epochs, batch_size, output_dir, model_save_dir, ext,n_class):

    #generator = Network.generator(image_shape,n_class)
    #generator = efficientNet.EfficientNet(2.0, 3.1, 600, 0.5,input_shape=image_shape,classes=n_class)#(last_conv output is 14,14)
    generator = resnet_common.ResNet152V2(image_shape,n_class)#(last_conv output is 14,14)
    #generator = densenet.DenseNet201([3, 3, 5, 5],True, image_shape,n_class)#(last_conv output is 12,12)
    #generator = inception_resnet_v2.InceptionResNetV2(True,image_shape,n_class)#(last_conv output is 10,10)
    #generator = inception_v4.create_inception_v4(image_shape,n_class)#(last_conv output is 12,12)

    #print('generator :')
    #print(generator.summary())
    discriminator = Network.discriminator(image_shape2,n_class+1)
    #print('discriminator :')
    #print(discriminator.summary())

    #optimizer = SGD(learning_rate=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(learning_rate=0.00001, decay=1e-6)
    discriminator.compile(loss="categorical_crossentropy", optimizer=optimizer)
    
    xdgan = XDGan_network(discriminator,generator, optimizer)
    #print('xdgan :')
    #print(xdgan.summary())
    
    data_csv = []
    img0 = np.zeros((out_size,out_size), dtype=np.uint8)
    img0=img0[np.newaxis,...]

    x_train, y_train,labels2 = Utils.load_training_data('train', ext,n_class,out_size)
    x_test, y_test,labels = Utils.load_training_data('test', ext,n_class,out_size)
            
    batch_count = int(x_train.shape[0] / batch_size)
    batch_count2 = int(x_test.shape[0] / batch_size)

    indx3 = tf.one_hot(n_class, n_class+1)
    indx3 = indx3[np.newaxis,...]
    indx3 = np.array(indx3)

    my_list = [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500]

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        loss = 0
        loss1 = 0
        loss2 = 0

        for num in tqdm(range(batch_count)):

            #one hot
            indx = tf.one_hot(labels2[num], n_class)
            indx = indx[np.newaxis,...]
            indx = np.array(indx)

            indx2 = tf.one_hot(labels2[num], n_class+1)
            indx2 = indx2[np.newaxis,...]
            indx2 = np.array(indx2)

            prob = generator.train_on_batch(x_train[num],indx)
            loss1 = loss1 + prob[0]

            prob, hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7 = generator.predict(x_train[num])
            discriminator.trainable = True
                               
            if  labels2[num] ==0:
                loss_1 = discriminator.train_on_batch([y_train[num],img0,img0,img0,
                    img0,img0,img0,img0], indx2)
            elif  labels2[num] ==1:
                loss_1 = discriminator.train_on_batch([img0,y_train[num],img0,img0,
                    img0,img0,img0,img0], indx2)
            elif  labels2[num] ==2:
                loss_1 = discriminator.train_on_batch([img0,img0,y_train[num],img0,
                    img0,img0,img0,img0], indx2)
            elif  labels2[num] ==3:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,y_train[num],
                    img0,img0,img0,img0], indx2)
            elif  labels2[num] ==4:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,
                    img0,y_train[num],img0,img0,img0], indx2)
            elif  labels2[num] ==5:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,
                    img0,img0,y_train[num],img0,img0], indx2)
            elif  labels2[num] ==6:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,
                    img0,img0,img0,y_train[num],img0], indx2)
            elif  labels2[num] ==7:
                loss_1 = discriminator.train_on_batch([img0,img0,img0,
                    img0,img0,img0,img0,y_train[num]], indx2)
                        
            loss_2 = discriminator.train_on_batch([hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7], indx3)

            loss1 = loss1 + 0.5 * np.add(loss_1, loss_2)

            discriminator.trainable = False
            xdgan_loss = xdgan.train_on_batch(x_train[num],indx2)

        # this part can be removed
        for num in tqdm(range(batch_count2)):
            res, res20,res21,res22,res23,res24,res25,res26,res27 = generator.predict(x_test[num])
            r=np.argmax(res)

            if r == labels[num]:
                loss2 = loss2 + 1
                        
        res, res20,res21,res22,res23,res24,res25,res26,res27 = generator.predict(x_test[0])
        print(np.floor(res*100),' - ',labels[0])
        res, res20,res21,res22,res23,res24,res25,res26,res27 = generator.predict(x_test[20])
        print(np.floor(res*100),' - ',labels[20])
        res, res20,res21,res22,res23,res24,res25,res26,res27 = generator.predict(x_test[35])
        print(np.floor(res*100),' - ',labels[10])
        

        print('Accuracy: ',loss2/batch_count2)
        #print('GAN Loss: ',xdgan_loss)
        #print('Disc Loss: ',loss1/batch_count)
        data_csv.append([loss2/batch_count2,loss1/batch_count,xdgan_loss])
        #data_csv.append([loss2/batch_count2,loss1/batch_count])

        #num1 = np.random.randint(0, x_train.shape[0]-1, size=batch_size)#get random batch_size numbers
        #num2 = np.random.randint(0, x_test.shape[0]-1, size=batch_size)#get random batch_size numbers

        Utils.plot_generated_images(output_dir, generator, x_test[1], x_test[0],e)

        if e in my_list:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
        #discriminator.save(model_save_dir + 'dis_model%d.h5' % e)

    with open(model_save_dir+'log_file.csv', 'w',newline='') as fout:#save training loss
        writer = csv.writer(fout)
        writer.writerows(map(lambda x: [x], data_csv))
'''
if __name__== "__main__":
                 
    batch_size=1
    epochs=500
    model_save_dir='./model/'
    output_dir='./output/'
    ext='.bmp'
    n_class = 8
    train(epochs, batch_size, output_dir, model_save_dir,ext,n_class)