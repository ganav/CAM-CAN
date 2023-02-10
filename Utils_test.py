#This CAM-CAN source was made by Ganbayar Batchuluun 
#if you have any question feel free to ask ganabata87@gmail.com

import tensorflow as tf
#from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
#from scipy.misc import imresize
from scipy import ndimage
import os,cv2,csv,sys, glob,math
import matplotlib.pyplot as plt
import tensorflow.keras as Ker
from tensorflow.keras.preprocessing.image import img_to_array
# Display
from IPython.display import Image, display
import matplotlib.cm as cm


plt.switch_backend('agg')

def get_img_parts(img_path):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')

    leng=len(parts)
    filename = parts[leng-1]
    path=parts[0]+'\\'
 
    for i in range(1,leng-3):
        path=path+parts[i]+'\\'
    return path+'b\\'+parts[4]+'\\',  filename

def normalize(input_data):
    return (input_data / 255.).astype(np.float32)#(input_data.astype(np.float32) - 127.5)/127.5

    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
    
def load_data_from_dirs(dirs):
    files1,files2,files3 = [],[],[]

    with open('.\\data\\'+dirs+'\\a\\'+'list.csv', 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    #aaa=['wave', 'punch', 'kick','lie down', 'walk', 'run', 'stand','sit']
    for d in range(len(data)):
        imgs_A = cv2.imread('.\\data\\'+dirs+'\\a\\'+data[d][0]+'\\'+data[d][1],0)
        imgs_B = cv2.imread('.\\data\\'+dirs+'\\b\\'+data[d][0]+'\\'+data[d][1],0)
        #cv2.imshow('a',imgs_A)
        #cv2.imshow('b',imgs_B)
            
        #print(aaa[int(data[d][0])-1])
        #cv2.waitKey(0)
        imgs_B = cv2.resize(imgs_B, (12, 12), interpolation = cv2.INTER_AREA)
        #img0 = np.zeros((12,12,1), dtype=np.uint8)

        #create black 8 channel image
        #imgs_C = img0
        #for i in range(n_class-1):
            #imgs_C = np.dstack((imgs_C,img0))

        #cv2.imshow('4',imgs_C[:,:,fold-1])
        #imgs_C[:,:,fold-1] = imgs_B# assign GT to corresponding class channel
            
            
        #sys.exit()
        #print(imgs_A.shape)

        #comment below 4 lines in case of color input
        imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
        imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
        imgs_A = np.moveaxis(imgs_A, 0, -1)#reorder the shape of dimension

        imgs_B = imgs_B[np.newaxis,...]#add 1 more dimension
        imgs_B = imgs_B[np.newaxis,...]#add 1 more dimension
        imgs_B = np.moveaxis(imgs_B, 0, -1)#reorder the shape of dimension


        files1.append(imgs_A)
        files2.append(imgs_B)
        files3.append(int(data[d][0])-1)#minus 1 coz class starts from 1 but in python array it starts from 0

    files2 = normalize(array(files2))
    files1 = normalize(array(files1))

    #print("loaded data:")
    print(len(files1))
    #print(len(files2))
    print(len(files3))

    return files1,files2,files3     
    
def load_training_data(directory):
    x_train,y_train,labels = load_data_from_dirs(directory)
    return x_train,y_train,labels

# While training save generated image(in form LR, SR, HR)
# Save only one image as sample  
def sq(hp0,img):
    
    hp0=np.squeeze(hp0, axis=-1)#remove previously added dimension
    hp0=np.squeeze(hp0, axis=0)#remove previously added dimension
    heatmap = np.uint8(255 * hp0)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    img = np.uint8(255 *img)
    img2=np.dstack((img,img))
    img3=np.dstack((img2,img))
    jet_heatmap = Ker.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = Ker.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap*0.4 + img3
    hp0 = Ker.preprocessing.image.array_to_img(superimposed_img)
    #out=np.squeeze(out, axis=0)#remove previously added dimension
    return hp0

def conv(hp0,img):
    heatmap = np.uint8(255 * hp0)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    img=np.squeeze(img, axis=0)#remove previously added dimension
    img=np.squeeze(img, axis=-1)#remove previously added dimension
    img = np.uint8(255 *img)
    img2=np.dstack((img,img))
    img3=np.dstack((img2,img))
    jet_heatmap = Ker.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = Ker.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap*0.4 + img3
    hp0 = Ker.preprocessing.image.array_to_img(superimposed_img)

    return hp0

def cam(img,model, last_conv_layer_name,last_dense_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], model.get_layer(last_conv_layer_name).output)

    A_k = grad_model(img)
    Y_c = model.get_layer(last_dense_layer_name)
    weights = Y_c.get_weights()[0]
    A_k_ = A_k[0]

    hp0 = A_k_ @ weights[:, 0][..., tf.newaxis]
    hp1 = A_k_ @ weights[:, 1][..., tf.newaxis]
    hp2 = A_k_ @ weights[:, 2][..., tf.newaxis]
    hp3 = A_k_ @ weights[:, 3][..., tf.newaxis]
    hp4 = A_k_ @ weights[:, 4][..., tf.newaxis]
    hp5 = A_k_ @ weights[:, 5][..., tf.newaxis]
    hp6 = A_k_ @ weights[:, 6][..., tf.newaxis]
    hp7 = A_k_ @ weights[:, 7][..., tf.newaxis]

    hp0 = tf.squeeze(hp0)
    hp1 = tf.squeeze(hp1)
    hp2 = tf.squeeze(hp2)
    hp3 = tf.squeeze(hp3)
    hp4 = tf.squeeze(hp4)
    hp5 = tf.squeeze(hp5)
    hp6 = tf.squeeze(hp6)
    hp7 = tf.squeeze(hp7)

    hp0 = tf.maximum(hp0, 0) / tf.math.reduce_max(hp0)
    hp1 = tf.maximum(hp1, 0) / tf.math.reduce_max(hp1)
    hp2 = tf.maximum(hp2, 0) / tf.math.reduce_max(hp2)
    hp3 = tf.maximum(hp3, 0) / tf.math.reduce_max(hp3)
    hp4 = tf.maximum(hp4, 0) / tf.math.reduce_max(hp4)
    hp5 = tf.maximum(hp5, 0) / tf.math.reduce_max(hp5)
    hp6 = tf.maximum(hp6, 0) / tf.math.reduce_max(hp6)
    hp7 = tf.maximum(hp7, 0) / tf.math.reduce_max(hp7)

    hp0=conv(hp0,img)
    hp1=conv(hp1,img)
    hp2=conv(hp2,img)
    hp3=conv(hp3,img)
    hp4=conv(hp4,img)
    hp5=conv(hp5,img)
    hp6=conv(hp6,img)
    hp7=conv(hp7,img)

    return hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7

def plot_generated_images(output_dir, generator, x_train, x_test,e, dim=(2, 9), figsize=(15, 5)):
    #train data
    input_img = denormalize(x_train)
    input_img=np.squeeze(input_img, axis=-1)#remove previously added dimension
    input_img=np.squeeze(input_img, axis=0)#remove previously added dimension
    input_img=np.squeeze(input_img, axis=0)#remove previously added dimension

    inp = np.squeeze(x_train, axis=0)
    prob = generator.predict(inp)#generated_image = denormalize(gen_img)
    hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7 = cam(inp,generator,'last_conv','last_dense')
    r=np.argmax(prob)

    #test data
    input_img2 = denormalize(x_test)
    input_img2=np.squeeze(input_img2, axis=-1)#remove previously added dimension
    input_img2=np.squeeze(input_img2, axis=0)#remove previously added dimension
    input_img2=np.squeeze(input_img2, axis=0)#remove previously added dimension

    inp2 = np.squeeze(x_test, axis=0)
    prob2 = generator.predict(inp2)

    hp0_,hp1_,hp2_,hp3_,hp4_,hp5_,hp6_,hp7_ = cam(inp2,generator,'last_conv','last_dense')
    r2=np.argmax(prob2)

    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(input_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Train image, class: %d' % r)
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(hp0, cmap='gray', vmin=0, vmax=255)
    plt.title('class0 : %f' % prob[0][0])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(hp1, cmap='gray', vmin=0, vmax=255)
    plt.title('class1 : %f' % prob[0][1])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 4)
    plt.imshow(hp2, cmap='gray', vmin=0, vmax=255)
    plt.title('class2 : %f' % prob[0][2])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 5)
    plt.imshow(hp3, cmap='gray', vmin=0, vmax=255)
    plt.title('class3 : %f' % prob[0][3])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 6)
    plt.imshow(hp4, cmap='gray', vmin=0, vmax=255)
    plt.title('class4 : %f' % prob[0][4])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 7)
    plt.imshow(hp5, cmap='gray', vmin=0, vmax=255)
    plt.title('class5 : %f' % prob[0][5])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 8)
    plt.imshow(hp6, cmap='gray', vmin=0, vmax=255)
    plt.title('class6 : %f' % prob[0][6])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 9)
    plt.imshow(hp7, cmap='gray', vmin=0, vmax=255)
    plt.title('class7 : %f' % prob[0][7])
    plt.axis('off')


    plt.subplot(dim[0], dim[1], 10)
    plt.imshow(input_img2, cmap='gray', vmin=0, vmax=255)
    plt.title('Test image, class: %d' % r2)
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 11)
    plt.imshow(hp0_, cmap='gray', vmin=0, vmax=255)
    plt.title('class0 : %f' % prob2[0][0])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 12)
    plt.imshow(hp1_, cmap='gray', vmin=0, vmax=255)
    plt.title('class1 : %f' % prob2[0][1])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 13)
    plt.imshow(hp2_, cmap='gray', vmin=0, vmax=255)
    plt.title('class2 : %f' % prob2[0][2])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 14)
    plt.imshow(hp3_, cmap='gray', vmin=0, vmax=255)
    plt.title('class3 : %f' % prob2[0][3])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 15)
    plt.imshow(hp4_, cmap='gray', vmin=0, vmax=255)
    plt.title('class4 : %f' % prob2[0][4])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 16)
    plt.imshow(hp5_, cmap='gray', vmin=0, vmax=255)
    plt.title('class5 : %f' % prob2[0][5])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 17)
    plt.imshow(hp6_, cmap='gray', vmin=0, vmax=255)
    plt.title('class6 : %f' % prob2[0][6])
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 18)
    plt.imshow(hp7_, cmap='gray', vmin=0, vmax=255)
    plt.title('class7 : %f' % prob2[0][7])
    plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir +'image_%d.png' % e)
    plt.close('all')