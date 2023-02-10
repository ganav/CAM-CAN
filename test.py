#This CAM-CAN source was made by Ganbayar Batchuluun 
#if you have any question feel free to ask ganabata87@gmail.com


from __future__ import print_function, division
from PIL import Image 
#import scipy
#from scipy.misc import imresize
#from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras as Ker
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
import sys,glob,os, cv2,datetime
import numpy as np
from time import time
import  Utils, mgrad
from tqdm import tqdm

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def gradcam(prob,img,img_path,nam,model, last_conv_layer_name,last_dense_layer_name):
	grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer(last_dense_layer_name).output])

	with tf.GradientTape() as tape0:
		A_k, preds = grad_model(img)
		class_channel0 = preds[:, 0]
	grads0 = tape0.gradient(class_channel0, A_k)

	with tf.GradientTape() as tape1:
		A_k, preds = grad_model(img)
		class_channel1 = preds[:, 1]
	grads1 = tape1.gradient(class_channel1, A_k)

	with tf.GradientTape() as tape2:
		A_k, preds = grad_model(img)
		class_channel2 = preds[:, 2]
	grads2 = tape2.gradient(class_channel2, A_k)

	with tf.GradientTape() as tape3:
		A_k, preds = grad_model(img)
		class_channel3 = preds[:, 3]
	grads3 = tape3.gradient(class_channel3, A_k)

	with tf.GradientTape() as tape4:
		A_k, preds = grad_model(img)
		class_channel4 = preds[:, 4]
	grads4 = tape4.gradient(class_channel4, A_k)

	with tf.GradientTape() as tape5:
		A_k, preds = grad_model(img)
		class_channel5 = preds[:, 5]
	grads5 = tape5.gradient(class_channel5, A_k)

	with tf.GradientTape() as tape6:
		A_k, preds = grad_model(img)
		class_channel6 = preds[:, 6]
	grads6 = tape6.gradient(class_channel6, A_k)

	with tf.GradientTape() as tape7:
		A_k, preds = grad_model(img)
		class_channel7 = preds[:, 7]
	grads7 = tape7.gradient(class_channel7, A_k)

	weights0 = tf.reduce_mean(grads0, axis=(0, 1, 2))
	weights1 = tf.reduce_mean(grads1, axis=(0, 1, 2))
	weights2 = tf.reduce_mean(grads2, axis=(0, 1, 2))
	weights3 = tf.reduce_mean(grads3, axis=(0, 1, 2))
	weights4 = tf.reduce_mean(grads4, axis=(0, 1, 2))
	weights5 = tf.reduce_mean(grads5, axis=(0, 1, 2))
	weights6 = tf.reduce_mean(grads6, axis=(0, 1, 2))
	weights7 = tf.reduce_mean(grads7, axis=(0, 1, 2))

	A_k = A_k[0]

	hp0 = A_k @ weights0[..., tf.newaxis]
	hp1 = A_k @ weights1[..., tf.newaxis]
	hp2 = A_k @ weights2[..., tf.newaxis]
	hp3 = A_k @ weights3[..., tf.newaxis]
	hp4 = A_k @ weights4[..., tf.newaxis]
	hp5 = A_k @ weights5[..., tf.newaxis]
	hp6 = A_k @ weights6[..., tf.newaxis]
	hp7 = A_k @ weights7[..., tf.newaxis]

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

	hp0=conv(hp0.numpy(),img)
	hp1=conv(hp1.numpy(),img)
	hp2=conv(hp2.numpy(),img)
	hp3=conv(hp3.numpy(),img)
	hp4=conv(hp4.numpy(),img)
	hp5=conv(hp5.numpy(),img)
	hp6=conv(hp6.numpy(),img)
	hp7=conv(hp7.numpy(),img)

	# Save the superimposed image
	hp0.save(img_path+'0'+nam)
	hp1.save(img_path+'1'+nam)
	hp2.save(img_path+'2'+nam)
	hp3.save(img_path+'3'+nam)
	hp4.save(img_path+'4'+nam)
	hp5.save(img_path+'5'+nam)
	hp6.save(img_path+'6'+nam)
	hp7.save(img_path+'7'+nam)

def cam(prob,img,img_path,nam,model, last_conv_layer_name,last_dense_layer_name):
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
	hp8 = A_k_ @ weights[:, 8][..., tf.newaxis]
	hp9 = A_k_ @ weights[:, 9][..., tf.newaxis]

	hp0 = tf.squeeze(hp0)
	hp1 = tf.squeeze(hp1)
	hp2 = tf.squeeze(hp2)
	hp3 = tf.squeeze(hp3)
	hp4 = tf.squeeze(hp4)
	hp5 = tf.squeeze(hp5)
	hp6 = tf.squeeze(hp6)
	hp7 = tf.squeeze(hp7)
	hp8 = tf.squeeze(hp8)
	hp9 = tf.squeeze(hp9)

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

	hp0=conv(hp0,img)
	hp1=conv(hp1,img)
	hp2=conv(hp2,img)
	hp3=conv(hp3,img)
	hp4=conv(hp4,img)
	hp5=conv(hp5,img)
	hp6=conv(hp6,img)
	hp7=conv(hp7,img)
	hp8=conv(hp8,img)
	hp9=conv(hp9,img)
	

	# Save the superimposed image
	#hp0.save(img_path+'0'+nam)
	#hp1.save(img_path+'1'+nam)
	#hp2.save(img_path+'2'+nam)
	#hp3.save(img_path+'3'+nam)
	#hp4.save(img_path+'4'+nam)
	#hp5.save(img_path+'5'+nam)
	#hp6.save(img_path+'6'+nam)
	#hp7.save(img_path+'7'+nam)
	figsize=(30, 10)
	dim=(1, 10)
	plt.figure(figsize=figsize)
	plt.subplot(dim[0], dim[1], 1)
	plt.imshow(hp0, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 1 : %f' % prob[0][0])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 2)
	plt.imshow(hp1, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 2 : %f' % prob[0][1])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 3)
	plt.imshow(hp2, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 3 : %f' % prob[0][2])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 4)
	plt.imshow(hp3, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 4 : %f' % prob[0][3])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 5)
	plt.imshow(hp4, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 5 : %f' % prob[0][4])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 6)
	plt.imshow(hp5, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 6 : %f' % prob[0][5])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 7)
	plt.imshow(hp6, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 7 : %f' % prob[0][6])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 8)
	plt.imshow(hp7, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 8 : %f' % prob[0][7])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 9)
	plt.imshow(hp8, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 9 : %f' % prob[0][8])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 10)
	plt.imshow(hp9, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 10 : %f' % prob[0][9])
	plt.axis('off')

	plt.tight_layout()
	plt.savefig('./0cam/image_'+nam)
	plt.close('all')

def sq(hp0,img):
	img = Utils.denormalize(img)
	img=np.squeeze(img, axis=-1)#remove previously added dimension
	img=np.squeeze(img, axis=0)#remove previously added dimension
	#img=np.squeeze(img, axis=0)#remove previously added dimension
	
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

def cam2(hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9,prob,img,img_path,nam,model, last_conv_layer_name,last_dense_layer_name):
	'''grad_model = tf.keras.models.Model([model.inputs], model.get_layer(last_conv_layer_name).output)

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
	hp7 = A_k_ @ weights[:, 7][..., tf.newaxis]'''

	hp0 = tf.squeeze(hp0)
	hp1 = tf.squeeze(hp1)
	hp2 = tf.squeeze(hp2)
	hp3 = tf.squeeze(hp3)
	hp4 = tf.squeeze(hp4)
	hp5 = tf.squeeze(hp5)
	hp6 = tf.squeeze(hp6)
	hp7 = tf.squeeze(hp7)
	hp8 = tf.squeeze(hp8)
	hp9 = tf.squeeze(hp9)

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


	hp0=conv(hp0,img)
	hp1=conv(hp1,img)
	hp2=conv(hp2,img)
	hp3=conv(hp3,img)
	hp4=conv(hp4,img)
	hp5=conv(hp5,img)
	hp6=conv(hp6,img)
	hp7=conv(hp7,img)
	hp8=conv(hp8,img)
	hp9=conv(hp9,img)

	'''hp0=sq(hp0,img)
	hp1=sq(hp1,img)
	hp2=sq(hp2,img)
	hp3=sq(hp3,img)
	hp4=sq(hp4,img)
	hp5=sq(hp5,img)
	hp6=sq(hp6,img)
	hp7=sq(hp7,img)'''

	

	# Save the superimposed image
	#hp0.save(img_path+'0'+nam)
	#hp1.save(img_path+'1'+nam)
	#hp2.save(img_path+'2'+nam)
	#hp3.save(img_path+'3'+nam)
	#hp4.save(img_path+'4'+nam)
	#hp5.save(img_path+'5'+nam)
	#hp6.save(img_path+'6'+nam)
	#hp7.save(img_path+'7'+nam)
	figsize=(30, 10)
	dim=(1, 10)
	plt.figure(figsize=figsize)
	plt.subplot(dim[0], dim[1], 1)
	plt.imshow(hp0, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 1 : %f' % prob[0][0])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 2)
	plt.imshow(hp1, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 2 : %f' % prob[0][1])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 3)
	plt.imshow(hp2, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 3 : %f' % prob[0][2])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 4)
	plt.imshow(hp3, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 4 : %f' % prob[0][3])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 5)
	plt.imshow(hp4, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 5 : %f' % prob[0][4])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 6)
	plt.imshow(hp5, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 6 : %f' % prob[0][5])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 7)
	plt.imshow(hp6, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 7 : %f' % prob[0][6])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 8)
	plt.imshow(hp7, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 8 : %f' % prob[0][7])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 9)
	plt.imshow(hp8, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 9 : %f' % prob[0][8])
	plt.axis('off')
	plt.subplot(dim[0], dim[1], 10)
	plt.imshow(hp9, cmap='gray', vmin=0, vmax=255)
	plt.title('Class 10 : %f' % prob[0][9])
	plt.axis('off')

	plt.tight_layout()
	plt.savefig('./0cam/image_'+nam)
	plt.close('all')

n_class = 10
#print(model.summary())
input_dir='test\\'
last_conv_layer_name = 'last_conv'
last_dense_layer_name = 'last_dense'
for k in range(1):
#DenseNet201 80 EfficientNet 200 inception_v4 10 InceptionResNetV2 30 ResNet152V2 90
	na = 'DenseNet201 cam'
	model = load_model('./comparison 500 db2/epoch 2/'+na+' 2/gen_model80.h5')#model - Copy
	#print(model.summary())
	# Remove last layer's softmax
	#model.layers[-1].activation = None

	for i in range(1):
		#x_train, y_train= Utils.load_training_data(input_dir,'3\\', 'png')
		x_test, y_test, labels= Utils.load_training_data('test', '.png',n_class,12)
		batch_count = x_test.shape[0]

		for j in tqdm(range(batch_count)):

			#out=model.predict(x_test[j])
			out,hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9=model.predict(x_test[j])
			#print(np.argmax(out))

			# Generate class activation heatmap
			#gradcam(out,x_test[j],"./0cam/img4_",str(j)+".png",model, last_conv_layer_name,last_dense_layer_name)
			cam2(hp0,hp1,hp2,hp3,hp4,hp5,hp6,hp7,hp8,hp9,out,x_test[j],"./0cam/img2_",str(j)+'_'+na+".png",model, last_conv_layer_name,last_dense_layer_name)
			
			#cam(out,x_test[j],"./0cam/img2_",str(j)+'_'+na+".png",model, last_conv_layer_name,last_dense_layer_name)
			#sys.exit()

			#post processing
			#x=Utils.denormalize(x)
			#x=np.squeeze(x, axis=0)#remove previously added dimension

			#save images
			#path='./data/'+input_dir+'result'+str(k)+'\\'
			#os.makedirs(path, exist_ok=True)
			#cv2.imwrite(path+'img_'+str(i)+'_'+str(j)+'.png',x)

print('end')






