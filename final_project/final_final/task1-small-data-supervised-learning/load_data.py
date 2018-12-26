import numpy as np
from scipy.misc import imread, imsave
import time
import os
import sys


def load_image(image_dir):


	h=28
	w=28    
	

	all_image_filenames = os.listdir(image_dir)
	# all_image_filenames = sorted([f for f in all_image_filenames ])
	def getint(name):
	    num, png = name.split('.')
	    return int(num)
	all_image_filenames.sort(key = getint)
	

	total_num_img = len(all_image_filenames)

	all_data = np.zeros((total_num_img, h, w), dtype=np.uint8)

	for idx, all_image_filenames in enumerate(all_image_filenames):
	    img = imread('{}/{}'.format(image_dir, all_image_filenames))
	    all_data[idx] = img[:h, :w]

	return all_data



# data_dir = 'Fashion_MNIST_student/train'


def load_data(data_dir):

	h=28
	w=28    
	n_chan= 1

	all_labels = os.listdir(data_dir)
	all_labels = sorted([f for f in all_labels ])

	total_num_labels = len(all_labels)

	X_train =  np.zeros((2000, h, w), dtype=np.uint8)
	

	# image_dir = 'Fashion_MNIST_student/train/class_0'
	for i in range(total_num_labels):
		image_dir = data_dir+'/'+all_labels[i]
		X_part = load_image(image_dir)
		X_train[i*200:(i+1)*200] = X_part
		


	X_train = X_train.reshape(2000,h,w,n_chan)
	X_train = X_train.astype('float32')
	X_train /= 127.5
	X_train-=1

	return X_train

def load_labels():
	Y_train = np.zeros(2000)
	for i in range (10):
		Y_train[200*i:200*(i+1)] = np.ones(200)*i

	return Y_train


def load_labels_split():
	Y_train = np.zeros(1800)
	for i in range (10):
		Y_train[180*i:180*(i+1)] = np.ones(180)*i

	Y_val = np.zeros(200)
	for i in range (10):
		Y_val[20*i:20*(i+1)] = np.ones(20)*i

	
	return Y_train,Y_val



def load_data_split(data_dir):

	h=28
	w=28    
	n_chan= 1

	all_labels = os.listdir(data_dir)
	all_labels = sorted([f for f in all_labels ])

	total_num_labels = len(all_labels)

	# X_train =  np.zeros((2000, h, w), dtype=np.uint8)
	X_train =  np.zeros((1800, h, w), dtype=np.uint8)

	X_val =  np.zeros((200, h, w), dtype=np.uint8)


	# image_dir = 'Fashion_MNIST_student/train/class_0'
	for i in range(total_num_labels):
		image_dir = data_dir+'/'+all_labels[i]
		X_part = load_image(image_dir)
		# X_train[i*200:(i+1)*200] = X_part
		X_train[i*180:(i+1)*180] = X_part[:180]
		X_val[i*20:(i+1)*20] = X_part[180:]



	# X_train = X_train.reshape(2000,h,w,n_chan)
	X_train = X_train.reshape(1800,h,w,n_chan)
	X_train = X_train.astype('float32')
	X_train /= 127.5
	X_train-=1

	X_val = X_val.reshape(200,h,w,n_chan)
	X_val = X_val.astype('float32')
	X_val /= 127.5
	X_val-=1


	return X_train, X_val






