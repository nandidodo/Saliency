import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import SGD
from keras.datasets import cifar10, mnist
from keras import backend as K
from keras.callbacks import TensorBoard

def SimpleConv(input_shape,weights_path=None):
	input_img = Input(input_shape)
	
	#Encoder
	x = Conv2D(64,2,strides=(1,1),activation='relu',padding='same')(input_img)
	print x.shape
	x = Conv2D(64,2,strides=(2,2), activation='relu', padding='same')(x)
	print x.shape
	x = Conv2D(32,1,strides=(1,1), activation='relu', padding='same')(x)
	print x.shape
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	print x.shape

	print "ENCODER COMPLETE"

	x = Conv2DTranspose(64,2,strides=(2,2), activation='relu',padding='same')(x)
	print x.shape
	x = Conv2DTranspose(32,1,strides=(1,1),activation='relu', padding='same')(x)
	print x.shape
	x = Conv2DTranspose(1, 2,strides=(2,2),activation='relu',padding='valid')(x)
	print x.shape

	#Decoder
	#x = ZeroPadding2D((1,1))(x)
	#print x.shape
	#x = Conv2DTranspose(64, 3,strides=(1,1), activation='relu', padding='same')(x)
	#print x.shape
	#x = ZeroPadding2D((1,1))(x)
	#print x.shape
	#x = Conv2DTranspose(64,2,strides=(2,2),activation='relu')(x)
	#print x.shape
	#x = Conv2DTranspose(1,2,strides=(1,1),activation='relu', padding='valid')(x)
	#print x.shape
	#x = UpSampling2D((2,2))(x)
	#print x.shape

	#if weights_path:
	#	model.load_weights(weights_path)

	model = Model(input_img, x)
	return model


def SimpleConvDropoutBatchNorm(input_shape):
	
	input_img = Input(input_shape)
	#Encoder
	x = Conv2D(64,2,strides=(1,1),activation='relu',padding='same')(input_img)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2D(64,2,strides=(2,2), activation='relu', padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2D(32,1,strides=(1,1), activation='relu', padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Dropout(0.1)(x)
	print x.shape
	print "ENCODER COMPLETE"

	#DECODER
	x = Conv2DTranspose(64,2,strides=(2,2), activation='relu',padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2DTranspose(32,1,strides=(1,1),activation='relu', padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2DTranspose(1, 2,strides=(2,2),activation='relu',padding='valid')(x)
	print x.shape
	#I don't think we normalize final thing for obvious reasons
	print "DECODER COMPLETE"

	model = Model(input_img, x)
	return model

def SimpleSequentialModel(input_shape):
	#this is just a test of a sequential version of the conv net batch norm
	#hopefully it will work?
	model=Sequential()
	model.add(Conv2D(64,2,strides=(1,1),activation='relu',padding='same', input_shape=input_shape))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Conv2D(64,2,strides=(2,2), activation='relu', padding='same'))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Conv2D(32,1,strides=(1,1), activation='relu', padding='same'))
	model.add(BatchNormalization(momentum=0.9))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Dropout(0.1))
	print "ENCODER COMPLETE"

	#DECODER
	model.add(Conv2DTranspose(64,2,strides=(2,2), activation='relu',padding='same'))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Conv2DTranspose(32,1,strides=(1,1),activation='relu', padding='same'))
	model.add(BatchNormalization(momentum=0.9))
	model.add(Conv2DTranspose(1, 2,strides=(2,2),activation='relu',padding='valid'))
	print "DECODER COMPLETE"
	model.summary()
	return model
	

def SimpleAutoencoder(input_shape, weights_path=None):
	
	input_img = Input(input_shape)

	#encoder	
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)

	print "shape of encoded", K.int_shape(encoded)

	#decoder
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)

	# In original tutorial, border_mode='same' was used. 
	# then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
	#x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) 
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x) 

	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
	print "shape of decoded", K.int_shape(decoded)
	
	#compile model
	model = Model(input_img, decoded)
	return model

