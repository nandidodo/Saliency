# okay, this is where we write fairly simple gestalt models, with the aim of hopefully being simple
# so let's test a few

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


def SimpleContinuingConvModel(input_shape):
	#this is a bunch of convolutional models which try to keep the miage of the same shape
	# in the line of the 1x1 convolutions all convolutional networks
	#I'm seeing how powerful the convolutional image transformers are
	#I was thinking this could map from our error maps to their given sal maps
	#for actual apples to apples comparison of ROC curves and so forth of our method to theirs
	input_img = Input(input_shape)
	x = Conv2D(32,2,strides=(1,1), activation='relu',padding='same')(input_img)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2D(64,2,strides=(1,1), activation='relu',padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2D(64,1,strides=(1,1), activation='relu',padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2D(32,1,strides=(1,1), activation='relu',padding='same')(x)
	x = BatchNormalization(momentum=0.9)(x)
	print x.shape
	x = Conv2D(1,2,strides=(1,1), activation='relu',padding='valid')(x)
	print x.shape

	model = Model(input_img, x)
	return model

if __name__ =='__main__':
	"""
	(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
	xtrain = xtrain[:,:,:,0]
	xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2],1))
	shape = xtrain.shape
	input_shape = (shape[1], shape[2], shape[3])
	model = SimpleAutoencoder(input_shape)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer = sgd, loss='mse')
	model.fit(xtrain, xtrain, batch_size=64, epochs=5)
	"""
	(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
	xtrain = xtrain[:,:,:,0].astype('float32')/255.
	xtest = xtest[:,:,:,0].astype('float32')/255.
	xtrain = np.reshape(xtrain, (len(xtrain), 32,32,1))
	xtest = np.reshape(xtest, (len(xtest), 32,32,1))
	print xtrain.shape
	#model = SimpleAutoencoder((28,28,1))
	model=SimpleConvDropoutBatchNorm((32,32,1))
	model.compile(optimizer='sgd', loss='mse')


	model.fit(xtrain, xtrain, nb_epoch=5, batch_size=128, shuffle=True, validation_data=(xtest, xtest), verbose=1, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])









