# okay, this is where we write fairly simple gestalt models, with the aim of hopefully being simple
# so let's test a few

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def SimpleConv(weights_path=None, input_shape):
	model = Sequential()
	
	#Encoder
	model.add(ZeroPadding2D((1,1), input_shape=(input_shape)))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(ZeroPadding2D(1,1)))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	#Decoder
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2DTranspose(64, 3,3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2DTranspose(64,3,3,activation='relu'))
	model.add(UpSampling2D((2,2)))

	if weights_path:
		model.load_weights(weights_path)

	return model


