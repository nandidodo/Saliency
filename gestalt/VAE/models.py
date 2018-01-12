# This is where I put my keras model definitions for the autoencoders and stuff

import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from losses import *

# we define model constants here - think of a better plce later
filters = 64
kernel = (3,3)
batch_size=64
latent_dim = 2
intermediate_dim = 124
epsilon_std = 1.0
activation ='relu'
optimizer = 'sgd'


#output shapes - this is going to be the tricky bit to get working
intermediate_output_shape = (batch_size, 50, 10, filters)
output_shape = (batch_size, 101, 21, filters)


def reparametrised_sample(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
	return z_mean + K.exp(z_log_var) * epsilon

def DCVAE(input_shape, weights_path=None): # Deep convolutoinal VAE
	width, height,channels = input_shape

	x_input = Input(shape=input_shape)
	#convolutional encoder model
	x = Conv2D(channels, kernel_size=(2,2), padding='same', activation=activation)(x_input)
	x = Conv2D(filters, kernel_size=(2,2),padding='same', activation=activation, strides=(2,2))(x)
	x = Conv2D(filters, kernel_size=kernel, padding='same',activation=activation,strides=1)(x)
	x = Conv2D(filters, kernel_size=kernel, padding='same', activation=activation,strides=1)(x)
	#flatten to map to latent dims
	flat = Flatten()(x)
	hidden = Dense(intermediate_dim, activation=activation)(flat)

	z_mean = Dense(latent_dim)(hidden)
	z_log_var = Dense(latent_dim)(hidden)

	#now for mapping to latent space
	z = Lambda(reparametrised_sample, output_shape=(latent_dim,))([z_mean, z_log_var])

	#okay, now for the decoder models, and this need to get renamed as they will be used multiple time

	
	decoder_hid = Dense(intermediate_dim, activation=activation)
	decoder_upsample=Dense(filters * intermediate_output_shape[1] * intermediate_output_shape[2], activation=activation)

	decoder_reshape=Reshape(intermediate_output_shape[1:])
	decoder_deconv1 = Conv2DTranspose(filters, kernel_size=kernel, padding='same',strides=1, activation=activation)
	decoder_deconv2 = Conv2DTranspose(filters, kernel_size=kernel, padding='same', strides=1, activation=activation)
	decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=kernel, strides=(2,2), padding='valid', activation=activation)

	decoder_mean_squash = Conv2D(channels, kernel_size=2, padding='valid', activation='sigmoid')

	#okay, now for first decoder
	y = decoder_hid(z)
	y = decoder_upsample(y)
	y=decoder_reshape(y)
	y=decoder_deconv1(y)
	y=decoder_deconv2(y)
	y = decoder_deconv3_upsamp(y)
	y = decoder_mean_squash(y)
	
	y= CustomVariationalLayer(z_mean, z_log_var)([x,y])

	vae= Model(x_input,y)
	#vae.compile(optimizer=optimizer, loss=None)
	vae.summary()

	#we build the encoder too
	encoder = Model(x_input, z_mean)

	#and the generator
	gen_input = Input(shape=(latent_dim,))
	gen = decoder_hid(gen_input)
	gen = decoder_upsample(gen)
	gen = decoder_reshape(gen)
	gen = decoder_deconv1(gen)
	gen = decoder_deconv2(gen)
	gen = decoder_deconv3(gen)
	gen = decoder_mean_squash(gen)
	generator = Model(gen_input, gen)

	#if we have a weightspath we save
	if weights_path:
		vae.save(weight_path + ".hdf5")

	#and return our three models easily from the function to get at them!
	return vae, encoder, decoder


