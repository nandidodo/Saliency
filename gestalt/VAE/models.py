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


def reparametrised_sample(z_mean, z_log_var):
	epsilon = K.random.normal(shape=K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
	return z_mean + K.exp(z_log_var) * epsilon

def DCVAE(input_shape, weights_path): # Deep convolutoinal VAE
	N, width, height,channels = input_shape

	x = Input(shape=input_shape)
	#convolutional encoder model
	x = Conv2D(channels, kernel_size=(2,2), padding='same', activation=activation)(x)
	x = Conv2D(filters, kernel_size=(2,2),padding='same', activation=activation, strides=(2,2))(x)
	x = Conv2D(filters, kernel_size=kernel, padding='same',activation=activation,strides=1)(x)
	x = Conv2D(filters, kernel_size=kernel, padding='same', activation=activationstrides=1)(x)
	#flatten to map to latent dims
	flat = Flatten()(x)
	hidden = Dense(intermediate_dim, activation=activation)(flat)

	z_mean = Dense(latent_dim)(hidden)
	z_log_var = Dense(latent_dim)(hidden)

	#now for mapping to latent space
	z = Lambda(reparametrised_sample, output_shape=(latent_dim,))(z_mean, z_log_var)

	#okay, now for the decoder models, and this need to get renamed as they will be used multiple time

	
	decoder_hid = Dense(intermediate_dim, activation=activation)
	decoder_upsample=Dense(filters * 14 * 14, activation=activation)

	decoder_reshape=Reshape(intermediate_output_shape[1:])
	decoder_deconv1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same',strides=1, activation=activation)
	decoder_deconv2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation=activation)
	decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=(3,3), strides=(2,2), padding='valid', activation=activation)

	decoder_mean_squash = Conv2D(img_channels, kernel_size=2, padding='valid', activation='sigmoid')

	#okay, now for first decoder
	y = decoder_hid(z)
	y = decoder_upsample(y)
	y=decoder_reshape(y)
	y=decoder_deconv1(y)
	y=decoder_deconv2(y)
	y = decoder_deconv3_upsamp(y)
	y = decoder_mean_squash(y)
	
	y= CustomVariationalLayer()([x,y])

	vae= Model(x,y)
	vae.compile(optimizer=optimizer, loss=None)
	vae.summary()

	#we build the encoder too
	encoder = Model(x, z_mean)

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

	#and return our three models easily from the function to get at them!
	return vae, encoder, decoder


