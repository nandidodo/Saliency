# This is where I put my keras model definitions for the autoencoders and stuff

import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

# we define model constants here - think of a better plce later
filters = 64
kernel = (3,3)
batch_size=64
latent_dim = 2
intermediate_dim = 124
epsilon_std = 1.0
activation ='relu'


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

